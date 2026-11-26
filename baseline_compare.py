"""naive vs mix 基线对比脚本。

为什么要有这个脚本
------------------
后续要做的「问题拆解 → 子问题检索 → 证据检查 → 有限补检索 → 汇总回答」属于多步检索，
判断它到底有没有价值，必须先有一个**单次检索的天花板**做对照组。这个脚本产出的就是那个对照组。

设计上的两个关键点
------------------
1. 语料按 `\\n\\n` 分段 + split_by_character_only=True，强制每段独立成一个 chunk。
   LightRAG 默认 chunk_token_size=1200，不显式分段的话整篇 5 段会挤进同一个 chunk，
   naive 和 mix 都会拿到全部信息，对比失去意义。
2. 参数（top_k / chunk_top_k）两种模式完全一致，只变 mode。
   差异只能来自「naive 只有 chunk 向量，mix 还有实体/关系向量 + 图谱遍历」这一条，
   否则就成了不公平对比。

用法
----
    .venv\\Scripts\\python.exe baseline_compare.py
    .venv\\Scripts\\python.exe baseline_compare.py --modes naive,mix --top-k 3 --chunk-top-k 3
    .venv\\Scripts\\python.exe baseline_compare.py --reset          # 清空后重建索引
    .venv\\Scripts\\python.exe baseline_compare.py --answer-modes mix # 只补跑某模式

输出
----
    baseline/results/<时间戳>/raw.json   完整返回（含 data / metadata / 答案）
    baseline/results/<时间戳>/report.md  人看的对比表
    baseline/results/latest/             指向最近一次，方便 diff
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from datetime import datetime
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

ROOT = Path(__file__).resolve().parent
BASELINE_DIR = ROOT / "baseline"
WORKING_DIR = ROOT / "rag_storage_baseline"
load_dotenv(ROOT / ".env.smoke", override=False)


# ── LLM / Embedding 接线（与 smoke_rag.py 保持一致，改一处要同步改另一处） ──
async def chat(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
):
    return await openai_complete_if_cache(
        os.environ["LLM_MODEL"],
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        **kwargs,
    )


def build_rag(enable_cache: bool = True) -> LightRAG:
    return LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=chat,
        llm_model_name=os.environ["LLM_MODEL"],
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.environ["EMBEDDING_DIM"]),
            max_token_size=8192,
            model_name=os.environ["EMBEDDING_MODEL"],
            func=partial(
                openai_embed.func,
                model=os.environ["EMBEDDING_MODEL"],
                api_key=os.environ["EMBEDDING_API_KEY"],
                base_url=os.environ["EMBEDDING_BASE_URL"],
            ),
        ),
        embedding_batch_num=10,
        llm_model_max_async=2,
        entity_extract_max_gleaning=0,
        # 关掉缓存才能保证每次都是真实调用，否则重复跑耗时会假性变低，
        # 两种模式被缓存命中的比例不同就变成不公平对比。
        enable_llm_cache=enable_cache,
        enable_llm_cache_for_entity_extract=enable_cache,
    )


# ── 入库幂等：同 file_path 已在 doc_status 里 processed 就跳过 ──
def already_ingested(file_path: str) -> bool:
    status_file = WORKING_DIR / "kv_store_doc_status.json"
    if not status_file.exists():
        return False
    try:
        records = json.loads(status_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    for rec in records.values():
        if rec.get("file_path") == file_path and rec.get("status") == "processed":
            return True
    return False


# ── 结果整理 ──
def summarize_data(data) -> dict:
    """统计检索证据的规模。只做一层展开，够看规模，不去猜具体 schema。"""
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        if isinstance(v, (list, dict)):
            out[k] = len(v)
        else:
            out[k] = f"{type(v).__name__}:{str(v)[:40]}"
    return out


def collect_contents(obj, acc: list) -> None:
    """递归捞回所有 content 字段 —— 用来判断「召回的是哪几段」。"""
    if isinstance(obj, dict):
        c = obj.get("content")
        if isinstance(c, str) and c.strip():
            acc.append(c.strip())
        for v in obj.values():
            collect_contents(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            collect_contents(v, acc)


def recalled_chunks(data) -> list:
    acc: list = []
    collect_contents(data, acc)
    seen, uniq = set(), []
    for c in acc:
        key = c[:60]
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def hit_keywords(answer: str, expects: list) -> list:
    """答案关键词命中 —— 只作粗筛。

    注意：这是纯子串匹配，遇到「并未提及周敏」这种否定句也会算命中，
    所以它不能单独用来判定对错，最终必须人工读一遍答案。
    """
    text = answer or ""
    return [e for e in expects if e in text]


def matched_paragraphs(recalled: list, paragraphs: list) -> list:
    """把召回到的 chunk 文本映射回语料段落索引。

    取每段前 20 字做指纹：chunk 是按段落切出来的，正常情况下应原样包含该前缀。
    """
    hit = []
    for i, p in enumerate(paragraphs):
        key = p[:20]
        if any(key in c for c in recalled):
            hit.append(i)
    return hit


def evidence_recall(matched: list, gold: list) -> dict:
    """证据召回率：gold 里的段落有几段被真正检索到。

    这是比答案关键词更硬核的指标 —— 证据没召回，答案即使蒙对也是运气。
    """
    if not gold:
        return {"recall": None, "covered": [], "missed": []}
    covered = [g for g in gold if g in matched]
    missed = [g for g in gold if g not in matched]
    return {
        "recall": round(len(covered) / len(gold), 2),
        "covered": covered,
        "missed": missed,
    }


# ── 主流程 ──
async def run(args) -> dict:
    cases = json.loads((BASELINE_DIR / "cases.json").read_text(encoding="utf-8"))
    corpus = cases["corpus"]
    sep = corpus["separator"]
    doc_text = sep.join(corpus["paragraphs"])
    file_path = corpus["file_path"]

    if args.reset and WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)

    rag = build_rag(enable_cache=not args.no_cache)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = BASELINE_DIR / "results" / stamp
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        await rag.initialize_storages()

        # ① 入库（幂等）
        if already_ingested(file_path):
            print(f"[ingest] {file_path} 已入库，跳过（加 --reset 可强制重建）")
            ingest = {"skipped": True}
        else:
            t0 = time.perf_counter()
            await rag.ainsert(
                doc_text,
                file_paths=[file_path],
                split_by_character=sep,
                split_by_character_only=True,
            )
            cost = time.perf_counter() - t0
            print(f"[ingest] 完成，耗时 {cost:.1f}s")
            ingest = {"skipped": False, "seconds": round(cost, 2)}

        # ② 逐题 × 逐模式
        records = []
        for q in cases["questions"]:
            for mode in args.modes:
                print(f"[query] {q['id']} mode={mode} ...", flush=True)
                t0 = time.perf_counter()
                try:
                    res = await rag.aquery_llm(
                        q["query"],
                        param=QueryParam(
                            mode=mode,
                            top_k=args.top_k,
                            chunk_top_k=args.chunk_top_k,
                            enable_rerank=False,
                            stream=False,
                        ),
                    )
                    elapsed = time.perf_counter() - t0
                    answer = (res.get("llm_response") or {}).get("content") or ""
                    data = res.get("data") or {}
                    recalled = recalled_chunks(data)
                    matched = matched_paragraphs(recalled, corpus["paragraphs"])
                    ev = evidence_recall(matched, q.get("gold") or [])
                    records.append(
                        {
                            "qid": q["id"],
                            "hop": q["hop"],
                            "mode": mode,
                            "query": q["query"],
                            "expect": q["expect"],
                            "hits": hit_keywords(answer, q["expect"]),
                            "answer": answer,
                            "seconds": round(elapsed, 2),
                            "status": res.get("status"),
                            "data_summary": summarize_data(data),
                            "recalled_chunks": recalled,
                            "evidence": ev,
                            "raw": res,
                        }
                    )
                    print(
                        f"        -> {round(elapsed, 1)}s  "
                        f"hits={len(records[-1]['hits'])}/{len(q['expect'])}  "
                        f"evidence={ev['recall']}  recall_n={len(recalled)}"
                    )
                except Exception as exc:  # 单题失败不该终止整轮对比
                    elapsed = time.perf_counter() - t0
                    records.append(
                        {
                            "qid": q["id"],
                            "hop": q["hop"],
                            "mode": mode,
                            "query": q["query"],
                            "expect": q["expect"],
                            "hits": [],
                            "answer": "",
                            "seconds": round(elapsed, 2),
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "data_summary": {},
                            "recalled_chunks": [],
                            "evidence": {"recall": None, "covered": [], "missed": []},
                            "raw": {},
                        }
                    )
                    print(f"        -> ERROR {type(exc).__name__}: {exc}")

        payload = {
            "timestamp": stamp,
            "params": {
                "modes": args.modes,
                "top_k": args.top_k,
                "chunk_top_k": args.chunk_top_k,
                "enable_rerank": False,
                "llm_cache": not args.no_cache,
                "llm_model": os.environ["LLM_MODEL"],
                "embedding_model": os.environ["EMBEDDING_MODEL"],
                "embedding_dim": int(os.environ["EMBEDDING_DIM"]),
                "chunk_paragraphs": len(corpus["paragraphs"]),
            },
            "ingest": ingest,
            "records": records,
        }

        (result_dir / "raw.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (result_dir / "report.md").write_text(
            render_report(payload, cases), encoding="utf-8"
        )

        latest = BASELINE_DIR / "results" / "latest"
        if latest.exists():
            shutil.rmtree(latest)
        shutil.copytree(result_dir, latest)

        print(f"\n[out] {result_dir}")
        return payload

    finally:
        await rag.finalize_storages()


def render_report(payload: dict, cases: dict) -> str:
    recs = payload["records"]
    p = payload["params"]
    lines = []
    lines.append("# naive vs mix 基线对比\n")
    lines.append(f"- 生成时间：{payload['timestamp']}")
    lines.append(
        f"- 模型：LLM `{p['llm_model']}` / Embedding `{p['embedding_model']}`"
        f"（{p['embedding_dim']} 维）"
    )
    lines.append(f"- 参数：top_k={p['top_k']}, chunk_top_k={p['chunk_top_k']}, rerank=off")
    lines.append(f"- LLM 缓存：{'关（每次真实调用）' if not p.get('llm_cache', True) else '开（可能命中旧答案，耗时仅供参考）'}")
    lines.append(f"- 语料：{p['chunk_paragraphs']} 段 → 每段独立成 chunk")
    if payload["ingest"].get("skipped"):
        lines.append("- 入库：跳过（复用已有索引）")
    else:
        lines.append(f"- 入库：{payload['ingest'].get('seconds')}s")
    lines.append("")

    # 得分总表
    lines.append("## 得分总表\n")
    lines.append(
        "> 「证据召回」= gold 证据段里有几段真被检索到，是硬指标。"
        "「要点命中」是答案关键词的**子串匹配，只作粗筛** —— "
        "答案里出现「并未提及周敏」也会算命中，必须人工读答案定论。\n"
    )
    lines.append("| 问题 | 跳数 | 模式 | 证据召回 | 要点命中(粗筛) | 耗时 | 召回块数 | 状态 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in recs:
        ev = r.get("evidence") or {}
        ev_txt = "n/a" if ev.get("recall") is None else f"{ev['recall'] * 100:.0f}%"
        lines.append(
            f"| {r['qid']} | {r['hop']} | {r['mode']} | {ev_txt} | "
            f"{len(r['hits'])}/{len(r['expect'])} | {r['seconds']}s | "
            f"{len(r['recalled_chunks'])} | {r['status']} |"
        )
    lines.append("")

    # 按模式的命中率汇总
    lines.append("## 按模式汇总\n")
    lines.append("| 模式 | 证据召回率 | 要点命中率(粗筛) | 平均耗时 |")
    lines.append("|---|---|---|---|")
    modes = sorted({r["mode"] for r in recs})
    for m in modes:
        sub = [r for r in recs if r["mode"] == m]
        evs = [r["evidence"]["recall"] for r in sub if (r.get("evidence") or {}).get("recall") is not None]
        ev_avg = f"{sum(evs) / len(evs) * 100:.0f}%" if evs else "-"
        tot = sum(len(r["expect"]) for r in sub)
        got = sum(len(r["hits"]) for r in sub)
        avg = round(sum(r["seconds"] for r in sub) / max(len(sub), 1), 2)
        rate = f"{got / tot * 100:.0f}%" if tot else "-"
        lines.append(f"| {m} | {ev_avg} | {rate} | {avg}s |")
    lines.append("")

    # 逐题详情
    lines.append("## 逐题详情\n")
    notes = {q["id"]: q.get("note", "") for q in cases["questions"]}
    for q in cases["questions"]:
        gold = q.get("gold") or []
        lines.append(f"### {q['id']}（{q['hop']} 跳）\n")
        lines.append(f"**问**：{q['query']}\n")
        lines.append(f"> {notes.get(q['id'], '')}\n")
        lines.append(f"**期望要点**：{'、'.join(q['expect'])}")
        lines.append(f"**gold 证据段**：{gold if gold else '未标注'}\n")
        for r in [x for x in recs if x["qid"] == q["id"]]:
            ev = r.get("evidence") or {}
            full = len(r["hits"]) == len(r["expect"])
            mark = "✅" if full else ("🟡" if r["hits"] else "❌")
            lines.append(
                f"**{r['mode']}** {mark} — {r['seconds']}s，"
                f"证据召回 {ev.get('recall')}（覆盖 {ev.get('covered')}，漏 {ev.get('missed')}），"
                f"要点命中 {'、'.join(r['hits']) or '无'}"
            )
            if r.get("error"):
                lines.append(f"\n```\n{r['error']}\n```\n")
                continue
            lines.append(f"\n<details><summary>答案</summary>\n")
            lines.append("```text")
            lines.append(r["answer"].strip() or "(空)")
            lines.append("```")
            lines.append("</details>\n")
            if r["recalled_chunks"]:
                lines.append(f"<details><summary>召回的块（{len(r['recalled_chunks'])}）</summary>\n")
                for i, c in enumerate(r["recalled_chunks"], 1):
                    lines.append(f"{i}. {c[:80]}")
                lines.append("\n</details>\n")
        lines.append("---\n")

    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--modes",
        default="naive,mix",
        help="逗号分隔的检索模式，默认 naive,mix",
    )
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--chunk-top-k", type=int, default=3)
    ap.add_argument("--reset", action="store_true", help="清空 rag_storage_baseline 后重建")
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="关闭 LLM 缓存，保证每次真实调用（对比耗时时建议开）",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    asyncio.run(run(args))

"""Retrieve source evidence for a previously generated question plan."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .decompose import QuestionPlan


async def retrieve_plan(
    plan: QuestionPlan,
    query: Callable[[str], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Keep per-question outcomes and deduplicate evidence by chunk ID."""
    started = time.perf_counter()
    outcomes = []
    merged = {}
    for index, question in enumerate(plan.subquestions, start=1):
        try:
            result = await query(question)
        except Exception as exc:
            outcomes.append({
                "index": index, "question": question, "status": "error",
                "error": type(exc).__name__, "chunks": [],
            })
            continue

        data = result.get("data") or {}
        chunks = [
            chunk for chunk in data.get("chunks", [])
            if chunk.get("chunk_id") and chunk.get("content")
        ] if result.get("status") == "success" else []
        reason = (result.get("metadata") or {}).get("failure_reason")
        if result.get("status") == "success" or reason == "no_results":
            status = "retrieved" if chunks else "empty"
        else:
            status = "error"
        outcomes.append({
            "index": index, "question": question, "status": status,
            "chunks": chunks, "references": data.get("references", []),
        })
        for chunk in chunks:
            record = merged.setdefault(chunk["chunk_id"], {
                "chunk_id": chunk["chunk_id"], "content": chunk["content"],
                "file_path": chunk.get("file_path"),
                "subquestion_indexes": [],
            })
            if index not in record["subquestion_indexes"]:
                record["subquestion_indexes"].append(index)

    return {
        "plan": plan.model_dump(),
        "subquestion_results": outcomes,
        "merged_chunks": list(merged.values()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


async def run(plan: QuestionPlan, args: argparse.Namespace) -> dict[str, Any]:
    from baseline_compare import WORKING_DIR, build_rag
    from lightrag import QueryParam

    if not (WORKING_DIR / "vdb_chunks.json").is_file():
        raise FileNotFoundError(f"Baseline index is missing: {WORKING_DIR}")
    rag = build_rag(enable_cache=False)
    await rag.initialize_storages()
    try:
        async def query(question: str) -> dict[str, Any]:
            return await rag.aquery_data(
                question,
                param=QueryParam(
                    mode=args.mode, top_k=args.top_k,
                    chunk_top_k=args.chunk_top_k, enable_rerank=False,
                ),
            )

        result = await retrieve_plan(plan, query)
        result["retrieval"] = {
            "mode": args.mode, "top_k": args.top_k,
            "chunk_top_k": args.chunk_top_k, "rerank": False,
        }
        return result
    finally:
        await rag.finalize_storages()


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve evidence for a saved plan")
    parser.add_argument("plan_file", type=Path)
    parser.add_argument("--mode", choices=("naive", "mix"), default="naive")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.top_k, args.chunk_top_k) < 1:
        parser.error("Retrieval limits must be positive")
    if args.output.exists():
        parser.error("--output already exists; choose a new path")
    try:
        payload = json.loads(args.plan_file.read_text(encoding="utf-8"))
        plan = QuestionPlan.model_validate(payload["plan"])
        if any(not q.strip() or len(q) > 4000 for q in plan.subquestions):
            raise ValueError("Invalid subquestion length")
        if plan.strategy == "direct" and plan.subquestions != [plan.original_question]:
            raise ValueError("A direct plan must preserve the original question")
        if plan.strategy == "decompose" and len(plan.subquestions) < 2:
            raise ValueError("A decomposed plan requires at least two subquestions")
        result = asyncio.run(run(plan, args))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Saved {len(result['merged_chunks'])} unique chunks to {args.output}")
        for outcome in result["subquestion_results"]:
            print(f"Q{outcome['index']}: {outcome['status']} ({len(outcome['chunks'])} chunks)")
        return int(any(item["status"] == "error" for item in result["subquestion_results"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Retrieval failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

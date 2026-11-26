import asyncio
import json
import os
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env.smoke", override=False)


async def chat(
    prompt, system_prompt=None, history_messages=None,
    keyword_extraction=False, **kwargs
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


async def main():
    rag = LightRAG(
        working_dir=str(ROOT / "rag_storage_smoke_v2"),
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
    )

    try:
        await rag.initialize_storages()

        # ① 入库：文本块、向量、实体关系及图谱。
        await rag.ainsert(
            "星河公司开发了产品A。产品A支持离线部署。"
            "远航公司开发了产品B。产品B仅支持在线服务，不支持离线部署。",
            file_paths=["demo.txt"],
        )

        # ② 一次查询，同时取得证据和模型答案。
        result = await rag.aquery_llm(
            "产品A和产品B，哪个支持离线部署？",
            param=QueryParam(
                mode="naive",
                top_k=5,
                chunk_top_k=5,
                enable_rerank=False,
                stream=False,
            ),
        )

        print("\n===== 检索证据 =====")
        print(json.dumps(result.get("data", {}), ensure_ascii=False, indent=2))

        print("\n===== 模型答案 =====")
        print(result.get("llm_response", {}).get("content", ""))

    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
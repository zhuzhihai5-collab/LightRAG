"""Run from the repository root with python -m complex_qa."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

from .decompose import DecompositionError, QuestionDecomposer

DEMO_QUESTION = "产品A和产品B，哪个支持离线部署，各有什么限制？"
DEMO_REPLY = {
    "strategy": "decompose",
    "subquestions": [
        "产品A是否支持离线部署？离线部署有什么限制？",
        "产品B是否支持离线部署？离线部署有什么限制？",
    ],
    "reason": "分别检索两个产品的离线部署能力与限制，再进行比较。",
}


async def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    usage = None
    if args.demo:
        async def complete(prompt: str, **kwargs) -> str:
            return json.dumps(DEMO_REPLY, ensure_ascii=False)

        question = DEMO_QUESTION
        model = None
    else:
        from dotenv import load_dotenv
        from openai import AsyncOpenAI

        root = Path(__file__).resolve().parents[1]
        load_dotenv(root / ".env.smoke", override=False)
        required = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
        missing = [name for name in required if not os.environ.get(name, "").strip()]
        if missing:
            raise DecompositionError("Missing configuration: " + ", ".join(missing))
        question = args.question
        model = os.environ["LLM_MODEL"]

        async def complete(prompt: str, *, system_prompt: str) -> str:
            nonlocal usage
            async with AsyncOpenAI(
                api_key=os.environ["LLM_API_KEY"],
                base_url=os.environ["LLM_BASE_URL"],
                max_retries=0,
                timeout=args.timeout,
            ) as client:
                reply = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=1500,
                )
            usage = reply.usage.model_dump() if reply.usage else None
            if not reply.choices:
                raise DecompositionError("The model returned no choices")
            return reply.choices[0].message.content or ""

    planner = QuestionDecomposer(
        complete,
        max_subquestions=args.max_subquestions,
        timeout_seconds=args.timeout,
    )
    plan = await planner.decompose(question)
    return {
        "source": "offline_fixture" if args.demo else "live_model",
        "model": model,
        "plan": plan.model_dump(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "usage": usage,
        "validation": "structural_only; semantic faithfulness requires review",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Decompose a question before retrieval")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--demo", action="store_true", help="Use a fixed offline example")
    parser.add_argument("--max-subquestions", type=int, choices=range(2, 5), default=4)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, help="Save JSON to a new file; never overwrite")
    args = parser.parse_args()
    if args.demo and args.question is not None:
        parser.error("--demo uses its bundled question; omit the positional question")
    if not args.demo and not (args.question and args.question.strip()):
        parser.error("Provide a question or use --demo")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.output and args.output.exists():
        parser.error("--output already exists; choose a new path")
    try:
        result = asyncio.run(run(args))
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
        print(rendered)
    except (DecompositionError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

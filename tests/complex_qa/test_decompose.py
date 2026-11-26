"""Offline contract tests for the application-level planner."""

import asyncio
import json
import unittest

from complex_qa import DecompositionError, QuestionDecomposer


def response(strategy="decompose", questions=None, **extra):
    return json.dumps({
        "strategy": strategy,
        "subquestions": questions if questions is not None else ["A支持离线部署吗？", "B支持离线部署吗？"],
        "reason": "分别检索两个对象。",
        **extra,
    }, ensure_ascii=False)


class PlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_comparison_preserves_question_and_calls_once(self):
        calls = []

        async def complete(prompt, **kwargs):
            calls.append((prompt, kwargs))
            return response()

        question = '比较A和B的离线部署能力，限定"社区版"。'
        plan = await QuestionDecomposer(complete).decompose(question)
        self.assertEqual(plan.original_question, question)
        self.assertEqual(len(plan.subquestions), 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0][0].split("\n", 1)[1]), question)
        self.assertIn("independently", calls[0][1]["system_prompt"])

    async def test_direct_never_loses_original_conditions(self):
        async def complete(*args, **kwargs):
            return response("direct", ["A支持部署吗？"])

        question = "A社区版是否支持无公网环境下的离线部署？"
        plan = await QuestionDecomposer(complete).decompose(question)
        self.assertEqual(plan.subquestions, [question])

    async def test_empty_question_does_not_call_model(self):
        async def complete(*args, **kwargs):
            self.fail("The model must not receive an empty question")

        with self.assertRaises(DecompositionError):
            await QuestionDecomposer(complete).decompose("  ")

    async def test_oversized_question_does_not_call_model(self):
        async def complete(*args, **kwargs):
            self.fail("The model must not receive oversized input")

        with self.assertRaises(DecompositionError):
            await QuestionDecomposer(complete).decompose("问" * 4001)

    async def test_provider_failure_is_explicit_and_redacted(self):
        async def complete(*args, **kwargs):
            raise RuntimeError("private provider detail")

        with self.assertRaises(DecompositionError) as caught:
            await QuestionDecomposer(complete).decompose("问题")
        self.assertNotIn("private provider detail", str(caught.exception))

    async def test_timeout_does_not_produce_a_fake_plan(self):
        async def complete(*args, **kwargs):
            await asyncio.sleep(1)
            return response()

        with self.assertRaises(DecompositionError):
            await QuestionDecomposer(complete, timeout_seconds=0.01).decompose("问题")

    async def test_cancellation_propagates(self):
        async def complete(*args, **kwargs):
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await QuestionDecomposer(complete).decompose("问题")


class SchemaTests(unittest.TestCase):
    def setUp(self):
        async def unused(*args, **kwargs):
            raise AssertionError("Parsing must not call the model")

        self.planner = QuestionDecomposer(unused)

    def test_json_code_fence_is_supported(self):
        plan = self.planner.parse("比较A和B", "```json\n" + response() + "\n```")
        self.assertEqual(plan.strategy, "decompose")

    def test_bad_json_is_rejected(self):
        for raw in ("not JSON", "[]", "{}", "解释：" + response(), "x" * 20001):
            with self.subTest(raw=raw[:30]), self.assertRaises(DecompositionError):
                self.planner.parse("问题", raw)

    def test_unexpected_fields_are_rejected(self):
        with self.assertRaises(DecompositionError):
            self.planner.parse("问题", response(answer="unsupported fact"))

    def test_wrong_question_type_is_rejected(self):
        with self.assertRaises(DecompositionError):
            self.planner.parse("问题", response(questions=[123, "B？"]))

    def test_count_limits_are_enforced(self):
        for raw in (response(questions=["A？"]), response("direct", ["A？", "B？"]),
                    response(questions=["A？", "B？", "C？", "D？", "E？"])):
            with self.subTest(raw=raw), self.assertRaises(DecompositionError):
                self.planner.parse("问题", raw)

    def test_duplicates_and_blank_questions_are_rejected(self):
        for questions in (["A？", " A？ "], [" ", "B？"]):
            with self.subTest(questions=questions), self.assertRaises(DecompositionError):
                self.planner.parse("问题", response(questions=questions))

    def test_custom_question_limit_is_enforced(self):
        self.planner.max_subquestions = 2
        with self.assertRaises(DecompositionError):
            self.planner.parse("问题", response(questions=["A？", "B？", "C？"]))

    def test_bad_configuration_is_rejected(self):
        for count in (0, 1, 5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                QuestionDecomposer(None, max_subquestions=count)
        with self.assertRaises(ValueError):
            QuestionDecomposer(None, timeout_seconds=0)


if __name__ == "__main__":
    unittest.main()

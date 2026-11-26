"""Check evidence provenance and retrieval failures without external calls."""

import unittest

from complex_qa import QuestionPlan
from complex_qa.retrieval import retrieve_plan


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.plan = QuestionPlan(
            original_question="Compare A and B",
            strategy="decompose",
            subquestions=["Check A", "Check B"],
            reason="Independent questions",
        )

    async def test_shared_chunk_retains_both_questions(self):
        async def query(question):
            return {
                "status": "success",
                "data": {
                    "chunks": [
                        {"chunk_id": "shared", "content": "A and B", "file_path": "docs.txt"},
                        {"chunk_id": question, "content": question},
                    ],
                    "references": [{"file_path": "docs.txt"}],
                },
            }

        result = await retrieve_plan(self.plan, query)
        self.assertEqual(len(result["merged_chunks"]), 3)
        self.assertEqual(result["merged_chunks"][0]["subquestion_indexes"], [1, 2])
        self.assertEqual(result["subquestion_results"][1]["references"], [{"file_path": "docs.txt"}])

    async def test_failure_and_empty_outcomes_are_distinct(self):
        async def query(question):
            if question == "Check A":
                raise TimeoutError("provider detail")
            return {"status": "failure", "metadata": {"failure_reason": "no_results"}}

        result = await retrieve_plan(self.plan, query)
        self.assertEqual(
            [item["status"] for item in result["subquestion_results"]],
            ["error", "empty"],
        )
        self.assertEqual(result["merged_chunks"], [])
        self.assertEqual(result["subquestion_results"][0]["error"], "TimeoutError")

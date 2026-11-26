import json
import copy
import unittest

from complex_qa.answer import answer_retrieval


class AnswerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.retrieval = {
            "plan": {"original_question": "A and B?"},
            "subquestion_results": [
                {"index": 1, "question": "A?", "status": "retrieved",
                 "chunks": [{"chunk_id": "a", "content": "A works"}]},
                {"index": 2, "question": "B?", "status": "retrieved",
                 "chunks": [{"chunk_id": "b", "content": "B fails"}]},
            ],
            "merged_chunks": [
                {"chunk_id": "a", "content": "A works", "file_path": "a.txt"},
                {"chunk_id": "b", "content": "B fails", "file_path": "b.txt"},
            ],
        }

    async def test_checks_each_question_and_generates_cited_answers(self):
        replies = iter([
            {"supported": True, "evidence_ids": ["a"], "reason": "A works"},
            {"supported": True, "evidence_ids": ["b"], "reason": "B fails"},
            {"answers": [{"index": 1, "answer": "A works", "evidence_ids": ["a"]},
                         {"index": 2, "answer": "B fails", "evidence_ids": ["b"]}]},
        ])
        async def complete(*args, **kwargs):
            return json.dumps(next(replies))
        result = await answer_retrieval(self.retrieval, complete)
        self.assertEqual(len(result["answers"]), 2)
        self.assertEqual(result["answers"][1]["evidence_ids"], ["b"])

    async def test_empty_retrieval_does_not_trigger_model(self):
        for item in self.retrieval["subquestion_results"]:
            item["status"] = "empty"
            item["chunks"] = []
        async def complete(*args, **kwargs):
            self.fail("Model must not be called for empty evidence")
        result = await answer_retrieval(self.retrieval, complete)
        self.assertEqual(result["answers"], [])
        self.assertEqual([x["status"] for x in result["checks"]], ["insufficient", "insufficient"])

    async def test_rejects_cross_question_citation(self):
        replies = iter([
            {"supported": True, "evidence_ids": ["a"], "reason": "yes"},
            {"supported": True, "evidence_ids": ["b"], "reason": "yes"},
            {"answers": [{"index": 1, "answer": "A", "evidence_ids": ["b"]},
                         {"index": 2, "answer": "B", "evidence_ids": ["b"]}]},
        ])
        async def complete(*args, **kwargs):
            return json.dumps(next(replies))
        with self.assertRaises(ValueError):
            await answer_retrieval(self.retrieval, complete)

    async def test_partial_support_keeps_unanswered_question(self):
        self.retrieval["subquestion_results"][1]["status"] = "error"
        replies = iter([
            {"supported": True, "evidence_ids": ["a"], "reason": "yes"},
            {"answers": [{"index": 1, "answer": "A works", "evidence_ids": ["a"]}]},
        ])
        async def complete(*args, **kwargs):
            return json.dumps(next(replies))
        result = await answer_retrieval(self.retrieval, complete)
        self.assertEqual(result["checks"][1]["status"], "retrieval_error")
        self.assertEqual(len(result["answers"]), 1)

    async def test_supplement_rechecks_once_and_does_not_mutate_input(self):
        self.retrieval["subquestion_results"] = self.retrieval["subquestion_results"][:1]
        original = copy.deepcopy(self.retrieval)
        replies = iter([
            {"supported": False, "evidence_ids": ["a"], "reason": "B missing", "missing_query": "B?"},
            {"supported": True, "evidence_ids": ["a", "b"], "reason": "both found"},
            {"answers": [{"index": 1, "answer": "A works and B fails", "evidence_ids": ["a", "b"]}]},
        ])
        calls = []
        async def complete(*args, **kwargs):
            return json.dumps(next(replies))
        async def query(q):
            calls.append(q)
            return {"status": "success", "data": {"chunks": [{"chunk_id": "b", "content": "B fails"}]}}
        result = await answer_retrieval(self.retrieval, complete, query)
        self.assertEqual(calls, ["B?"])
        self.assertEqual(result["initial_checks"][0]["status"], "partial")
        self.assertEqual(result["checks"][0]["status"], "supported")
        self.assertEqual(result["answers"][0]["evidence_ids"], ["a", "b"])
        self.assertEqual(self.retrieval, original)

    async def test_failed_supplement_preserves_partial_answer(self):
        self.retrieval["subquestion_results"] = self.retrieval["subquestion_results"][:1]
        replies = iter([
            {"supported": False, "evidence_ids": ["a"], "reason": "B missing", "missing_query": "B?"},
            {"answers": [{"index": 1, "answer": "A works; B unknown", "evidence_ids": ["a"]}]},
        ])
        async def complete(*args, **kwargs):
            return json.dumps(next(replies))
        async def query(q):
            raise TimeoutError()
        result = await answer_retrieval(self.retrieval, complete, query)
        self.assertEqual(result["supplemental_retrieval"][0]["status"], "error")
        self.assertEqual(result["checks"][0]["status"], "partial")
        self.assertIn("A works", result["final_answer"])
        self.assertIn("B missing", result["final_answer"])


if __name__ == "__main__":
    unittest.main()

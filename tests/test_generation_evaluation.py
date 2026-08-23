"""
tests/test_generation_evaluation.py

Phase 3 unit tests for the Answer Accuracy Evaluation Engine.
All tests use mocked production components — no real Ollama or ChromaDB calls.
"""

import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.Evaluation_Engine.answer_accuracy_evaluator import (
    AnswerAccuracyEvaluator,
    classify_abstention,
    classify_accuracy,
    classify_failure_mode,
    deterministic_fact_check
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_evaluator(judge_responses=None, generate_responses=None):
    """
    Build a fully mocked AnswerAccuracyEvaluator.
    
    judge_responses: list of raw answer strings that the generation_engine
                     returns per successive .generate() call.  The first N-K
                     calls are RAG answer generation; the final calls are judge
                     invocations.
    """
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 384

    retriever = MagicMock()
    retriever.retrieve_and_gate.return_value = {
        "results": [
            {
                "chunk_id": "chunk_0",
                "text": "Bhupender has a B.Tech in AI and Data Science from VIPS.",
                "score": 0.85,
                "metadata": {"file_name": "test.txt"}
            }
        ]
    }

    context_builder = MagicMock()
    context_builder.build.return_value = {"prompt": "CONTEXT... Q: ..."}

    generation_engine = MagicMock()
    if generate_responses is not None:
        generation_engine.generate.side_effect = [
            {"answer": r} for r in generate_responses
        ]

    return AnswerAccuracyEvaluator(
        embedder=embedder,
        retriever=retriever,
        context_builder=context_builder,
        generation_engine=generation_engine
    )


# ---------------------------------------------------------------------------
# Test 1: All expected facts supported → CORRECT
# ---------------------------------------------------------------------------
class TestAllFactsSupported(unittest.TestCase):
    def test_correct_classification(self):
        """All expected facts present in judge output → CORRECT."""
        judge_json = json.dumps({
            "facts": [
                {"fact": "B.Tech in AI and Data Science", "supported": True},
                {"fact": "VIPS", "supported": True}
            ]
        })
        # 1st call: RAG answer generation; 2nd call: judge
        ev = make_evaluator(generate_responses=[
            "Bhupender has a B.Tech in AI and Data Science from VIPS.",
            judge_json
        ])
        result = ev.evaluate([{
            "test_id": "q001",
            "question": "What is Bhupender's degree?",
            "answerable": True,
            "expected_sources": ["test.txt"],
            "expected_facts": ["B.Tech in AI and Data Science", "VIPS"]
        }])
        q = result["per_query"][0]
        self.assertEqual(q["accuracy_class"], "CORRECT")
        self.assertEqual(q["facts_supported"], 2)
        self.assertEqual(q["facts_expected"], 2)


# ---------------------------------------------------------------------------
# Test 2: Some expected facts supported → PARTIALLY_CORRECT
# ---------------------------------------------------------------------------
class TestSomeFactsSupported(unittest.TestCase):
    def test_partially_correct_classification(self):
        """One of two expected facts present → PARTIALLY_CORRECT."""
        judge_json = json.dumps({
            "facts": [
                {"fact": "B.Tech in AI and Data Science", "supported": True},
                {"fact": "graduated 2025", "supported": False}
            ]
        })
        ev = make_evaluator(generate_responses=[
            "Bhupender completed a B.Tech in AI and Data Science.",
            judge_json
        ])
        result = ev.evaluate([{
            "test_id": "q002",
            "question": "What is Bhupender's degree and graduation year?",
            "answerable": True,
            "expected_sources": ["test.txt"],
            "expected_facts": ["B.Tech in AI and Data Science", "graduated 2025"]
        }])
        q = result["per_query"][0]
        self.assertEqual(q["accuracy_class"], "PARTIALLY_CORRECT")
        self.assertEqual(q["facts_supported"], 1)


# ---------------------------------------------------------------------------
# Test 3: No expected facts supported → INCORRECT
# ---------------------------------------------------------------------------
class TestNoFactsSupported(unittest.TestCase):
    def test_incorrect_classification(self):
        """No expected facts in answer → INCORRECT."""
        judge_json = json.dumps({
            "facts": [
                {"fact": "B.Tech in AI and Data Science", "supported": False},
                {"fact": "VIPS", "supported": False}
            ]
        })
        ev = make_evaluator(generate_responses=[
            "Bhupender studied literature.",
            judge_json
        ])
        result = ev.evaluate([{
            "test_id": "q003",
            "question": "What is Bhupender's degree?",
            "answerable": True,
            "expected_sources": ["test.txt"],
            "expected_facts": ["B.Tech in AI and Data Science", "VIPS"]
        }])
        q = result["per_query"][0]
        self.assertEqual(q["accuracy_class"], "INCORRECT")
        self.assertEqual(q["facts_supported"], 0)


# ---------------------------------------------------------------------------
# Test 4: Correct unanswerable refusal → CORRECT_ABSTENTION
# ---------------------------------------------------------------------------
class TestCorrectAbstention(unittest.TestCase):
    def test_correct_abstention(self):
        """Answer clearly refuses → CORRECT_ABSTENTION."""
        ev = make_evaluator(generate_responses=[
            "I couldn't find relevant information about his salary in the uploaded documents."
        ])
        # Unanswerable — retriever returns no chunks
        ev.retriever.retrieve_and_gate.return_value = {"results": []}
        result = ev.evaluate([{
            "test_id": "q004",
            "question": "What is Bhupender's salary?",
            "answerable": False,
            "expected_sources": [],
            "expected_facts": []
        }])
        q = result["per_query"][0]
        self.assertEqual(q["accuracy_class"], "CORRECT_ABSTENTION")
        self.assertFalse(q["answerable"])


# ---------------------------------------------------------------------------
# Test 5: Unsupported claim for unanswerable question → HALLUCINATED_ANSWER
# ---------------------------------------------------------------------------
class TestHallucinatedAnswer(unittest.TestCase):
    def test_hallucinated_answer(self):
        """Answer makes a specific factual claim → HALLUCINATED_ANSWER."""
        ev = make_evaluator(generate_responses=[
            "His salary is ₹20 LPA at JLL."
        ])
        ev.retriever.retrieve_and_gate.return_value = {"results": []}
        result = ev.evaluate([{
            "test_id": "q005",
            "question": "What is Bhupender's salary?",
            "answerable": False,
            "expected_sources": [],
            "expected_facts": []
        }])
        q = result["per_query"][0]
        self.assertEqual(q["accuracy_class"], "HALLUCINATED_ANSWER")


# ---------------------------------------------------------------------------
# Test 6: Malformed judge JSON → fallback to deterministic matching
# ---------------------------------------------------------------------------
class TestMalformedJudgeJson(unittest.TestCase):
    def test_deterministic_fallback(self):
        """Judge returns invalid JSON twice → deterministic fallback is used."""
        ev = make_evaluator(generate_responses=[
            # RAG answer
            "Bhupender has a B.Tech in AI and Data Science from VIPS.",
            # Judge response 1 (invalid)
            "Here are the facts: yes no maybe {broken json]",
            # Judge response 2 (still invalid)
            "I am unable to parse this correctly"
        ])
        result = ev.evaluate([{
            "test_id": "q006",
            "question": "What is Bhupender's degree?",
            "answerable": True,
            "expected_sources": ["test.txt"],
            "expected_facts": ["B.Tech in AI and Data Science", "VIPS"]
        }])
        q = result["per_query"][0]
        self.assertEqual(q["evaluation_method"], "DETERMINISTIC_FALLBACK")
        self.assertIn("accuracy_class", q)


# ---------------------------------------------------------------------------
# Test 7: Retrieval success + wrong answer → GENERATION_FAILURE
# ---------------------------------------------------------------------------
class TestGenerationFailure(unittest.TestCase):
    def test_generation_failure_classification(self):
        """Correct chunk retrieved but answer is wrong → GENERATION_FAILURE."""
        judge_json = json.dumps({
            "facts": [
                {"fact": "B.Tech in AI and Data Science", "supported": False}
            ]
        })
        ev = make_evaluator(generate_responses=[
            "He studied philosophy at Delhi University.",
            judge_json
        ])
        result = ev.evaluate([{
            "test_id": "q007",
            "question": "What is Bhupender's degree?",
            "answerable": True,
            "expected_sources": ["test.txt"],
            "expected_facts": ["B.Tech in AI and Data Science"]
        }])
        q = result["per_query"][0]
        self.assertTrue(q["retrieval_success"])
        self.assertEqual(q["accuracy_class"], "INCORRECT")
        self.assertEqual(q["failure_mode"], "GENERATION_FAILURE")


# ---------------------------------------------------------------------------
# Test 8: Retrieval failure + wrong answer → RETRIEVAL_FAILURE
# ---------------------------------------------------------------------------
class TestRetrievalFailure(unittest.TestCase):
    def test_retrieval_failure_classification(self):
        """No matching source retrieved, wrong answer → RETRIEVAL_FAILURE."""
        judge_json = json.dumps({
            "facts": [
                {"fact": "B.Tech in AI and Data Science", "supported": False}
            ]
        })
        ev = make_evaluator(generate_responses=[
            "He graduated from a local university.",
            judge_json
        ])
        # Override retriever to return wrong source
        ev.retriever.retrieve_and_gate.return_value = {
            "results": [
                {
                    "chunk_id": "chunk_x",
                    "text": "Some unrelated text.",
                    "score": 0.72,
                    "metadata": {"file_name": "other_doc.txt"}  # wrong source
                }
            ]
        }
        result = ev.evaluate([{
            "test_id": "q008",
            "question": "What is Bhupender's degree?",
            "answerable": True,
            "expected_sources": ["test.txt"],
            "expected_facts": ["B.Tech in AI and Data Science"]
        }])
        q = result["per_query"][0]
        self.assertFalse(q["retrieval_success"])
        self.assertEqual(q["failure_mode"], "RETRIEVAL_FAILURE")


# ---------------------------------------------------------------------------
# Test 9: Empty generated answer is handled gracefully
# ---------------------------------------------------------------------------
class TestEmptyGeneratedAnswer(unittest.TestCase):
    def test_empty_answer_handled(self):
        """An empty generated answer should not crash; fact matching should fail gracefully."""
        judge_json = json.dumps({
            "facts": [
                {"fact": "B.Tech in AI and Data Science", "supported": False}
            ]
        })
        ev = make_evaluator(generate_responses=["", judge_json])
        result = ev.evaluate([{
            "test_id": "q009",
            "question": "What is Bhupender's degree?",
            "answerable": True,
            "expected_sources": ["test.txt"],
            "expected_facts": ["B.Tech in AI and Data Science"]
        }])
        q = result["per_query"][0]
        self.assertIsNotNone(q["accuracy_class"])
        self.assertIn(q["accuracy_class"], ["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"])


# ---------------------------------------------------------------------------
# Test 10: Missing expected_facts (empty list) handled gracefully
# ---------------------------------------------------------------------------
class TestMissingExpectedFacts(unittest.TestCase):
    def test_empty_expected_facts(self):
        """Empty expected_facts for an answerable query → marked CORRECT if answer non-empty."""
        ev = make_evaluator(generate_responses=[
            "Bhupender is an AI/ML engineer."
        ])
        result = ev.evaluate([{
            "test_id": "q010",
            "question": "Tell me about Bhupender.",
            "answerable": True,
            "expected_sources": ["test.txt"],
            "expected_facts": []
        }])
        q = result["per_query"][0]
        # Should not crash; fact_results is empty, accuracy_class defaults CORRECT
        self.assertEqual(q["accuracy_class"], "CORRECT")
        self.assertEqual(q["facts_expected"], 0)
        self.assertEqual(q["evaluation_method"], "NO_FACTS")


# ---------------------------------------------------------------------------
# Helper Function Unit Tests
# ---------------------------------------------------------------------------

class TestClassifyAbstention(unittest.TestCase):
    def test_refusal_phrase(self):
        self.assertEqual(
            classify_abstention("I couldn't find relevant information about that."),
            "CORRECT_ABSTENTION"
        )

    def test_hallucination(self):
        self.assertEqual(
            classify_abstention("His salary is ₹20 LPA."),
            "HALLUCINATED_ANSWER"
        )

    def test_empty_answer(self):
        self.assertEqual(classify_abstention(""), "UNSAFE_ABSTENTION")


class TestClassifyAccuracy(unittest.TestCase):
    def test_all_correct(self):
        self.assertEqual(classify_accuracy(3, 3), "CORRECT")

    def test_partial(self):
        self.assertEqual(classify_accuracy(1, 3), "PARTIALLY_CORRECT")

    def test_none(self):
        self.assertEqual(classify_accuracy(0, 3), "INCORRECT")

    def test_no_facts_expected(self):
        self.assertEqual(classify_accuracy(0, 0), "CORRECT")


class TestClassifyFailureMode(unittest.TestCase):
    def test_e2e_success(self):
        self.assertEqual(classify_failure_mode(True, "CORRECT"), "END_TO_END_SUCCESS")

    def test_gen_failure(self):
        self.assertEqual(classify_failure_mode(True, "INCORRECT"), "GENERATION_FAILURE")

    def test_ret_failure(self):
        self.assertEqual(classify_failure_mode(False, "INCORRECT"), "RETRIEVAL_FAILURE")


class TestDeterministicFactCheck(unittest.TestCase):
    def test_matching(self):
        results = deterministic_fact_check(
            ["B.Tech Artificial Intelligence", "VIPS"],
            "Bhupender has a B.Tech in Artificial Intelligence from VIPS."
        )
        self.assertTrue(results[0]["supported"])
        self.assertTrue(results[1]["supported"])

    def test_no_match(self):
        results = deterministic_fact_check(
            ["Harvard University", "Computer Science"],
            "He studied literature at a local school."
        )
        self.assertFalse(results[0]["supported"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
tests/test_citation_evaluation.py

Unit tests for Phase 5 — Simple Citation Evaluation.
All tests are fully mocked.
"""

import unittest
from unittest.mock import MagicMock
import json

from Evaluation_Engine.citation_evaluator import (
    CitationEvaluator,
    parse_citations,
    deduplicate_sources,
    deterministic_support_check,
    is_refusal
)

class TestCitationEvaluationEngine(unittest.TestCase):

    def setUp(self):
        self.mock_embedder = MagicMock()
        self.mock_embedder.embed_query.return_value = [0.1] * 384
        
        self.mock_retriever = MagicMock()
        self.mock_retriever.retrieve_and_gate.return_value = {
            "results": [
                {"chunk_id": "c1", "text": "Bhupender works at JLL.", "metadata": {"file_name": "jll.txt", "document_id": "doc1"}},
                {"chunk_id": "c2", "text": "He completed B.Tech at VIPS.", "metadata": {"file_name": "edu.txt", "document_id": "doc2"}}
            ]
        }
        
        self.mock_context_builder = MagicMock()
        self.mock_context_builder.build.return_value = {
            "query": "...",
            "context": "...",
            "sources": [],
            "has_context": True
        }
        
        self.mock_generator = MagicMock()
        
        self.evaluator = CitationEvaluator(
            embedder=self.mock_embedder,
            retriever=self.mock_retriever,
            context_builder=self.mock_context_builder,
            generation_engine=self.mock_generator
        )

    # 1. Correct citation
    def test_correct_citation(self):
        self.mock_generator.generate.side_effect = [
            {"answer": "Bhupender works at JLL [Source 1]."},  # Pipeline answer
            {"answer": "SUPPORTED"}                          # Judge answer
        ]
        
        test_case = {
            "test_id": "cit_001",
            "question": "Where does Bhupender work?",
            "answerable": True,
            "expected_sources": ["jll.txt"]
        }
        
        res = self.evaluator.evaluate([test_case])
        q_res = res["per_query"][0]
        
        self.assertEqual(q_res["classification"], "CORRECT")
        self.assertTrue(q_res["citation_present"])
        self.assertEqual(q_res["cited_sources"], ["jll.txt"])
        self.assertEqual(res["summary"]["correct_citations"], 1)

    # 2. Missing citation
    def test_missing_citation(self):
        self.mock_generator.generate.side_effect = [
            {"answer": "Bhupender works at JLL."}  # Factual answer but no source tag
        ]
        
        test_case = {
            "test_id": "cit_002",
            "question": "Where does Bhupender work?",
            "answerable": True,
            "expected_sources": ["jll.txt"]
        }
        
        res = self.evaluator.evaluate([test_case])
        q_res = res["per_query"][0]
        
        self.assertEqual(q_res["classification"], "MISSING_CITATION")
        self.assertFalse(q_res["citation_present"])
        self.assertEqual(res["summary"]["missing_citations"], 1)

    # 3. Invalid citation number
    def test_invalid_citation_number(self):
        self.mock_generator.generate.side_effect = [
            {"answer": "Bhupender works at JLL [Source 5]."}  # Invalid index (only 2 sources exist)
        ]
        
        test_case = {
            "test_id": "cit_003",
            "question": "Where does Bhupender work?",
            "answerable": True,
            "expected_sources": ["jll.txt"]
        }
        
        res = self.evaluator.evaluate([test_case])
        q_res = res["per_query"][0]
        
        self.assertEqual(q_res["classification"], "WRONG_CITATION")
        self.assertIn("Invalid citation index", q_res["reason"])
        self.assertEqual(res["summary"]["wrong_citations"], 1)

    # 4. Citation points to unsupported source
    def test_unsupported_source_citation(self):
        self.mock_generator.generate.side_effect = [
            {"answer": "Bhupender works at JLL [Source 1]."},
            {"answer": "NOT_SUPPORTED"}  # Judge verdict
        ]
        
        test_case = {
            "test_id": "cit_004",
            "question": "Where does Bhupender work?",
            "answerable": True,
            "expected_sources": ["jll.txt"]
        }
        
        res = self.evaluator.evaluate([test_case])
        q_res = res["per_query"][0]
        
        self.assertEqual(q_res["classification"], "UNSUPPORTED_CITATION")
        self.assertIn("does not support the answer", q_res["reason"])
        self.assertEqual(res["summary"]["unsupported_citations"], 1)

    # 5. Correct abstention without citation
    def test_correct_abstention_no_citation(self):
        self.mock_generator.generate.side_effect = [
            {"answer": "I couldn't find relevant information in the uploaded documents."}
        ]
        
        test_case = {
            "test_id": "cit_005",
            "question": "What is his salary?",
            "answerable": False,
            "expected_sources": []
        }
        
        res = self.evaluator.evaluate([test_case])
        q_res = res["per_query"][0]
        
        self.assertEqual(q_res["classification"], "CORRECT_ABSTENTION")
        self.assertFalse(q_res["citation_present"])
        self.assertEqual(res["summary"]["correct_abstentions"], 1)

    # 6. Multiple valid citations
    def test_multiple_citations(self):
        self.mock_generator.generate.side_effect = [
            {"answer": "Bhupender works at JLL [Source 1] and completed B.Tech at VIPS [Source 2]."},
            {"answer": "SUPPORTED"},  # Judge call for [Source 1]
            {"answer": "SUPPORTED"}   # Judge call for [Source 2]
        ]
        
        test_case = {
            "test_id": "cit_006",
            "question": "Provide details on Bhupender.",
            "answerable": True,
            "expected_sources": ["jll.txt", "edu.txt"]
        }
        
        res = self.evaluator.evaluate([test_case])
        q_res = res["per_query"][0]
        
        self.assertEqual(q_res["classification"], "CORRECT")
        self.assertEqual(sorted(q_res["cited_sources"]), sorted(["jll.txt", "edu.txt"]))
        self.assertEqual(res["summary"]["correct_citations"], 1)

    # Helper function tests
    def test_parse_citations(self):
        self.assertEqual(parse_citations("Hello [Source 1] and [Source 2] and [Source 1]"), [1, 2])
        self.assertEqual(parse_citations("No source here"), [])

    def test_is_refusal(self):
        self.assertTrue(is_refusal("I couldn't find relevant information."))
        self.assertFalse(is_refusal("He works at JLL."))

    def test_deterministic_support_check(self):
        # Good overlap
        verdict = deterministic_support_check("Bhupender works at Jones Lang LaSalle JLL", "Bhupender worked at Jones Lang LaSalle (JLL)")
        self.assertEqual(verdict, "SUPPORTED")
        
        # Bad overlap
        verdict2 = deterministic_support_check("He has ten years of experience at Microsoft", "Bhupender works at JLL")
        self.assertEqual(verdict2, "NOT_SUPPORTED")

if __name__ == "__main__":
    unittest.main()

"""
tests/test_grounding_evaluation.py

Phase 4 — Groundedness & Hallucination Evaluation test suite.

All tests are fully mocked (no Ollama, no ChromaDB, no embedding model).
10 test cases covering the full specification.
"""

import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../engines")))

from Evaluation_Engine.grounding_evaluator import (
    GroundingEvaluator,
    is_refusal,
    deterministic_extract_claims,
    deterministic_check_grounding,
    classify_grounding,
    classify_unanswerable,
)


# ---------------------------------------------------------------------------
# Shared context fixture
# ---------------------------------------------------------------------------

SAMPLE_CONTEXT = (
    "Bhupender completed a B.Tech in Artificial Intelligence and Data Science "
    "from Vivekananda Institute of Professional Studies. "
    "He currently works at Jones Lang LaSalle (JLL) as an Analyst from July 2025. "
    "He was a Founding ML Engineer at Tecraki Tech Solutions from December 2023 to June 2025."
)

SAMPLE_CHUNKS = [
    {
        "chunk_id": "doc_chunk_0",
        "text": SAMPLE_CONTEXT,
        "metadata": {"file_name": "test.txt", "source_path": "test.txt"},
        "score": 0.75,
    }
]


def _make_evaluator(
    generate_side_effects=None,
    context_text=SAMPLE_CONTEXT,
    accepted_chunks=None,
):
    """Build a GroundingEvaluator with all dependencies fully mocked."""
    if accepted_chunks is None:
        accepted_chunks = SAMPLE_CHUNKS

    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 384

    retriever = MagicMock()
    retriever.retrieve_and_gate.return_value = {"results": accepted_chunks}

    context_builder = MagicMock()
    context_builder.build.return_value = {
        "query": "...",
        "context": context_text,
        "sources": accepted_chunks,
        "has_context": bool(context_text),
    }

    gen_engine = MagicMock()
    if generate_side_effects:
        gen_engine.generate.side_effect = generate_side_effects
    else:
        gen_engine.generate.return_value = {"answer": "", "model": "qwen2.5:3b"}

    return GroundingEvaluator(
        embedder=embedder,
        retriever=retriever,
        context_builder=context_builder,
        generation_engine=gen_engine,
    )


def _make_test_case(
    test_id="g001",
    question="Test?",
    answerable=True,
    expected_sources=None,
    expected_facts=None,
):
    return {
        "test_id": test_id,
        "question": question,
        "answerable": answerable,
        "expected_sources": expected_sources or ["test.txt"],
        "expected_facts": expected_facts or [],
    }


# ---------------------------------------------------------------------------
# Test 1: Fully grounded answer
# ---------------------------------------------------------------------------

class TestFullyGroundedAnswer(unittest.TestCase):
    """All extracted claims are SUPPORTED by the context."""

    def test_classification_is_grounded(self):
        """Fully grounded answer: all claims SUPPORTED -> GROUNDED."""
        # SLM responses: generate answer, extract claims, judge each claim
        generate_side_effects = [
            # 1st call: generate answer
            {"answer": "Bhupender completed a B.Tech at Vivekananda Institute. He works at JLL."},
            # 2nd call: claim extraction
            {"answer": json.dumps({"claims": [
                "Bhupender completed a B.Tech at Vivekananda Institute.",
                "He works at JLL.",
            ]})},
            # 3rd call: judge claim 1
            {"answer": json.dumps({
                "claim": "Bhupender completed a B.Tech at Vivekananda Institute.",
                "status": "SUPPORTED",
                "evidence": "Bhupender completed a B.Tech in AI...",
                "reason": "Context explicitly states this.",
            })},
            # 4th call: judge claim 2
            {"answer": json.dumps({
                "claim": "He works at JLL.",
                "status": "SUPPORTED",
                "evidence": "He currently works at JLL as an Analyst.",
                "reason": "Context confirms current role.",
            })},
        ]
        ev = _make_evaluator(generate_side_effects=generate_side_effects)
        result = ev.evaluate([_make_test_case()], top_k=3, score_threshold=0.40)
        pq = result["per_query"][0]

        self.assertEqual(pq["classification"], "GROUNDED")
        self.assertEqual(pq["unsupported_claims"], 0)
        self.assertEqual(pq["supported_claims"], 2)
        self.assertEqual(result["summary"]["grounded"], 1)
        self.assertAlmostEqual(result["summary"]["groundedness_rate"], 1.0)
        print("1. Fully grounded answer -> GROUNDED. [PASS]")


# ---------------------------------------------------------------------------
# Test 2: Partially grounded answer
# ---------------------------------------------------------------------------

class TestPartiallyGroundedAnswer(unittest.TestCase):
    """One claim is SUPPORTED, one is UNSUPPORTED -> PARTIALLY_GROUNDED."""

    def test_classification_is_partially_grounded(self):
        generate_side_effects = [
            {"answer": "Bhupender works at JLL. He has 10 years of experience."},
            {"answer": json.dumps({"claims": [
                "Bhupender works at JLL.",
                "He has 10 years of experience.",
            ]})},
            {"answer": json.dumps({
                "claim": "Bhupender works at JLL.",
                "status": "SUPPORTED",
                "evidence": "He currently works at JLL as an Analyst.",
                "reason": "Context confirms this.",
            })},
            {"answer": json.dumps({
                "claim": "He has 10 years of experience.",
                "status": "UNSUPPORTED",
                "evidence": "",
                "reason": "Context does not mention years of experience.",
            })},
        ]
        ev = _make_evaluator(generate_side_effects=generate_side_effects)
        result = ev.evaluate([_make_test_case()], top_k=3, score_threshold=0.40)
        pq = result["per_query"][0]

        self.assertEqual(pq["classification"], "PARTIALLY_GROUNDED")
        self.assertEqual(pq["supported_claims"], 1)
        self.assertEqual(pq["unsupported_claims"], 1)
        self.assertEqual(result["summary"]["partially_grounded"], 1)
        print("2. Partially grounded answer -> PARTIALLY_GROUNDED. [PASS]")


# ---------------------------------------------------------------------------
# Test 3: Completely hallucinated answer
# ---------------------------------------------------------------------------

class TestCompletelyHallucinatedAnswer(unittest.TestCase):
    """All claims are UNSUPPORTED -> UNGROUNDED."""

    def test_classification_is_ungrounded(self):
        generate_side_effects = [
            {"answer": "Bhupender studied at Oxford. He has 10 years at Google."},
            {"answer": json.dumps({"claims": [
                "Bhupender studied at Oxford.",
                "He has 10 years at Google.",
            ]})},
            {"answer": json.dumps({
                "claim": "Bhupender studied at Oxford.",
                "status": "UNSUPPORTED",
                "evidence": "",
                "reason": "No mention of Oxford in context.",
            })},
            {"answer": json.dumps({
                "claim": "He has 10 years at Google.",
                "status": "UNSUPPORTED",
                "evidence": "",
                "reason": "No mention of Google in context.",
            })},
        ]
        ev = _make_evaluator(generate_side_effects=generate_side_effects)
        result = ev.evaluate([_make_test_case()], top_k=3, score_threshold=0.40)
        pq = result["per_query"][0]

        self.assertEqual(pq["classification"], "UNGROUNDED")
        self.assertEqual(pq["supported_claims"], 0)
        self.assertEqual(pq["unsupported_claims"], 2)
        self.assertAlmostEqual(result["summary"]["hallucination_rate"], 1.0)
        print("3. Completely hallucinated answer -> UNGROUNDED. [PASS]")


# ---------------------------------------------------------------------------
# Test 4: Correct abstention (unanswerable)
# ---------------------------------------------------------------------------

class TestCorrectAbstention(unittest.TestCase):
    """Unanswerable query answered with refusal -> CORRECT_ABSTENTION."""

    def test_correct_abstention(self):
        generate_side_effects = [
            # Generate refusal answer
            {"answer": "I couldn't find relevant information about that in the uploaded documents."},
        ]
        ev = _make_evaluator(generate_side_effects=generate_side_effects)
        case = _make_test_case(test_id="g_unans_1", answerable=False, expected_sources=[])
        result = ev.evaluate([case], top_k=3, score_threshold=0.40)
        pq = result["per_query"][0]

        self.assertEqual(pq["classification"], "CORRECT_ABSTENTION")
        self.assertAlmostEqual(result["summary"]["correct_abstention_rate"], 1.0)
        self.assertAlmostEqual(result["summary"]["false_answer_rate"], 0.0)
        print("4. Unanswerable + refusal -> CORRECT_ABSTENTION. [PASS]")


# ---------------------------------------------------------------------------
# Test 5: False answer to unanswerable question
# ---------------------------------------------------------------------------

class TestFalseAnswerUnanswerable(unittest.TestCase):
    """Unanswerable query answered with fabricated facts -> FALSE_ANSWER."""

    def test_false_answer_on_unanswerable(self):
        generate_side_effects = [
            # Generate fabricated answer for unanswerable
            {"answer": "Bhupender's girlfriend is Sarah Connor."},
            # Extract claims
            {"answer": json.dumps({"claims": ["Bhupender's girlfriend is Sarah Connor."]})},
            # Judge claim
            {"answer": json.dumps({
                "claim": "Bhupender's girlfriend is Sarah Connor.",
                "status": "UNSUPPORTED",
                "evidence": "",
                "reason": "No mention of girlfriend in context.",
            })},
        ]
        ev = _make_evaluator(generate_side_effects=generate_side_effects)
        case = _make_test_case(test_id="g_false_1", answerable=False, expected_sources=[])
        result = ev.evaluate([case], top_k=3, score_threshold=0.40)
        pq = result["per_query"][0]

        self.assertIn(pq["classification"], {"FALSE_ANSWER", "HALLUCINATION"})
        self.assertGreater(result["summary"]["false_answer_rate"], 0.0)
        print("5. Unanswerable + fabricated answer -> FALSE_ANSWER/HALLUCINATION. [PASS]")


# ---------------------------------------------------------------------------
# Test 6: Answer with no factual claims
# ---------------------------------------------------------------------------

class TestNoFactualClaims(unittest.TestCase):
    """Answer contains only greetings/filler -> NO_FACTUAL_CLAIMS."""

    def test_no_factual_claims(self):
        generate_side_effects = [
            {"answer": "Sure! I'm happy to help you today."},
            # Claim extraction returns empty list
            {"answer": json.dumps({"claims": []})},
        ]
        ev = _make_evaluator(generate_side_effects=generate_side_effects)
        result = ev.evaluate([_make_test_case()], top_k=3, score_threshold=0.40)
        pq = result["per_query"][0]

        self.assertEqual(pq["classification"], "NO_FACTUAL_CLAIMS")
        self.assertEqual(pq["total_claims"], 0)
        self.assertEqual(result["summary"]["no_factual_claims"], 1)
        print("6. No factual claims in answer -> NO_FACTUAL_CLAIMS. [PASS]")


# ---------------------------------------------------------------------------
# Test 7: Multiple claims, only one unsupported
# ---------------------------------------------------------------------------

class TestMultipleClaimsOneUnsupported(unittest.TestCase):
    """2 supported + 1 unsupported -> PARTIALLY_GROUNDED."""

    def test_partially_grounded_three_claims(self):
        generate_side_effects = [
            {"answer": "Bhupender works at JLL. He studied at VIPS. He speaks 5 languages."},
            {"answer": json.dumps({"claims": [
                "Bhupender works at JLL.",
                "He studied at VIPS.",
                "He speaks 5 languages.",
            ]})},
            {"answer": json.dumps({"claim": "Bhupender works at JLL.", "status": "SUPPORTED", "evidence": "...", "reason": "..."})},
            {"answer": json.dumps({"claim": "He studied at VIPS.", "status": "SUPPORTED", "evidence": "...", "reason": "..."})},
            {"answer": json.dumps({"claim": "He speaks 5 languages.", "status": "UNSUPPORTED", "evidence": "", "reason": "Not in context."})},
        ]
        ev = _make_evaluator(generate_side_effects=generate_side_effects)
        result = ev.evaluate([_make_test_case()], top_k=3, score_threshold=0.40)
        pq = result["per_query"][0]

        self.assertEqual(pq["classification"], "PARTIALLY_GROUNDED")
        self.assertEqual(pq["supported_claims"], 2)
        self.assertEqual(pq["unsupported_claims"], 1)
        print("7. 2 supported + 1 unsupported -> PARTIALLY_GROUNDED. [PASS]")


# ---------------------------------------------------------------------------
# Test 8: Ollama timeout -> deterministic fallback
# ---------------------------------------------------------------------------

class TestOllamaTimeoutFallback(unittest.TestCase):
    """If SLM raises an exception on claim extraction, fallback to deterministic."""

    def test_deterministic_fallback_on_timeout(self):
        generate_side_effects = [
            # 1st call: generate answer (succeeds)
            {"answer": "Bhupender studied at Vivekananda Institute. He works at JLL."},
            # 2nd call: claim extraction (simulated timeout)
            RuntimeError("Connection timed out: Ollama not responding"),
            # After deterministic extraction, judge calls will also be attempted (SLM).
            # Simulate both judge attempts failing too -> falls to deterministic judge.
            RuntimeError("Connection timed out"),   # judge attempt 1 for claim 1
            RuntimeError("Connection timed out"),   # judge attempt 2 for claim 1 (retry)
            RuntimeError("Connection timed out"),   # judge attempt 1 for claim 2
            RuntimeError("Connection timed out"),   # judge attempt 2 for claim 2
        ]
        ev = _make_evaluator(generate_side_effects=generate_side_effects)
        result = ev.evaluate([_make_test_case()], top_k=3, score_threshold=0.40)
        pq = result["per_query"][0]

        # Should have used deterministic extractor
        self.assertEqual(pq["extraction_method"], "DETERMINISTIC_EXTRACTOR")
        # Should still produce a valid classification
        self.assertIn(pq["classification"], {
            "GROUNDED", "PARTIALLY_GROUNDED", "UNGROUNDED", "NO_FACTUAL_CLAIMS"
        })
        # All claim evaluations should be deterministic fallback
        if pq["claims"]:
            for c in pq["claims"]:
                self.assertEqual(c["evaluation_method"], "DETERMINISTIC_FALLBACK")
        print("8. Ollama timeout -> deterministic fallback activated. [PASS]")


# ---------------------------------------------------------------------------
# Test 9: Malformed SLM JSON -> retry -> fallback
# ---------------------------------------------------------------------------

class TestMalformedSLMJsonFallback(unittest.TestCase):
    """If both SLM attempts for grounding judge return invalid JSON, use deterministic fallback."""

    def test_malformed_judge_json_fallback(self):
        generate_side_effects = [
            # 1st: generate answer
            {"answer": "Bhupender studied at Vivekananda Institute of Professional Studies."},
            # 2nd: claim extraction (succeeds)
            {"answer": json.dumps({"claims": ["Bhupender studied at Vivekananda Institute of Professional Studies."]})},
            # 3rd: judge attempt 1 — malformed
            {"answer": "Sorry, I can't process that. Here is some text: status: good, reason: yes."},
            # 4th: judge attempt 2 — still malformed (retry strict prompt)
            {"answer": "```\n{status: SUPPORTED, reason: yep}\n```"},
        ]
        ev = _make_evaluator(generate_side_effects=generate_side_effects)
        result = ev.evaluate([_make_test_case()], top_k=3, score_threshold=0.40)
        pq = result["per_query"][0]

        # Should have fallen back to deterministic judge
        self.assertTrue(len(pq["claims"]) > 0)
        fallback_claims = [c for c in pq["claims"] if c["evaluation_method"] == "DETERMINISTIC_FALLBACK"]
        self.assertEqual(len(fallback_claims), 1)
        print("9. Malformed SLM JSON x2 -> deterministic fallback for judge. [PASS]")


# ---------------------------------------------------------------------------
# Test 10: Empty retrieved context
# ---------------------------------------------------------------------------

class TestEmptyRetrievedContext(unittest.TestCase):
    """If no context is retrieved, all claims must be UNSUPPORTED -> UNGROUNDED."""

    def test_empty_context_yields_ungrounded(self):
        generate_side_effects = [
            # 1st: generate answer despite no context
            {"answer": "Bhupender works at JLL as an Analyst."},
            # 2nd: claim extraction
            {"answer": json.dumps({"claims": ["Bhupender works at JLL as an Analyst."]})},
        ]

        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1] * 384

        retriever = MagicMock()
        retriever.retrieve_and_gate.return_value = {"results": []}  # empty results

        context_builder = MagicMock()
        context_builder.build.return_value = {
            "query": "...",
            "context": "",  # empty context
            "sources": [],
            "has_context": False,
        }

        gen_engine = MagicMock()
        gen_engine.generate.side_effect = generate_side_effects

        ev = GroundingEvaluator(
            embedder=embedder,
            retriever=retriever,
            context_builder=context_builder,
            generation_engine=gen_engine,
        )

        case = _make_test_case(test_id="g_empty_ctx")
        result = ev.evaluate([case], top_k=3, score_threshold=0.40)
        pq = result["per_query"][0]

        # All claims unsupported (no context)
        self.assertEqual(pq["unsupported_claims"], pq["total_claims"])
        self.assertIn(pq["classification"], {"UNGROUNDED", "NO_FACTUAL_CLAIMS"})
        for claim in pq.get("claims", []):
            self.assertEqual(claim["status"], "UNSUPPORTED")
            self.assertEqual(claim["evaluation_method"], "NO_CONTEXT_FALLBACK")
        print("10. Empty retrieved context -> all claims UNSUPPORTED. [PASS]")


# ---------------------------------------------------------------------------
# Standalone helper function tests
# ---------------------------------------------------------------------------

class TestIsRefusal(unittest.TestCase):
    def test_known_refusal_phrases(self):
        self.assertTrue(is_refusal("I couldn't find relevant information about that."))
        self.assertTrue(is_refusal("The uploaded documents do not contain this information."))
        self.assertFalse(is_refusal("Bhupender works at JLL as an Analyst."))

    def test_empty_string(self):
        self.assertTrue(is_refusal(""))
        self.assertTrue(is_refusal("   "))


class TestDeterministicExtractClaims(unittest.TestCase):
    def test_basic_extraction(self):
        answer = "Bhupender studied at VIPS. He works at JLL. Based on the context, he is an AI Engineer."
        claims = deterministic_extract_claims(answer)
        # "Based on the context..." should be filtered
        self.assertFalse(any("Based on" in c for c in claims))
        self.assertTrue(any("VIPS" in c for c in claims))

    def test_empty_answer(self):
        self.assertEqual(deterministic_extract_claims(""), [])

    def test_short_sentences_filtered(self):
        answer = "Yes. Bhupender Singh Negi is an AI engineer at JLL in Delhi."
        claims = deterministic_extract_claims(answer)
        # "Yes." should be filtered (too short)
        self.assertFalse(any(c.strip() == "Yes." for c in claims))


class TestDeterministicCheckGrounding(unittest.TestCase):
    def test_high_overlap_supported(self):
        claim = "Bhupender completed B.Tech from Vivekananda Institute."
        result = deterministic_check_grounding(claim, SAMPLE_CONTEXT)
        self.assertIn(result["status"], {"SUPPORTED", "PARTIALLY_SUPPORTED"})

    def test_no_overlap_unsupported(self):
        claim = "Bhupender studied at Oxford University in London."
        result = deterministic_check_grounding(claim, SAMPLE_CONTEXT)
        self.assertEqual(result["status"], "UNSUPPORTED")

    def test_empty_context(self):
        result = deterministic_check_grounding("Any claim.", "")
        self.assertEqual(result["status"], "UNSUPPORTED")
        self.assertIn("No retrieved context", result["reason"])


class TestClassifyGrounding(unittest.TestCase):
    def test_all_supported_grounded(self):
        results = [
            {"status": "SUPPORTED"},
            {"status": "SUPPORTED"},
        ]
        cls, counts = classify_grounding(results)
        self.assertEqual(cls, "GROUNDED")
        self.assertEqual(counts["unsupported_claims"], 0)

    def test_mixed_partially_grounded(self):
        results = [
            {"status": "SUPPORTED"},
            {"status": "UNSUPPORTED"},
        ]
        cls, _ = classify_grounding(results)
        self.assertEqual(cls, "PARTIALLY_GROUNDED")

    def test_all_unsupported_ungrounded(self):
        results = [
            {"status": "UNSUPPORTED"},
            {"status": "UNSUPPORTED"},
        ]
        cls, _ = classify_grounding(results)
        self.assertEqual(cls, "UNGROUNDED")

    def test_empty_no_factual_claims(self):
        cls, counts = classify_grounding([])
        self.assertEqual(cls, "NO_FACTUAL_CLAIMS")
        self.assertEqual(counts["total_claims"], 0)


class TestClassifyUnanswerable(unittest.TestCase):
    def test_refusal_is_correct_abstention(self):
        answer = "I couldn't find relevant information in the uploaded documents."
        self.assertEqual(classify_unanswerable(answer, []), "CORRECT_ABSTENTION")

    def test_fabricated_claims_is_hallucination(self):
        answer = "Bhupender's girlfriend is Sarah Connor."
        claim_results = [{"status": "UNSUPPORTED"}]
        self.assertIn(
            classify_unanswerable(answer, claim_results),
            {"HALLUCINATION", "FALSE_ANSWER"}
        )

    def test_non_refusal_no_claims_is_false_answer(self):
        answer = "Bhupender is married."
        self.assertEqual(classify_unanswerable(answer, []), "FALSE_ANSWER")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
engines/Evaluation_Engine/grounding_evaluator.py

Phase 4 — Groundedness & Hallucination Evaluation.

Evaluates whether every factual claim in a generated answer is strictly
supported by the retrieved context (not the original document).

Pipeline per query:
  question → embed → retrieve+gate → context_build → generate
  → claim_extract (SLM or deterministic)
  → per-claim grounding judge (SLM or deterministic)
  → classification + metrics
"""

import json
import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFUSAL_PHRASES = [
    "couldn't find relevant information",
    "could not find",
    "no relevant information",
    "does not contain",
    "not found in the uploaded documents",
    "i don't have",
    "i do not have",
    "no information about",
    "unable to find",
    "not available in the provided",
    "the uploaded documents do not",
    "i couldn't find",
    "not mentioned in",
    "i cannot find",
    "i am unable to",
]

# Sentences/phrases to discard during deterministic claim extraction
IGNORE_PATTERNS = [
    r"^based on (the |this )?(context|provided|document)",
    r"^according to (the |this )?(context|provided|document)",
    r"^the (context|document|provided information) (states?|says?|indicates?|mentions?)",
    r"^\[source \d+\]",
    r"^in (summary|conclusion|short)",
    r"^(note|please note)",
    r"^(yes|no|sure|okay|certainly|of course)[,\.]",
    r"^i (hope|think|believe|suggest|recommend)",
]

# ---------------------------------------------------------------------------
# Standalone helper functions (importable for unit tests)
# ---------------------------------------------------------------------------

def is_refusal(answer: str) -> bool:
    """Return True if the answer matches a known abstention/refusal pattern."""
    if not answer or not answer.strip():
        return True
    lower = answer.lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in lower:
            return True
    # Short "I don't know" variants
    if len(answer.strip()) < 100 and any(
        t in lower for t in ["don't know", "unsure", "no idea", "cannot answer"]
    ):
        return True
    return False


def deterministic_extract_claims(answer: str) -> list[str]:
    """
    Fallback claim extractor: splits answer into sentences and filters out
    non-factual lines using heuristics.

    Conservative: keeps anything that looks like a factual statement.
    """
    if not answer or not answer.strip():
        return []

    # Split on period, exclamation, newline
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())

    claims = []
    for sent in raw_sentences:
        sent = sent.strip().strip("•-–—*")
        sent = re.sub(r"\[Source \d+\]", "", sent, flags=re.IGNORECASE).strip()

        # Too short to be meaningful
        if len(sent) < 15:
            continue

        # Match ignore patterns
        skip = False
        for pat in IGNORE_PATTERNS:
            if re.match(pat, sent, re.IGNORECASE):
                skip = True
                break
        if skip:
            continue

        # Skip pure questions
        if sent.endswith("?"):
            continue

        claims.append(sent)

    return claims


def deterministic_check_grounding(claim: str, context: str) -> dict:
    """
    Deterministic token-overlap grounding check.

    Conservative: marks UNSUPPORTED when overlap is insufficient.
    Returns a grounding result dict (without 'evaluation_method').
    """
    if not context or not context.strip():
        return {
            "claim": claim,
            "status": "UNSUPPORTED",
            "evidence": "",
            "reason": "No retrieved context available to verify claim.",
        }

    # Extract meaningful words (≥4 chars) from both claim and context
    claim_words = set(re.findall(r"\b\w{4,}\b", claim.lower()))
    context_lower = context.lower()

    if not claim_words:
        return {
            "claim": claim,
            "status": "UNSUPPORTED",
            "evidence": "",
            "reason": "Claim contains no verifiable tokens.",
        }

    matched = sum(1 for w in claim_words if w in context_lower)
    overlap = matched / len(claim_words)

    # Find the sentence in context that contains the most matching words
    evidence = ""
    best_overlap = 0.0
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", context):
        sent_lower = sentence.lower()
        sent_overlap = sum(1 for w in claim_words if w in sent_lower) / len(claim_words)
        if sent_overlap > best_overlap:
            best_overlap = sent_overlap
            evidence = sentence.strip()

    if overlap >= 0.70:
        status = "SUPPORTED"
        reason = f"Token overlap {overlap:.0%} — sufficient evidence in context."
    elif overlap >= 0.40:
        status = "PARTIALLY_SUPPORTED"
        reason = f"Token overlap {overlap:.0%} — partial evidence in context."
    else:
        status = "UNSUPPORTED"
        reason = f"Token overlap {overlap:.0%} — insufficient evidence in context."
        evidence = ""

    return {
        "claim": claim,
        "status": status,
        "evidence": evidence,
        "reason": reason,
    }


def classify_grounding(results: list[dict]) -> tuple[str, dict]:
    """
    Classify overall groundedness from per-claim results.

    Returns (classification_string, counts_dict).
    """
    if not results:
        return "NO_FACTUAL_CLAIMS", {
            "total_claims": 0,
            "supported_claims": 0,
            "partial_claims": 0,
            "unsupported_claims": 0,
        }

    supported = sum(1 for r in results if r["status"] == "SUPPORTED")
    partial = sum(1 for r in results if r["status"] == "PARTIALLY_SUPPORTED")
    unsupported = sum(1 for r in results if r["status"] == "UNSUPPORTED")
    total = len(results)

    if unsupported == 0 and total > 0:
        classification = "GROUNDED"
    elif supported > 0 and unsupported > 0:
        classification = "PARTIALLY_GROUNDED"
    elif partial > 0 and unsupported == 0:
        classification = "GROUNDED"  # all at least partial
    elif supported == 0 and unsupported > 0:
        classification = "UNGROUNDED"
    else:
        classification = "PARTIALLY_GROUNDED"

    return classification, {
        "total_claims": total,
        "supported_claims": supported,
        "partial_claims": partial,
        "unsupported_claims": unsupported,
    }


def classify_unanswerable(answer: str, claim_results: list[dict]) -> str:
    """
    Classify an unanswerable query's response.

    CORRECT_ABSTENTION: model correctly refused to answer.
    FALSE_ANSWER:       model gave an answer with fabricated/unsupported content.
    HALLUCINATION:      model gave an answer with explicitly unsupported claims.
    """
    if is_refusal(answer):
        return "CORRECT_ABSTENTION"

    # If claims were extracted and all are unsupported → HALLUCINATION
    if claim_results:
        unsupported = sum(1 for r in claim_results if r["status"] == "UNSUPPORTED")
        if unsupported == len(claim_results):
            return "HALLUCINATION"
        return "FALSE_ANSWER"

    # No claims extracted but answer is not a refusal → FALSE_ANSWER
    return "FALSE_ANSWER"


# ---------------------------------------------------------------------------
# GroundingEvaluator
# ---------------------------------------------------------------------------

class GroundingEvaluator:
    """
    Phase 4 — Groundedness & Hallucination Evaluator.

    Runs the full RAG pipeline per query, extracts factual claims from
    generated answers, judges each claim against the retrieved context
    using qwen2.5:3b, and produces a grounding classification.
    """

    # ---- Claim extraction prompts ----------------------------------------

    CLAIM_EXTRACT_PROMPT = (
        "You are a factual claim extractor.\n\n"
        "Extract all independently verifiable factual claims from the answer below.\n"
        "Ignore: greetings, opinions, conversational filler, phrases like "
        "\"Based on the context...\", citation markers like [Source 1].\n\n"
        "ANSWER:\n{answer}\n\n"
        "Return ONLY valid JSON. No preamble, no markdown fences:\n"
        "{{\"claims\": [\"claim 1\", \"claim 2\"]}}"
    )

    CLAIM_EXTRACT_STRICT_PROMPT = (
        "IMPORTANT: Output ONLY a JSON object.\n"
        "Extract factual claims from: {answer}\n"
        "{{\"claims\": [\"...\", \"...\"]}}"
    )

    # ---- Grounding judge prompts ------------------------------------------

    GROUNDING_JUDGE_PROMPT = (
        "You are a strict groundedness evaluator.\n\n"
        "RETRIEVED CONTEXT (this is the ONLY evidence allowed):\n"
        "{context}\n\n"
        "CLAIM TO EVALUATE:\n{claim}\n\n"
        "Is this claim SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED "
        "by the retrieved context above?\n"
        "Do NOT use any external knowledge. Only use what is written in the context.\n\n"
        "Return ONLY valid JSON. No preamble, no markdown fences:\n"
        "{{\"claim\": \"...\", \"status\": \"SUPPORTED\", "
        "\"evidence\": \"...\", \"reason\": \"...\"}}"
    )

    GROUNDING_JUDGE_STRICT_PROMPT = (
        "IMPORTANT: Output ONLY a JSON object. No text before or after.\n"
        "Context: {context}\n"
        "Claim: {claim}\n"
        "{{\"claim\": \"...\", \"status\": \"SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED\", "
        "\"evidence\": \"...\", \"reason\": \"...\"}}"
    )

    VALID_STATUSES = {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"}

    def __init__(self, embedder, retriever, context_builder, generation_engine):
        """
        Args:
            embedder:          EmbeddingEngine instance
            retriever:         RetrievalConnector connected to production DB
            context_builder:   ContextBuilder instance
            generation_engine: GenerationEngine (qwen2.5:3b, temperature=0)
        """
        self.embedder = embedder
        self.retriever = retriever
        self.context_builder = context_builder
        self.generation_engine = generation_engine

    # ------------------------------------------------------------------
    # Internal SLM helpers
    # ------------------------------------------------------------------

    def _parse_json_blob(self, raw: str) -> dict:
        """Strip markdown fences and extract first JSON object from raw text."""
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        return json.loads(raw)

    def _extract_claims_slm(self, answer: str) -> list[str]:
        """
        Use SLM to extract claims. Returns list of claim strings.
        Raises ValueError if JSON parsing fails after retry.
        """
        # Attempt 1
        prompt = self.CLAIM_EXTRACT_PROMPT.format(answer=answer)
        try:
            raw = self.generation_engine.generate(prompt).get("answer", "")
            data = self._parse_json_blob(raw)
            claims = data.get("claims", [])
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                return [c.strip() for c in claims if c.strip()]
        except (json.JSONDecodeError, KeyError, ValueError, AttributeError):
            pass

        # Attempt 2 — stricter prompt
        prompt2 = self.CLAIM_EXTRACT_STRICT_PROMPT.format(answer=answer)
        try:
            raw = self.generation_engine.generate(prompt2).get("answer", "")
            data = self._parse_json_blob(raw)
            claims = data.get("claims", [])
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                return [c.strip() for c in claims if c.strip()]
        except (json.JSONDecodeError, KeyError, ValueError, AttributeError):
            pass

        raise ValueError("SLM claim extraction failed after retry.")

    def _judge_claim_slm(self, claim: str, context: str) -> dict:
        """
        Use SLM to judge a single claim against the context.
        Returns a grounding result dict.
        Raises ValueError if JSON parsing fails after retry.
        """
        # Attempt 1
        prompt = self.GROUNDING_JUDGE_PROMPT.format(
            context=context[:3000],  # truncate to avoid token overflow
            claim=claim
        )
        try:
            raw = self.generation_engine.generate(prompt).get("answer", "")
            data = self._parse_json_blob(raw)
            status = data.get("status", "").upper()
            if status not in self.VALID_STATUSES:
                raise ValueError(f"Invalid status '{status}'")
            return {
                "claim": data.get("claim", claim),
                "status": status,
                "evidence": data.get("evidence", ""),
                "reason": data.get("reason", ""),
            }
        except (json.JSONDecodeError, KeyError, ValueError, AttributeError):
            pass

        # Attempt 2 — stricter prompt
        prompt2 = self.GROUNDING_JUDGE_STRICT_PROMPT.format(
            context=context[:2000],
            claim=claim
        )
        try:
            raw = self.generation_engine.generate(prompt2).get("answer", "")
            data = self._parse_json_blob(raw)
            status = data.get("status", "").upper()
            if status not in self.VALID_STATUSES:
                raise ValueError(f"Invalid status '{status}'")
            return {
                "claim": data.get("claim", claim),
                "status": status,
                "evidence": data.get("evidence", ""),
                "reason": data.get("reason", ""),
            }
        except (json.JSONDecodeError, KeyError, ValueError, AttributeError):
            pass

        raise ValueError("SLM grounding judge failed after retry.")

    # ------------------------------------------------------------------
    # Public evaluation entrypoint
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_cases: list[dict],
        top_k: int = 3,
        score_threshold: float = 0.40,
    ) -> dict:
        """
        Run Phase 4 grounding evaluation for all test cases.

        Args:
            test_cases:      Golden dataset entries (loaded by EvaluationDatasetLoader).
            top_k:           Retrieval K.
            score_threshold: Relevance gate threshold.

        Returns:
            dict with per_query list and aggregate summary.
        """
        per_query_results = []

        for case in test_cases:
            test_id = case["test_id"]
            question = case["question"]
            answerable = case["answerable"]
            expected_sources = case.get("expected_sources", [])

            # -- Step 1: embed query --
            try:
                query_vec = self.embedder.embed_query(question)
            except Exception as e:
                raise RuntimeError(
                    f"GENERATION_ENGINE_UNAVAILABLE: Embedding failed for '{test_id}': {e}"
                )

            # -- Step 2: retrieve + gate --
            try:
                gated = self.retriever.retrieve_and_gate(
                    query_vec, top_k=top_k, score_threshold=score_threshold
                )
            except Exception as e:
                raise RuntimeError(
                    f"GENERATION_ENGINE_UNAVAILABLE: Retrieval failed for '{test_id}': {e}"
                )

            accepted_chunks = gated.get("results", [])
            retrieved_sources = list(set(
                c.get("metadata", {}).get("file_name", "Unknown")
                for c in accepted_chunks
            ))
            retrieval_success = bool(accepted_chunks) and (
                not expected_sources or
                any(src in retrieved_sources for src in expected_sources)
            )

            # -- Step 3: build context --
            try:
                context_package = self.context_builder.build(question, accepted_chunks)
            except Exception as e:
                raise RuntimeError(
                    f"GENERATION_ENGINE_UNAVAILABLE: Context build failed for '{test_id}': {e}"
                )

            context_text = context_package.get("context", "")
            has_context = context_package.get("has_context", False)

            # -- Step 4: generate answer --
            try:
                gen_result = self.generation_engine.generate(context_package)
                generated_answer = gen_result.get("answer", "").strip()
            except Exception as e:
                raise RuntimeError(
                    f"GENERATION_ENGINE_UNAVAILABLE: Generation failed for '{test_id}': {e}"
                )

            # -- Step 5: handle unanswerable queries --
            if not answerable:
                result = self._evaluate_unanswerable(
                    test_id=test_id,
                    question=question,
                    generated_answer=generated_answer,
                    context_text=context_text,
                    retrieved_sources=retrieved_sources,
                    retrieval_success=retrieval_success,
                )
                per_query_results.append(result)
                continue

            # -- Step 6: extract claims --
            extraction_method = "SLM_EXTRACTOR"
            try:
                if not generated_answer or is_refusal(generated_answer):
                    claims = []
                    extraction_method = "REFUSAL_DETECTED"
                else:
                    claims = self._extract_claims_slm(generated_answer)
            except (ValueError, RuntimeError):
                claims = deterministic_extract_claims(generated_answer)
                extraction_method = "DETERMINISTIC_EXTRACTOR"

            # -- Step 7: no factual claims --
            if not claims:
                per_query_results.append({
                    "test_id": test_id,
                    "question": question,
                    "answer": generated_answer,
                    "context_text": context_text,
                    "answerable": True,
                    "classification": "NO_FACTUAL_CLAIMS",
                    "total_claims": 0,
                    "supported_claims": 0,
                    "partial_claims": 0,
                    "unsupported_claims": 0,
                    "claims": [],
                    "retrieval_success": retrieval_success,
                    "retrieved_sources": retrieved_sources,
                    "extraction_method": extraction_method,
                })
                continue

            # -- Step 8: judge each claim --
            claim_results = []
            for claim in claims:
                judge_method = "SLM_JUDGE"
                try:
                    if not has_context:
                        # No context → all claims are unsupported
                        result_entry = {
                            "claim": claim,
                            "status": "UNSUPPORTED",
                            "evidence": "",
                            "reason": "No retrieved context — claim cannot be verified.",
                        }
                        judge_method = "NO_CONTEXT_FALLBACK"
                    else:
                        result_entry = self._judge_claim_slm(claim, context_text)
                except (ValueError, RuntimeError):
                    result_entry = deterministic_check_grounding(claim, context_text)
                    judge_method = "DETERMINISTIC_FALLBACK"

                result_entry["evaluation_method"] = judge_method
                claim_results.append(result_entry)

            # -- Step 9: classify overall groundedness --
            classification, counts = classify_grounding(claim_results)

            per_query_results.append({
                "test_id": test_id,
                "question": question,
                "answer": generated_answer,
                "context_text": context_text,
                "answerable": True,
                "classification": classification,
                **counts,
                "claims": claim_results,
                "retrieval_success": retrieval_success,
                "retrieved_sources": retrieved_sources,
                "extraction_method": extraction_method,
            })

        return {
            "config": {"top_k": top_k, "score_threshold": score_threshold},
            "summary": self._aggregate(per_query_results),
            "per_query": per_query_results,
        }

    # ------------------------------------------------------------------
    # Unanswerable evaluation
    # ------------------------------------------------------------------

    def _evaluate_unanswerable(
        self,
        test_id: str,
        question: str,
        generated_answer: str,
        context_text: str,
        retrieved_sources: list[str],
        retrieval_success: bool,
    ) -> dict:
        """Evaluate grounding for an unanswerable query."""
        if is_refusal(generated_answer):
            return {
                "test_id": test_id,
                "question": question,
                "answer": generated_answer,
                "context_text": context_text,
                "answerable": False,
                "classification": "CORRECT_ABSTENTION",
                "total_claims": 0,
                "supported_claims": 0,
                "partial_claims": 0,
                "unsupported_claims": 0,
                "claims": [],
                "retrieval_success": retrieval_success,
                "retrieved_sources": retrieved_sources,
                "extraction_method": "REFUSAL_DETECTED",
            }

        # Model answered despite being unanswerable — evaluate the claims
        extraction_method = "SLM_EXTRACTOR"
        try:
            claims = self._extract_claims_slm(generated_answer)
        except (ValueError, RuntimeError):
            claims = deterministic_extract_claims(generated_answer)
            extraction_method = "DETERMINISTIC_EXTRACTOR"

        claim_results = []
        for claim in claims:
            judge_method = "SLM_JUDGE"
            try:
                result_entry = self._judge_claim_slm(claim, context_text)
            except (ValueError, RuntimeError):
                result_entry = deterministic_check_grounding(claim, context_text)
                judge_method = "DETERMINISTIC_FALLBACK"
            result_entry["evaluation_method"] = judge_method
            claim_results.append(result_entry)

        classification = classify_unanswerable(generated_answer, claim_results)
        _, counts = classify_grounding(claim_results)

        return {
            "test_id": test_id,
            "question": question,
            "answer": generated_answer,
            "context_text": context_text,
            "answerable": False,
            "classification": classification,
            **counts,
            "claims": claim_results,
            "retrieval_success": retrieval_success,
            "retrieved_sources": retrieved_sources,
            "extraction_method": extraction_method,
        }

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def _aggregate(self, results: list[dict]) -> dict:
        answerable = [r for r in results if r["answerable"]]
        unanswerable = [r for r in results if not r["answerable"]]

        evaluable = [
            r for r in answerable
            if r["classification"] != "NO_FACTUAL_CLAIMS"
        ]

        grounded = sum(1 for r in evaluable if r["classification"] == "GROUNDED")
        partial = sum(1 for r in evaluable if r["classification"] == "PARTIALLY_GROUNDED")
        ungrounded = sum(1 for r in evaluable if r["classification"] == "UNGROUNDED")
        no_claims = sum(1 for r in answerable if r["classification"] == "NO_FACTUAL_CLAIMS")

        correct_abstention = sum(
            1 for r in unanswerable if r["classification"] == "CORRECT_ABSTENTION"
        )
        false_answer = sum(
            1 for r in unanswerable if r["classification"] == "FALSE_ANSWER"
        )
        hallucination_unans = sum(
            1 for r in unanswerable if r["classification"] == "HALLUCINATION"
        )

        # Aggregate claim counts across all answerable queries
        total_claims = sum(r["total_claims"] for r in answerable)
        total_supported = sum(r["supported_claims"] for r in answerable)
        total_partial = sum(r["partial_claims"] for r in answerable)
        total_unsupported = sum(r["unsupported_claims"] for r in answerable)

        # Overall hallucination (answerable queries that are UNGROUNDED)
        hallucination_ans = ungrounded

        n_eval = len(evaluable) or 1
        n_unans = len(unanswerable) or 1
        n_claims = total_claims or 1

        return {
            "total_queries": len(results),
            "answerable_count": len(answerable),
            "unanswerable_count": len(unanswerable),
            "evaluable_count": len(evaluable),
            "grounded": grounded,
            "partially_grounded": partial,
            "ungrounded": ungrounded,
            "no_factual_claims": no_claims,
            "correct_abstention": correct_abstention,
            "false_answer": false_answer,
            "hallucination_unanswerable": hallucination_unans,
            "groundedness_rate": grounded / n_eval,
            "partial_grounding_rate": partial / n_eval,
            "hallucination_rate": hallucination_ans / n_eval,
            "claim_support_rate": total_supported / n_claims,
            "unsupported_claim_rate": total_unsupported / n_claims,
            "correct_abstention_rate": correct_abstention / n_unans,
            "false_answer_rate": (false_answer + hallucination_unans) / n_unans,
            "total_claims_evaluated": total_claims,
            "total_supported_claims": total_supported,
            "total_partial_claims": total_partial,
            "total_unsupported_claims": total_unsupported,
        }

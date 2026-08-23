"""
engines/Evaluation_Engine/citation_evaluator.py

Phase 5 — Simple Citation Evaluation.

Evaluates:
1. Citation Presence: Checks if the generated answer contains citation markers like [Source 1].
2. Citation Mapping: Maps [Source X] to document file names.
3. Citation Support: Checks if the retrieved context belonging to the cited document supports the answer.
"""

import re
import json
import sys
import os

# Import the production Citation Engine so we replicate the same citation injection step
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from Citation_Engine.citation_engine import CitationEngine as _CitationEngine
except ImportError:
    _CitationEngine = None

# Refusal/Abstention phrases to identify correct abstentions
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

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", 
    "where", "why", "how", "what", "who", "whom", "this", "that", "these", 
    "those", "is", "am", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", 
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up", 
    "about", "into", "over", "after", "before", "out", "also", "he", "she", 
    "it", "they", "we", "i", "his", "her", "its", "their", "our", "my"
}

def is_refusal(answer: str) -> bool:
    """Return True if the answer matches a refusal/abstention phrase."""
    if not answer or not answer.strip():
        return True
    lower = answer.lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in lower:
            return True
    if len(answer.strip()) < 100 and any(
        t in lower for t in ["don't know", "unsure", "no idea", "cannot answer"]
    ):
        return True
    return False

def parse_citations(answer: str) -> list[int]:
    """Parse all [Source X] references from answer string."""
    if not answer:
        return []
    matches = re.findall(r"\[Source (\d+)\]", answer)
    return sorted(list(set(int(m) for m in matches)))

def deduplicate_sources(retrieved_chunks: list[dict]) -> tuple[list[dict], dict]:
    """
    Deduplicate chunks to unique source documents, matching production logic.
    Returns (unique_docs, doc_id_to_citation_id).
    """
    if not retrieved_chunks:
        return [], {}

    unique_docs = []
    seen_doc_ids = set()
    doc_id_to_citation_id = {}

    for src in retrieved_chunks:
        if not isinstance(src, dict):
            continue
        meta = src.get("metadata", {}) or {}
        doc_id = meta.get("document_id") if isinstance(meta, dict) else None
        if not doc_id:
            doc_id = src.get("document_id")
        if not doc_id:
            doc_id = src.get("chunk_id", "unknown_doc")

        if doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            citation_id = len(unique_docs) + 1
            doc_id_to_citation_id[doc_id] = citation_id

            file_name = meta.get("file_name") if isinstance(meta, dict) else None
            if not file_name:
                file_name = src.get("file_name", "Unknown File")
            source_path = meta.get("source_path") if isinstance(meta, dict) else None
            if not source_path:
                source_path = src.get("source_path")

            unique_docs.append({
                "citation_id": citation_id,
                "document_id": doc_id,
                "file_name": file_name if file_name else "Unknown File",
                "source_path": source_path
            })

    return unique_docs, doc_id_to_citation_id

def deterministic_support_check(answer: str, source_context: str) -> str:
    """
    Fallback Jaccard / token overlap validation of citation support.
    Returns 'SUPPORTED' or 'NOT_SUPPORTED'.
    """
    if not source_context or not source_context.strip():
        return "NOT_SUPPORTED"

    # Normalize and extract meaningful words
    clean_ans = re.sub(r'[^\w\s]', '', answer.lower())
    ans_tokens = [w for w in clean_ans.split() if w not in STOPWORDS and len(w) >= 3]

    if not ans_tokens:
        return "NOT_SUPPORTED"

    context_lower = source_context.lower()
    matched = sum(1 for t in ans_tokens if t in context_lower)
    overlap_ratio = matched / len(ans_tokens)

    # If at least 40% of key answer tokens are found in the context block, count as supported
    return "SUPPORTED" if overlap_ratio >= 0.40 else "NOT_SUPPORTED"


class CitationEvaluator:
    """
    Phase 5 — Citation Presence, Correctness, and Support Evaluator.
    """

    JUDGE_PROMPT = (
        "You are a strict citation validation judge.\n"
        "Evaluate whether the cited source context supports the factual statement in the answer.\n\n"
        "QUESTION: {question}\n"
        "ANSWER: {answer}\n"
        "CITED SOURCE FILENAME: {source_name}\n"
        "CITED SOURCE CONTEXT:\n{source_context}\n\n"
        "Does the cited source context provide direct evidence supporting the answer to the question?\n"
        "Return ONLY \"SUPPORTED\" or \"NOT_SUPPORTED\". Do not output any other explanation or text."
    )

    def __init__(self, embedder, retriever, context_builder, generation_engine):
        self.embedder = embedder
        self.retriever = retriever
        self.context_builder = context_builder
        self.generation_engine = generation_engine
        # Use the production Citation Engine to inject [Source N] markers — same as app.py does
        self.citation_engine = _CitationEngine() if _CitationEngine is not None else None

    def _judge_citation_slm(self, question: str, answer: str, source_name: str, source_context: str) -> str:
        """Call Ollama to validate citation support. Fallback to deterministic on failure."""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            source_name=source_name,
            source_context=source_context[:2500]  # truncate to prevent prompt blowup
        )
        try:
            res = self.generation_engine.generate(prompt)
            verdict = res.get("answer", "").strip().upper()
            if "NOT_SUPPORTED" in verdict:
                return "NOT_SUPPORTED"
            if "SUPPORTED" in verdict:
                return "SUPPORTED"
        except Exception:
            pass
        return deterministic_support_check(answer, source_context)

    def evaluate(self, test_cases: list[dict], top_k: int = 3, score_threshold: float = 0.40) -> dict:
        """
        Run citation evaluation sweep.
        """
        per_query_results = []

        for case in test_cases:
            test_id = case["test_id"]
            question = case["question"]
            answerable = case["answerable"]
            expected_sources = case.get("expected_sources", [])

            # Step 1: Run RAG pipeline to get cited answer and retrieved chunks
            try:
                query_vec = self.embedder.embed_query(question)
                gated = self.retriever.retrieve_and_gate(query_vec, top_k=top_k, score_threshold=score_threshold)
                accepted_chunks = gated.get("results", [])
                context_package = self.context_builder.build(question, accepted_chunks)
                gen_result = self.generation_engine.generate(context_package)
                raw_answer = gen_result.get("answer", "").strip()

                # Step 1b: Run Citation Engine post-processing — same as production app.py does
                # This injects [Source N] markers so the evaluator can detect them
                if self.citation_engine is not None and accepted_chunks and not is_refusal(raw_answer):
                    try:
                        cite_result = self.citation_engine.generate_citations(raw_answer, context_package)
                        generated_answer = cite_result.get("answer", raw_answer).strip()
                    except Exception:
                        generated_answer = raw_answer
                else:
                    generated_answer = raw_answer
            except Exception as e:
                # If pipeline fails, raise error to runner
                raise RuntimeError(f"GENERATION_ENGINE_UNAVAILABLE: Pipeline failed on '{test_id}': {e}")

            # Step 2: Parse and Map Citations
            citations = parse_citations(generated_answer)
            citation_present = len(citations) > 0
            unique_docs, _ = deduplicate_sources(accepted_chunks)

            # Check if answer is refusal/abstention
            refused = is_refusal(generated_answer)

            # Classification logic
            classification = None
            reason = ""
            cited_sources_names = []

            # 1. Unanswerable correct abstention
            if not answerable:
                if refused and not citation_present:
                    classification = "CORRECT_ABSTENTION"
                else:
                    # Fabricated answer to unanswerable query
                    if citation_present:
                        classification = "UNSUPPORTED_CITATION"
                        reason = "Factual answer generated with citations for an unanswerable query."
                    else:
                        classification = "MISSING_CITATION"
                        reason = "Factual answer generated without citations for an unanswerable query."

            # 2. Answerable queries
            else:
                if refused:
                    classification = "UNSUPPORTED_CITATION"
                    reason = "Model incorrectly abstained from answering an answerable query."
                elif not citation_present:
                    classification = "MISSING_CITATION"
                    reason = "Factual answer was generated but no citations were present."
                else:
                    # Verify each citation mapping and support
                    wrong_citation = False
                    unsupported_citation = False
                    wrong_citation_details = []
                    unsupported_citation_details = []

                    for idx in citations:
                        # Check index validity
                        if idx <= 0 or idx > len(unique_docs):
                            wrong_citation = True
                            wrong_citation_details.append(f"Invalid citation index [{idx}] (total unique sources: {len(unique_docs)})")
                            continue

                        cited_doc = unique_docs[idx - 1]
                        file_name = cited_doc["file_name"]
                        cited_sources_names.append(file_name)

                        # Check if matches expected sources
                        if expected_sources and file_name not in expected_sources:
                            wrong_citation = True
                            wrong_citation_details.append(f"Cited source '{file_name}' is not in expected sources: {expected_sources}")
                            continue

                        # Extract context belonging to this file_name
                        source_chunks = []
                        for chunk in accepted_chunks:
                            meta = chunk.get("metadata", {}) or {}
                            cfn = meta.get("file_name") if isinstance(meta, dict) else chunk.get("file_name")
                            if not cfn:
                                cfn = chunk.get("file_name")
                            if cfn == file_name:
                                source_chunks.append(chunk.get("text", ""))
                        source_context = "\n".join(source_chunks)

                        # Check if the source context supports the answer
                        verdict = self._judge_citation_slm(question, generated_answer, file_name, source_context)
                        if verdict != "SUPPORTED":
                            unsupported_citation = True
                            unsupported_citation_details.append(f"Cited source '{file_name}' does not support the answer.")

                    if wrong_citation:
                        classification = "WRONG_CITATION"
                        reason = "; ".join(wrong_citation_details)
                    elif unsupported_citation:
                        classification = "UNSUPPORTED_CITATION"
                        reason = "; ".join(unsupported_citation_details)
                    else:
                        classification = "CORRECT"

            per_query_results.append({
                "test_id": test_id,
                "question": question,
                "answer": generated_answer,
                "citation_present": citation_present,
                "cited_sources": cited_sources_names,
                "classification": classification,
                "reason": reason,
                "answerable": answerable
            })

        return {
            "config": {"top_k": top_k, "score_threshold": score_threshold},
            "summary": self._aggregate(per_query_results),
            "per_query": per_query_results
        }

    def _aggregate(self, results: list[dict]) -> dict:
        total = len(results)
        correct = sum(1 for r in results if r["classification"] == "CORRECT")
        missing = sum(1 for r in results if r["classification"] == "MISSING_CITATION")
        wrong = sum(1 for r in results if r["classification"] == "WRONG_CITATION")
        unsupported = sum(1 for r in results if r["classification"] == "UNSUPPORTED_CITATION")
        abstention = sum(1 for r in results if r["classification"] == "CORRECT_ABSTENTION")

        # evaluated citations are answerable factual queries
        answerable = [r for r in results if r["answerable"]]
        abstentions_ans = sum(1 for r in answerable if is_refusal(r["answer"]))
        factual_answers = [r for r in answerable if not is_refusal(r["answer"])]
        
        # total evaluated citations: answerable queries that got a factual response
        total_eval_citations = len(factual_answers)
        
        # correctly supported citations
        correct_citations = sum(1 for r in factual_answers if r["classification"] == "CORRECT")

        citation_accuracy = correct_citations / total_eval_citations if total_eval_citations > 0 else 1.0
        citation_coverage = sum(1 for r in answerable if r["citation_present"]) / len(answerable) if len(answerable) > 0 else 1.0
        missing_rate = missing / len(answerable) if len(answerable) > 0 else 0.0
        unsupported_rate = unsupported / len(answerable) if len(answerable) > 0 else 0.0

        return {
            "total_queries": total,
            "correct_citations": correct,
            "missing_citations": missing,
            "wrong_citations": wrong,
            "unsupported_citations": unsupported,
            "correct_abstentions": abstention,
            "citation_accuracy": citation_accuracy,
            "citation_coverage": citation_coverage,
            "missing_citation_rate": missing_rate,
            "unsupported_citation_rate": unsupported_rate
        }

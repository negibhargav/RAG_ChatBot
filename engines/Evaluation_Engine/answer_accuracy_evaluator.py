import json
import re

# Deterministic refusal phrases used by the production ContextBuilder/GenerationEngine
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
    "not mentioned in"
]


def classify_abstention(answer: str) -> str:
    """
    Classifies the abstention type for unanswerable queries.

    Returns:
        "CORRECT_ABSTENTION"  — answer clearly refuses to answer.
        "HALLUCINATED_ANSWER" — answer makes factual claims.
        "UNSAFE_ABSTENTION"   — answer is ambiguous.
    """
    if not answer or not answer.strip():
        return "UNSAFE_ABSTENTION"

    lower = answer.lower()

    # Check known refusal markers
    for phrase in REFUSAL_PHRASES:
        if phrase in lower:
            return "CORRECT_ABSTENTION"

    # If answer is very short (e.g. "I don't know.") treat as correct abstention
    if len(answer.strip()) < 80 and ("don't know" in lower or "unsure" in lower or "no idea" in lower):
        return "CORRECT_ABSTENTION"

    # Answers containing numbers, currency symbols, specific names or assertions are hallucinations
    if re.search(r"(\d[\d,\.]+|\₹|\$|is\s+\w+|was\s+\w+|he\s+is|she\s+is|they\s+are)", lower):
        return "HALLUCINATED_ANSWER"

    return "UNSAFE_ABSTENTION"


def classify_accuracy(facts_supported: int, facts_expected: int) -> str:
    """
    Classify answer accuracy based on supported fact counts.

    Returns:
        "CORRECT"            — all expected facts supported
        "PARTIALLY_CORRECT"  — at least one but not all supported
        "INCORRECT"          — no facts supported
    """
    if facts_expected == 0:
        return "CORRECT"
    if facts_supported == facts_expected:
        return "CORRECT"
    if facts_supported > 0:
        return "PARTIALLY_CORRECT"
    return "INCORRECT"


def classify_failure_mode(retrieval_success: bool, accuracy_class: str) -> str:
    """
    Classify the failure mode for error analysis.

    Returns:
        "END_TO_END_SUCCESS" — retrieval succeeded and answer is correct.
        "GENERATION_FAILURE" — retrieval succeeded but answer is wrong.
        "RETRIEVAL_FAILURE"  — retrieval failed (no relevant chunk found).
    """
    if retrieval_success and accuracy_class == "CORRECT":
        return "END_TO_END_SUCCESS"
    if retrieval_success:
        return "GENERATION_FAILURE"
    return "RETRIEVAL_FAILURE"


def deterministic_fact_check(facts: list[str], answer: str) -> list[dict]:
    """
    Fallback deterministic fact matching via case-insensitive substring search.
    Used when SLM judge JSON parsing fails after one retry.

    Args:
        facts: List of expected fact strings.
        answer: Generated answer text.

    Returns:
        List of {"fact": str, "supported": bool} dicts.
    """
    results = []
    lower_answer = answer.lower()
    for fact in facts:
        # Check if core words of the fact appear in the answer (token overlap)
        fact_words = set(re.findall(r"\b\w{4,}\b", fact.lower()))
        if fact_words:
            matched_words = sum(1 for w in fact_words if w in lower_answer)
            supported = matched_words / len(fact_words) >= 0.6
        else:
            supported = fact.lower() in lower_answer
        results.append({"fact": fact, "supported": supported})
    return results


class AnswerAccuracyEvaluator:
    """
    Phase 3 Answer Accuracy Evaluator.
    Executes the full RAG pipeline per query and evaluates generated answers
    against golden expected facts using qwen2.5:3b as a semantic judge.
    """

    JUDGE_PROMPT_TEMPLATE = (
        "You are a strict fact-checking evaluator.\n\n"
        "QUESTION:\n{question}\n\n"
        "EXPECTED FACTS:\n{facts_block}\n\n"
        "GENERATED ANSWER:\n{answer}\n\n"
        "For each expected fact, determine whether the generated answer supports it.\n"
        "Return ONLY valid JSON with no explanation, preamble, or markdown. "
        "Do not include ```json or any code fences.\n\n"
        "Required output format:\n"
        "{{\n"
        "  \"facts\": [\n"
        "    {{\"fact\": \"...\", \"supported\": true}},\n"
        "    {{\"fact\": \"...\", \"supported\": false}}\n"
        "  ]\n"
        "}}"
    )

    JUDGE_STRICT_PROMPT_TEMPLATE = (
        "IMPORTANT: Return ONLY a JSON object. No text before or after.\n\n"
        "QUESTION: {question}\n\n"
        "EXPECTED FACTS:\n{facts_block}\n\n"
        "GENERATED ANSWER: {answer}\n\n"
        "JSON output only:\n"
        "{{\"facts\": [{{\"fact\": \"...\", \"supported\": true/false}}, ...]}}"
    )

    def __init__(self, embedder, retriever, context_builder, generation_engine):
        """
        Args:
            embedder:          EmbeddingEngine instance
            retriever:         RetrievalConnector instance connected to production DB
            context_builder:   ContextBuilder instance
            generation_engine: GenerationEngine instance (qwen2.5:3b)
        """
        self.embedder = embedder
        self.retriever = retriever
        self.context_builder = context_builder
        self.generation_engine = generation_engine

    def _call_judge(self, question: str, facts: list[str], answer: str) -> tuple[list[dict], str]:
        """
        Call the local SLM as a fact-checking judge with retry + fallback logic.

        Returns:
            (fact_results, evaluation_method) tuple
        """
        facts_block = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))

        # Attempt 1: Standard judge prompt
        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            facts_block=facts_block,
            answer=answer
        )
        try:
            response = self.generation_engine.generate(judge_prompt)
            raw = response.get("answer", "")
            parsed = self._parse_judge_json(raw, facts)
            return parsed, "SLM_JUDGE"
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Attempt 2: Stricter prompt
        strict_prompt = self.JUDGE_STRICT_PROMPT_TEMPLATE.format(
            question=question,
            facts_block=facts_block,
            answer=answer
        )
        try:
            response = self.generation_engine.generate(strict_prompt)
            raw = response.get("answer", "")
            parsed = self._parse_judge_json(raw, facts)
            return parsed, "SLM_JUDGE"
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback: deterministic matching
        return deterministic_fact_check(facts, answer), "DETERMINISTIC_FALLBACK"

    def _parse_judge_json(self, raw: str, facts: list[str]) -> list[dict]:
        """
        Parse strict JSON output from the judge model.
        Extracts a JSON block if surrounded by extra text.
        """
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        # Extract first {...} block in case of preamble
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        data = json.loads(raw)
        fact_results = data.get("facts", [])

        # Validate structure
        if not isinstance(fact_results, list):
            raise ValueError("Judge returned invalid 'facts' structure.")

        # Ensure all expected facts are covered
        if len(fact_results) != len(facts):
            raise ValueError(
                f"Judge returned {len(fact_results)} fact results but expected {len(facts)}."
            )

        for item in fact_results:
            if "fact" not in item or "supported" not in item:
                raise KeyError("Missing 'fact' or 'supported' in judge result entry.")
            if not isinstance(item["supported"], bool):
                raise ValueError("'supported' field must be a boolean.")

        return fact_results

    def evaluate(
        self,
        test_cases: list[dict],
        top_k: int = 3,
        score_threshold: float = 0.40
    ) -> dict:
        """
        Run the full RAG pipeline and evaluate answer accuracy for all test cases.

        Args:
            test_cases:      Loaded golden dataset entries.
            top_k:           Number of chunks to retrieve.
            score_threshold: Relevance gate threshold.

        Returns:
            dict with per_query results and aggregate summary.
        """
        per_query_results = []

        for case in test_cases:
            test_id = case["test_id"]
            question = case["question"]
            answerable = case["answerable"]
            expected_sources = case.get("expected_sources", [])
            expected_facts = case.get("expected_facts", [])

            # Step 1: Embed query
            try:
                query_vec = self.embedder.embed_query(question)
            except Exception as e:
                raise RuntimeError(f"GENERATION_ENGINE_UNAVAILABLE: Embedding failed for '{test_id}': {e}")

            # Step 2: Retrieve and gate
            try:
                gated = self.retriever.retrieve_and_gate(
                    query_vec, top_k=top_k, score_threshold=score_threshold
                )
            except Exception as e:
                raise RuntimeError(f"GENERATION_ENGINE_UNAVAILABLE: Retrieval failed for '{test_id}': {e}")

            accepted_chunks = gated.get("results", [])
            retrieved_sources = list(set(
                c.get("metadata", {}).get("file_name", "Unknown")
                for c in accepted_chunks
            ))
            has_context = len(accepted_chunks) > 0

            # Retrieval success = expected source appears in retrieved sources
            if answerable and expected_sources:
                retrieval_success = any(src in retrieved_sources for src in expected_sources)
            else:
                retrieval_success = has_context

            # Step 3: Build context prompt
            try:
                context_package = self.context_builder.build(question, accepted_chunks)
            except Exception as e:
                raise RuntimeError(f"GENERATION_ENGINE_UNAVAILABLE: Context building failed for '{test_id}': {e}")

            # Step 4: Generate answer
            try:
                gen_result = self.generation_engine.generate(context_package)
                generated_answer = gen_result.get("answer", "").strip()
            except Exception as e:
                raise RuntimeError(f"GENERATION_ENGINE_UNAVAILABLE: Generation failed for '{test_id}': {e}")

            # Step 5: Evaluate answer
            if not answerable:
                # Evaluate abstention
                abstention_class = classify_abstention(generated_answer)
                per_query_results.append({
                    "test_id": test_id,
                    "question": question,
                    "answerable": False,
                    "retrieval_success": retrieval_success,
                    "retrieved_sources": retrieved_sources,
                    "generated_answer": generated_answer,
                    "expected_facts": [],
                    "fact_results": [],
                    "facts_supported": None,
                    "facts_expected": None,
                    "accuracy_class": abstention_class,
                    "failure_mode": None,
                    "evaluation_method": "ABSTENTION_CLASSIFIER"
                })
            else:
                # Evaluate fact matching
                if not expected_facts:
                    # No facts to verify — mark CORRECT if answer is non-empty
                    fact_results = []
                    accuracy_class = "CORRECT" if generated_answer else "INCORRECT"
                    evaluation_method = "NO_FACTS"
                else:
                    fact_results, evaluation_method = self._call_judge(
                        question, expected_facts, generated_answer
                    )

                facts_supported = sum(1 for f in fact_results if f.get("supported"))
                facts_expected = len(expected_facts)
                accuracy_class = classify_accuracy(facts_supported, facts_expected)
                failure_mode = classify_failure_mode(retrieval_success, accuracy_class)

                per_query_results.append({
                    "test_id": test_id,
                    "question": question,
                    "answerable": True,
                    "retrieval_success": retrieval_success,
                    "retrieved_sources": retrieved_sources,
                    "generated_answer": generated_answer,
                    "expected_facts": expected_facts,
                    "fact_results": fact_results,
                    "facts_supported": facts_supported,
                    "facts_expected": facts_expected,
                    "accuracy_class": accuracy_class,
                    "failure_mode": failure_mode,
                    "evaluation_method": evaluation_method
                })

        # Aggregate metrics
        answerable_results = [r for r in per_query_results if r["answerable"]]
        unanswerable_results = [r for r in per_query_results if not r["answerable"]]

        correct = sum(1 for r in answerable_results if r["accuracy_class"] == "CORRECT")
        partial = sum(1 for r in answerable_results if r["accuracy_class"] == "PARTIALLY_CORRECT")
        incorrect = sum(1 for r in answerable_results if r["accuracy_class"] == "INCORRECT")

        correct_abstention = sum(1 for r in unanswerable_results if r["accuracy_class"] == "CORRECT_ABSTENTION")
        hallucinated = sum(1 for r in unanswerable_results if r["accuracy_class"] == "HALLUCINATED_ANSWER")
        unsafe_abstention = sum(1 for r in unanswerable_results if r["accuracy_class"] == "UNSAFE_ABSTENTION")

        e2e_success = sum(1 for r in answerable_results if r.get("failure_mode") == "END_TO_END_SUCCESS")
        gen_failure = sum(1 for r in answerable_results if r.get("failure_mode") == "GENERATION_FAILURE")
        ret_failure = sum(1 for r in answerable_results if r.get("failure_mode") == "RETRIEVAL_FAILURE")

        total_ans = len(answerable_results) or 1
        total_unans = len(unanswerable_results) or 1

        return {
            "config": {
                "top_k": top_k,
                "score_threshold": score_threshold
            },
            "summary": {
                "total_queries": len(per_query_results),
                "answerable_count": len(answerable_results),
                "unanswerable_count": len(unanswerable_results),
                "correct": correct,
                "partially_correct": partial,
                "incorrect": incorrect,
                "answer_accuracy": correct / total_ans,
                "partial_accuracy_rate": partial / total_ans,
                "incorrect_rate": incorrect / total_ans,
                "correct_abstention": correct_abstention,
                "hallucinated_answers": hallucinated,
                "unsafe_abstentions": unsafe_abstention,
                "correct_abstention_rate": correct_abstention / total_unans,
                "hallucination_rate": hallucinated / total_unans,
                "end_to_end_success": e2e_success,
                "generation_failures": gen_failure,
                "retrieval_failures": ret_failure
            },
            "per_query": per_query_results
        }

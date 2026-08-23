import time
from .metric_calculator import (
    calculate_hit_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_source_hit_at_k,
    calculate_source_recall_at_k,
    calculate_false_retrieval,
    calculate_mrr,
    calculate_false_retrieval_rate
)

class RetrievalEvaluator:
    """
    Retrieval Evaluator that runs queries, applies relevance gating,
    and calls pure metric functions to produce detailed evaluation statistics.
    """
    def __init__(self, retriever, embedder):
        """
        Initializes the RetrievalEvaluator.
        
        Args:
            retriever (RetrievalConnector): The production retrieval connector.
            embedder (EmbeddingEngine): The production embedding engine.
        """
        self.retriever = retriever
        self.embedder = embedder

    def evaluate(self, test_cases: list[dict], top_k: int = 3, score_threshold: float = None) -> dict:
        """
        Runs the evaluation dataset queries and computes metrics.
        
        Args:
            test_cases (list[dict]): List of validated test cases.
            top_k (int): K retrieved chunks to consider.
            score_threshold (float): Similarity threshold for the relevance gate.
            
        Returns:
            dict: Structured evaluation results and summary metrics.
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"Invalid K value: {top_k}. Must be a positive integer.")
            
        if score_threshold is not None:
            if not isinstance(score_threshold, (int, float)) or not (0.0 <= score_threshold <= 1.0):
                raise ValueError(f"Invalid score_threshold: {score_threshold}. Must be between 0.0 and 1.0.")

        results = []

        for case in test_cases:
            test_id = case["test_id"]
            question = case["question"]
            answerable = case["answerable"]
            expected_chunk_ids = case["expected_chunk_ids"]
            expected_sources = case["expected_sources"]

            # Error handling for query embedding and retrieval failures
            try:
                query_vec = self.embedder.embed_query(question)
                if query_vec is None:
                    raise ValueError("Query embedding generation returned None.")
            except Exception as e:
                # Proactively raise execution failure
                raise RuntimeError(f"Embedding failure for query '{question}': {e}")

            try:
                # Query database
                candidates = self.retriever.search(query_vec, top_k=top_k, score_threshold=None)
                # Query gating
                gated_res = self.retriever.retrieve_and_gate(query_vec, top_k=top_k, score_threshold=score_threshold)
            except Exception as e:
                raise RuntimeError(f"Retrieval engine failure for query '{question}': {e}")

            # Parse search results
            retrieved_chunk_ids = []
            retrieved_sources = []
            retrieval_scores = []

            for chunk in candidates:
                # Handle possible missing keys gracefully
                cid = chunk.get("chunk_id", "unknown_id")
                retrieved_chunk_ids.append(cid)
                
                score = chunk.get("score", 0.0)
                retrieval_scores.append(score)

                meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                file_name = meta.get("file_name") if meta else chunk.get("file_name")
                retrieved_sources.append(file_name if file_name else "Unknown")

            accepted_chunks = gated_res.get("results", [])
            accepted_chunks_count = len(accepted_chunks)
            accepted_chunk_ids = [c["chunk_id"] for c in accepted_chunks]
            accepted_sources = []
            for c in accepted_chunks:
                meta = c.get("metadata", {}) if isinstance(c, dict) else {}
                file_name = meta.get("file_name") if meta else c.get("file_name")
                accepted_sources.append(file_name if file_name else "Unknown")

            # Compute pure metrics based on accepted (gated) chunks
            hit = calculate_hit_at_k(accepted_chunk_ids, expected_chunk_ids, answerable)
            recall = calculate_recall_at_k(accepted_chunk_ids, expected_chunk_ids, answerable)
            rr = calculate_reciprocal_rank(accepted_chunk_ids, expected_chunk_ids, answerable)
            src_hit = calculate_source_hit_at_k(accepted_sources, expected_sources, answerable)
            src_recall = calculate_source_recall_at_k(accepted_sources, expected_sources, answerable)
            false_retrieval = calculate_false_retrieval(answerable, accepted_chunks)

            case_result = {
                "test_id": test_id,
                "question": question,
                "answerable": answerable,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "expected_chunk_ids": expected_chunk_ids,
                "expected_sources": expected_sources,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_sources": retrieved_sources,
                "retrieval_scores": retrieval_scores,
                "accepted_chunks": accepted_chunks_count,
                "hit_at_k": hit,
                "recall_at_k": recall,
                "reciprocal_rank": rr,
                "source_hit_at_k": src_hit,
                "source_recall_at_k": src_recall,
                "false_retrieval": false_retrieval
            }
            results.append(case_result)

        # Aggregated Metrics Calculation
        hit_rates = [r["hit_at_k"] for r in results if r["hit_at_k"] is not None]
        recalls = [r["recall_at_k"] for r in results if r["recall_at_k"] is not None]
        reciprocal_ranks = [r["reciprocal_rank"] for r in results if r["reciprocal_rank"] is not None]
        source_hits = [r["source_hit_at_k"] for r in results if r["source_hit_at_k"] is not None]
        source_recalls = [r["source_recall_at_k"] for r in results if r["source_recall_at_k"] is not None]
        false_retrieval_list = [r["false_retrieval"] for r in results if not r["answerable"]]

        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        mrr = calculate_mrr(reciprocal_ranks)
        avg_source_hit = sum(source_hits) / len(source_hits) if source_hits else 0.0
        avg_source_recall = sum(source_recalls) / len(source_recalls) if source_recalls else 0.0
        false_retrieval_rate = calculate_false_retrieval_rate(false_retrieval_list)

        return {
            "top_k": top_k,
            "score_threshold": score_threshold,
            "summary": {
                "total_queries": len(test_cases),
                "answerable_count": len(hit_rates),
                "unanswerable_count": len(false_retrieval_list),
                "hit_rate": avg_hit_rate,
                "recall": avg_recall,
                "mrr": mrr,
                "source_hit_rate": avg_source_hit,
                "source_recall": avg_source_recall,
                "false_retrieval_count": sum(1 for fr in false_retrieval_list if fr),
                "false_retrieval_rate": false_retrieval_rate
            },
            "results": results
        }

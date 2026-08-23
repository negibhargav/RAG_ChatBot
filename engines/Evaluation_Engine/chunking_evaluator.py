import statistics
import Vector_Store_Engine
import Chunking_Engine
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

class ChunkingEvaluator:
    """
    Chunking Evaluator that sweeps chunk_size and chunk_overlap configurations,
    calculates stats and quality metrics, and runs isolated retrieval evaluations.
    """
    def __init__(self, embedder):
        """
        Initializes the ChunkingEvaluator.
        
        Args:
            embedder (EmbeddingEngine): Production embedding engine.
        """
        self.embedder = embedder

    def evaluate_chunk_config(
        self, 
        documents: list[dict], 
        test_cases: list[dict], 
        chunk_size: int, 
        chunk_overlap: int,
        score_threshold: float = 0.40
    ) -> dict:
        """
        Ingests documents using specified chunk config into a temp collection,
        runs evaluation queries, computes stats and metrics, and cleans up.
        """
        # Validate parameters
        if chunk_size <= 0:
            raise ValueError(f"Invalid chunk_size: {chunk_size}. Must be positive.")
        if chunk_overlap < 0:
            raise ValueError(f"Invalid chunk_overlap: {chunk_overlap}. Cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"Overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size}).")

        col_name = f"rag_chunk_eval_{chunk_size}_{chunk_overlap}"
        db = Vector_Store_Engine.ChromaVectorStore(db_dir="Vector_DB", collection_name=col_name)

        # Ingest and gather statistics
        all_chunks = []
        for doc in documents:
            try:
                chunks = Chunking_Engine.split_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                # Cleanup DB collection before raising error
                try:
                    db.client.delete_collection(col_name)
                except Exception:
                    pass
                raise RuntimeError(f"Chunking failed for doc '{doc.get('document_id')}': {e}")

        if not all_chunks:
            try:
                db.client.delete_collection(col_name)
            except Exception:
                pass
            raise ValueError("No chunks generated from evaluation documents.")

        # Calculate statistics
        lengths = [len(c["text"]) for c in all_chunks]
        avg_len = sum(lengths) / len(lengths)
        median_len = statistics.median(lengths)
        min_len = min(lengths)
        max_len = max(lengths)

        very_small_count = sum(1 for l in lengths if l < 100)
        close_to_max_count = sum(1 for l in lengths if l >= 0.9 * chunk_size)

        # Chunk Quality Heuristics
        incomplete_sentences_count = 0
        orphaned_headings_count = 0

        for chunk in all_chunks:
            text = chunk["text"].strip()
            # 1. Incomplete Sentence check
            if text and text[-1] not in [".", "?", "!"]:
                incomplete_sentences_count += 1
            
            # 2. Orphaned Heading check: ends with a short line, capital starting, no punctuation
            lines = text.split("\n")
            if lines:
                last_line = lines[-1].strip()
                if last_line and len(last_line) < 40 and not any(char in last_line for char in [".", "?", "!"]):
                    if last_line[0].isupper() or last_line.isupper():
                        orphaned_headings_count += 1

        # Save to database
        try:
            embedded = self.embedder.embed_chunks(all_chunks)
            db.save_chunks(embedded)
        except Exception as e:
            try:
                db.client.delete_collection(col_name)
            except Exception:
                pass
            raise RuntimeError(f"Embedding/Saving failed during chunk evaluation: {e}")

        # Run retrieval queries
        results = []
        for case in test_cases:
            question = case["question"]
            answerable = case["answerable"]
            expected_sources = case["expected_sources"]
            expected_content = case.get("expected_content", [])

            # Dynamic expected chunks resolution from facts matching database chunks
            expected_chunk_ids = []
            if answerable:
                for chunk in all_chunks:
                    chunk_text = chunk.get("text", "").lower()
                    for fact in expected_content:
                        if fact.lower() in chunk_text:
                            expected_chunk_ids.append(chunk["chunk_id"])
                            break

            # Search retrieval (limit to K=5 to get candidates for Hit@1,3,5)
            query_vec = self.embedder.embed_query(question)
            
            # We query the database. search() will raise an error if collection is empty.
            candidates = db.search(query_vec, top_k=5, score_threshold=None)

            # Get accepted chunks under the specific score threshold
            # Distance returned is cosine distance: D = 1 - similarity
            accepted_chunks = []
            for c in candidates:
                score = c.get("score", 0.0)
                if score >= score_threshold:
                    accepted_chunks.append(c)

            retrieved_chunk_ids = [c["chunk_id"] for c in candidates]
            retrieved_sources = []
            for c in candidates:
                meta = c.get("metadata", {})
                file_name = meta.get("file_name") if meta else c.get("file_name")
                retrieved_sources.append(file_name if file_name else "Unknown")

            accepted_chunk_ids = [c["chunk_id"] for c in accepted_chunks]
            accepted_sources = []
            for c in accepted_chunks:
                meta = c.get("metadata", {})
                file_name = meta.get("file_name") if meta else c.get("file_name")
                accepted_sources.append(file_name if file_name else "Unknown")

            # Slice lists for K values
            # K=1
            hit_1 = calculate_hit_at_k(accepted_chunk_ids[:1], expected_chunk_ids, answerable)
            recall_1 = calculate_recall_at_k(accepted_chunk_ids[:1], expected_chunk_ids, answerable)

            # K=3
            hit_3 = calculate_hit_at_k(accepted_chunk_ids[:3], expected_chunk_ids, answerable)
            recall_3 = calculate_recall_at_k(accepted_chunk_ids[:3], expected_chunk_ids, answerable)
            src_hit_3 = calculate_source_hit_at_k(accepted_sources[:3], expected_sources, answerable)
            src_recall_3 = calculate_source_recall_at_k(accepted_sources[:3], expected_sources, answerable)

            # K=5
            hit_5 = calculate_hit_at_k(accepted_chunk_ids[:5], expected_chunk_ids, answerable)
            recall_5 = calculate_recall_at_k(accepted_chunk_ids[:5], expected_chunk_ids, answerable)
            rr_5 = calculate_reciprocal_rank(accepted_chunk_ids[:5], expected_chunk_ids, answerable)
            
            # Gating Abstention (False Retrieval is based on K=3 gate acceptance)
            false_retrieval = calculate_false_retrieval(answerable, accepted_chunks[:3])

            results.append({
                "hit_at_1": hit_1,
                "recall_at_1": recall_1,
                "hit_at_3": hit_3,
                "recall_at_3": recall_3,
                "src_hit_3": src_hit_3,
                "src_recall_3": src_recall_3,
                "hit_at_5": hit_5,
                "recall_at_5": recall_5,
                "rr_5": rr_5,
                "false_retrieval": false_retrieval,
                "answerable": answerable
            })

        # Clean up database collection
        try:
            db.client.delete_collection(col_name)
        except Exception:
            pass

        # Aggregate metrics
        answerable_cases = [r for r in results if r["answerable"]]
        unanswerable_cases = [r for r in results if not r["answerable"]]

        hit_rates_1 = [r["hit_at_1"] for r in answerable_cases if r["hit_at_1"] is not None]
        recalls_1 = [r["recall_at_1"] for r in answerable_cases if r["recall_at_1"] is not None]

        hit_rates_3 = [r["hit_at_3"] for r in answerable_cases if r["hit_at_3"] is not None]
        recalls_3 = [r["recall_at_3"] for r in answerable_cases if r["recall_at_3"] is not None]
        src_hits_3 = [r["src_hit_3"] for r in answerable_cases if r["src_hit_3"] is not None]
        src_recalls_3 = [r["src_recall_3"] for r in answerable_cases if r["src_recall_3"] is not None]

        hit_rates_5 = [r["hit_at_5"] for r in answerable_cases if r["hit_at_5"] is not None]
        recalls_5 = [r["recall_at_5"] for r in answerable_cases if r["recall_at_5"] is not None]
        reciprocal_ranks = [r["rr_5"] for r in answerable_cases if r["rr_5"] is not None]

        false_retrievals = [r["false_retrieval"] for r in unanswerable_cases]

        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "document_count": len(documents),
            "total_chunks": len(all_chunks),
            "chunk_statistics": {
                "average_length": avg_len,
                "median_length": median_len,
                "min_length": min_len,
                "max_length": max_len,
                "very_small_chunks": very_small_count,
                "close_to_max_chunks": close_to_max_count,
                "incomplete_sentences": incomplete_sentences_count,
                "orphaned_headings": orphaned_headings_count
            },
            "retrieval_metrics": {
                "hit_at_1": sum(hit_rates_1) / len(hit_rates_1) if hit_rates_1 else 0.0,
                "hit_at_3": sum(hit_rates_3) / len(hit_rates_3) if hit_rates_3 else 0.0,
                "hit_at_5": sum(hit_rates_5) / len(hit_rates_5) if hit_rates_5 else 0.0,
                "recall_at_1": sum(recalls_1) / len(recalls_1) if recalls_1 else 0.0,
                "recall_at_3": sum(recalls_3) / len(recalls_3) if recalls_3 else 0.0,
                "recall_at_5": sum(recalls_5) / len(recalls_5) if recalls_5 else 0.0,
                "mrr": calculate_mrr(reciprocal_ranks),
                "source_hit_at_3": sum(src_hits_3) / len(src_hits_3) if src_hits_3 else 0.0,
                "source_recall_at_3": sum(src_recalls_3) / len(src_recalls_3) if src_recalls_3 else 0.0,
                "false_retrieval_rate": calculate_false_retrieval_rate(false_retrievals)
            },
            "embedding_workload": {
                "chunk_count": len(all_chunks)
            }
        }

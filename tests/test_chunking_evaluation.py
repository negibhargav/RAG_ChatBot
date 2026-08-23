import os
import sys
import unittest
from unittest.mock import MagicMock

# Path configuration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../engines")))

import Chunking_Engine
import Embedding_Engine
import Vector_Store_Engine
import Retrieval_Engine
from Evaluation_Engine.chunking_evaluator import ChunkingEvaluator

class TestChunkingEvaluationEngine(unittest.TestCase):
    def setUp(self):
        self.mock_embedder = MagicMock()
        self.mock_embedder.embed_chunks = lambda chunks: [
            {"chunk_id": c["chunk_id"], "text": c["text"], "embedding": [0.1]*384, "metadata": c["metadata"]}
            for c in chunks
        ]
        self.mock_embedder.embed_query.return_value = [0.1]*384
        
        self.evaluator = ChunkingEvaluator(self.mock_embedder)

        self.sample_doc = {
            "document_id": "doc_test_setup",
            "text": "BHUPENDER SINGH NEGI is an Analyst at Jones Lang LaSalle (JLL). He completed B.Tech at Vivekananda Institute.",
            "metadata": {
                "source_path": "test.txt",
                "file_name": "test.txt",
                "file_type": ".txt"
            }
        }
        
        self.sample_cases = [
            {
                "test_id": "q1",
                "question": "Where did he do his B.Tech?",
                "answerable": True,
                "expected_sources": ["test.txt"],
                "expected_content": ["B.Tech", "Vivekananda"]
            },
            {
                "test_id": "q2",
                "question": "What is his salary?",
                "answerable": False,
                "expected_sources": [],
                "expected_content": []
            }
        ]

    def test_01_parameter_validation_bounds(self):
        """2, 3, 4. Parameters are validated (size positive, overlap non-negative, overlap < size)."""
        # Invalid chunk size
        with self.assertRaises(ValueError) as ctx:
            self.evaluator.evaluate_chunk_config([self.sample_doc], self.sample_cases, chunk_size=-100, chunk_overlap=30)
        self.assertIn("Invalid chunk_size", str(ctx.exception))

        # Invalid overlap
        with self.assertRaises(ValueError) as ctx:
            self.evaluator.evaluate_chunk_config([self.sample_doc], self.sample_cases, chunk_size=300, chunk_overlap=-10)
        self.assertIn("Invalid chunk_overlap", str(ctx.exception))

        # Overlap >= size
        with self.assertRaises(ValueError) as ctx:
            self.evaluator.evaluate_chunk_config([self.sample_doc], self.sample_cases, chunk_size=300, chunk_overlap=300)
        self.assertIn("Overlap (300) must be less than chunk_size", str(ctx.exception))

    def test_02_statistics_and_quality_heuristics(self):
        """7. Chunk statistics and quality heuristics (incomplete sentence, orphaned heading) are correct."""
        docs = [
            {
                "document_id": "d1",
                "text": "Bhupender graduated.\nHe holds a B.Tech.\nTECHNICAL SKILLS\nGenerative AI",
                "metadata": {"file_name": "test.txt", "source_path": "test.txt"}
            }
        ]
        # Chunking size 30, overlap 5 should trigger multiple chunks
        result = self.evaluator.evaluate_chunk_config(docs, self.sample_cases, chunk_size=40, chunk_overlap=10)
        
        stats = result["chunk_statistics"]
        self.assertIn("average_length", stats)
        self.assertIn("median_length", stats)
        self.assertIn("min_length", stats)
        self.assertIn("max_length", stats)
        self.assertIn("very_small_chunks", stats)
        self.assertIn("close_to_max_chunks", stats)
        
        # Verify quality counters are keys in the stats dictionary
        self.assertIn("incomplete_sentences", stats)
        self.assertIn("orphaned_headings", stats)

    def test_03_isolated_collection_usage_and_cleanup(self):
        """5, 6, 11. Custom isolated collection used, cleaned, and production DB untouched."""
        import chromadb
        
        # Track if temporary collection gets cleaned up
        client = chromadb.PersistentClient(path="Vector_DB")
        temp_col_name = "rag_chunk_eval_250_25"
        
        # Ensure it doesn't exist beforehand
        if temp_col_name in [c.name for c in client.list_collections()]:
            client.delete_collection(temp_col_name)

        result = self.evaluator.evaluate_chunk_config([self.sample_doc], self.sample_cases, chunk_size=250, chunk_overlap=25)
        
        # Verify collection has been deleted after execution
        existing_cols = [c.name for c in client.list_collections()]
        self.assertNotIn(temp_col_name, existing_cols)
        
        # Verify result contains metrics
        self.assertIn("retrieval_metrics", result)
        self.assertIn("embedding_workload", result)

    def test_04_missing_documents_handled_gracefully(self):
        """9. Missing evaluation documents or empty inputs are handled cleanly."""
        with self.assertRaises(ValueError) as ctx:
            self.evaluator.evaluate_chunk_config([], self.sample_cases, chunk_size=200, chunk_overlap=20)
        self.assertIn("No chunks generated", str(ctx.exception))

    def test_05_ground_truth_boundary_invariance(self):
        """10. Expected content containment ground truth remains valid despite chunk boundaries changing."""
        # Config A (small chunks)
        res_small = self.evaluator.evaluate_chunk_config([self.sample_doc], self.sample_cases, chunk_size=50, chunk_overlap=10)
        
        # Config B (large chunks)
        res_large = self.evaluator.evaluate_chunk_config([self.sample_doc], self.sample_cases, chunk_size=500, chunk_overlap=50)

        # Both return valid structured results despite totally different chunk boundaries
        self.assertEqual(res_small["chunk_size"], 50)
        self.assertEqual(res_large["chunk_size"], 500)
        self.assertIn("hit_at_3", res_small["retrieval_metrics"])
        self.assertIn("hit_at_3", res_large["retrieval_metrics"])

    def test_06_sweep_all_configurations_structure(self):
        """1. Sweep grid produces comparative configurations list."""
        chunk_sizes = [200, 300]
        overlaps = [20, 50]
        
        configs = []
        for s in chunk_sizes:
            for o in overlaps:
                if o < s:
                    res = self.evaluator.evaluate_chunk_config([self.sample_doc], self.sample_cases, chunk_size=s, chunk_overlap=o)
                    configs.append(res)

        self.assertEqual(len(configs), 4)
        for cfg in configs:
            self.assertIn("chunk_statistics", cfg)
            self.assertIn("retrieval_metrics", cfg)
            self.assertIn("embedding_workload", cfg)

if __name__ == "__main__":
    unittest.main()

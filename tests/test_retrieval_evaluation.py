import os
import sys
import unittest
import tempfile
import json
from unittest.mock import MagicMock

# Inject path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../engines")))

from Evaluation_Engine.dataset_loader import EvaluationDatasetLoader
from Evaluation_Engine.metric_calculator import (
    calculate_hit_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_source_hit_at_k,
    calculate_source_recall_at_k,
    calculate_false_retrieval,
    calculate_mrr,
    calculate_false_retrieval_rate
)
from Evaluation_Engine.retrieval_evaluator import RetrievalEvaluator

class TestRetrievalEvaluationPureMetrics(unittest.TestCase):
    def test_calculate_hit_at_k(self):
        """Test Hit@K logic: Hit@3 == 1 when relevant chunk retrieved."""
        expected = ["A"]
        retrieved = ["B", "A", "C"]
        self.assertEqual(calculate_hit_at_k(retrieved, expected, answerable=True), 1)

        # No match Hit@K == 0
        retrieved_miss = ["B", "C", "D"]
        self.assertEqual(calculate_hit_at_k(retrieved_miss, expected, answerable=True), 0)

        # Unanswerable query returns None
        self.assertIsNone(calculate_hit_at_k(retrieved, expected, answerable=False))

    def test_calculate_recall_at_k(self):
        """Test Recall@K logic: Recall@3 == 2/3 when 2 out of 3 retrieved."""
        expected = ["A", "B", "C"]
        retrieved = ["A", "D", "B"]
        self.assertAlmostEqual(calculate_recall_at_k(retrieved, expected, answerable=True), 2.0 / 3.0)

        # Unanswerable query returns None (as specified in user requirements)
        self.assertIsNone(calculate_recall_at_k(retrieved, expected, answerable=False))

    def test_calculate_reciprocal_rank(self):
        """Test Reciprocal Rank (RR): RR == 0.5 when first relevant rank is 2."""
        expected = ["A"]
        retrieved = ["B", "A", "C"]
        self.assertEqual(calculate_reciprocal_rank(retrieved, expected, answerable=True), 0.5)

        # RR == 0 when no match
        retrieved_miss = ["B", "C"]
        self.assertEqual(calculate_reciprocal_rank(retrieved_miss, expected, answerable=True), 0.0)

        # Unanswerable query returns None
        self.assertIsNone(calculate_reciprocal_rank(retrieved, expected, answerable=False))

    def test_calculate_source_hit_at_k(self):
        """Test Source Hit@K separately from chunk-level."""
        expected_src = ["test.txt"]
        retrieved_src = ["assignment.docx", "test.txt", "other.pdf"]
        self.assertEqual(calculate_source_hit_at_k(retrieved_src, expected_src, answerable=True), 1)

        retrieved_miss = ["other.pdf"]
        self.assertEqual(calculate_source_hit_at_k(retrieved_miss, expected_src, answerable=True), 0)

    def test_calculate_source_recall_at_k(self):
        """Test Source Recall@K separately from chunk-level."""
        expected_src = ["docA.txt", "docB.txt"]
        retrieved_src = ["docA.txt", "other.txt", "docB.txt"]
        self.assertEqual(calculate_source_recall_at_k(retrieved_src, expected_src, answerable=True), 1.0)

        retrieved_partial = ["docA.txt"]
        self.assertEqual(calculate_source_recall_at_k(retrieved_partial, expected_src, answerable=True), 0.5)

    def test_calculate_false_retrieval(self):
        """Test false retrieval logic for unanswerable queries."""
        # Unanswerable with no chunks accepted -> False
        self.assertFalse(calculate_false_retrieval(answerable=False, accepted_chunks=[]))

        # Unanswerable with accepted chunks -> True
        self.assertTrue(calculate_false_retrieval(answerable=False, accepted_chunks=["chunk_1"]))

        # Answerable query always returns False
        self.assertFalse(calculate_false_retrieval(answerable=True, accepted_chunks=["chunk_1"]))

    def test_calculate_mrr(self):
        """Test Mean Reciprocal Rank (MRR) aggregate metric."""
        rrs = [1.0, 0.5, None, 0.0]  # Contains unanswerable queries (None)
        # Valid MRR is (1.0 + 0.5 + 0.0) / 3 = 0.5
        self.assertAlmostEqual(calculate_mrr(rrs), 0.5)

    def test_calculate_false_retrieval_rate(self):
        """Test False Retrieval Rate aggregate metric."""
        frs = [True, False, None, True] # Contains None from answerable queries
        # Rate is (1.0 + 0.0 + 1.0) / 3 = 2/3 = 0.667
        self.assertAlmostEqual(calculate_false_retrieval_rate(frs), 2.0 / 3.0)


class TestRetrievalDatasetLoaderAndEvaluator(unittest.TestCase):
    def setUp(self):
        self.temp_files = []

    def tearDown(self):
        for path in self.temp_files:
            if os.path.exists(path):
                os.remove(path)

    def create_temp_dataset(self, data_list):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f)
        self.temp_files.append(path)
        return path

    def test_dataset_validation_duplicates(self):
        """Loader rejects duplicate test IDs."""
        data = [
            {
                "test_id": "q001",
                "question": "Q1?",
                "answerable": True,
                "expected_chunk_ids": ["c1"],
                "expected_sources": ["s1"],
                "expected_content": ["f1"],
                "expected_facts": ["fact1"]
            },
            {
                "test_id": "q001",
                "question": "Q2?",
                "answerable": True,
                "expected_chunk_ids": ["c2"],
                "expected_sources": ["s2"],
                "expected_content": ["f2"],
                "expected_facts": ["fact2"]
            }
        ]
        path = self.create_temp_dataset(data)
        with self.assertRaises(ValueError) as ctx:
            EvaluationDatasetLoader.load(path)
        self.assertIn("Duplicate test_id", str(ctx.exception))

    def test_dataset_validation_missing_key(self):
        """Loader rejects missing required fields."""
        data = [
            {
                "test_id": "q001",
                "question": "Q1?",
                "answerable": True,
                # Missing expected_sources and expected_content
                "expected_chunk_ids": ["c1"]
            }
        ]
        path = self.create_temp_dataset(data)
        with self.assertRaises(ValueError):
            EvaluationDatasetLoader.load(path)

    def test_dataset_validation_unanswerable_fields(self):
        """Loader rejects unanswerable case with non-empty expected chunks."""
        data = [
            {
                "test_id": "q001",
                "question": "Q1?",
                "answerable": False,
                "expected_chunk_ids": ["c1"],  # Must be empty
                "expected_sources": [],
                "expected_content": [],
                "expected_facts": []
            }
        ]
        path = self.create_temp_dataset(data)
        with self.assertRaises(ValueError) as ctx:
            EvaluationDatasetLoader.load(path)
        self.assertIn("must have empty expected_chunk_ids", str(ctx.exception))

    def test_edge_cases_empty_lists(self):
        """Metric calculations handle empty arrays gracefully."""
        self.assertEqual(calculate_hit_at_k([], [], answerable=True), 0)
        self.assertEqual(calculate_recall_at_k([], [], answerable=True), 0.0)
        self.assertEqual(calculate_reciprocal_rank([], [], answerable=True), 0.0)
        self.assertEqual(calculate_source_hit_at_k([], [], answerable=True), 0)
        self.assertEqual(calculate_source_recall_at_k([], [], answerable=True), 0.0)

    def test_evaluator_error_propagation(self):
        """Evaluator propagates query embedding and search failures."""
        test_cases = [
            {
                "test_id": "q001",
                "question": "Error?",
                "answerable": True,
                "expected_chunk_ids": ["c1"],
                "expected_sources": ["s1"]
            }
        ]
        
        # Test Query Embedding Failure
        mock_embedder = MagicMock()
        mock_embedder.embed_query.side_effect = RuntimeError("Mock Embedding API Out of Service")
        mock_retriever = MagicMock()
        
        evaluator = RetrievalEvaluator(mock_retriever, mock_embedder)
        with self.assertRaises(RuntimeError) as ctx:
            evaluator.evaluate(test_cases, top_k=3, score_threshold=0.40)
        self.assertIn("Embedding failure", str(ctx.exception))

        # Test Search Retrieval Failure
        mock_embedder_ok = MagicMock()
        mock_embedder_ok.embed_query.return_value = [0.1] * 384
        
        mock_retriever_fail = MagicMock()
        mock_retriever_fail.search.side_effect = RuntimeError("Mock Database Connection Lost")

        evaluator2 = RetrievalEvaluator(mock_retriever_fail, mock_embedder_ok)
        with self.assertRaises(RuntimeError) as ctx:
            evaluator2.evaluate(test_cases, top_k=3, score_threshold=0.40)
        self.assertIn("Retrieval engine failure", str(ctx.exception))


class TestEvaluationIntegration(unittest.TestCase):
    def test_retrieval_integration_pipeline(self):
        """Integration test querying isolated ChromaDB evaluation collection."""
        import Chunking_Engine
        import Embedding_Engine
        import Vector_Store_Engine
        import Retrieval_Engine

        db_dir = "Vector_DB"
        collection_name = "test_evaluation_pure_temp"
        db = Vector_Store_Engine.ChromaVectorStore(db_dir=db_dir, collection_name=collection_name)

        doc = {
            "document_id": "doc_test_setup",
            "text": "BHUPENDER SINGH NEGI is an Analyst at Jones Lang LaSalle (JLL) in Delhi/NCR, India.",
            "metadata": {
                "source_path": "test.txt",
                "file_name": "test.txt",
                "file_type": ".txt"
            }
        }
        chunks = Chunking_Engine.split_document(doc, chunk_size=300, chunk_overlap=30)
        
        embedder = Embedding_Engine.EmbeddingEngine()
        embedded_chunks = embedder.embed_chunks(chunks)
        db.save_chunks(embedded_chunks)

        retriever = Retrieval_Engine.RetrievalConnector(db)
        evaluator = RetrievalEvaluator(retriever, embedder)

        test_cases = [
            {
                "test_id": "q_int_1",
                "question": "What is Bhupender's current role?",
                "answerable": True,
                "expected_chunk_ids": ["doc_test_setup_chunk_0"],
                "expected_sources": ["test.txt"]
            },
            {
                "test_id": "q_int_2",
                "question": "Who is the Prime Minister of UK?",
                "answerable": False,
                "expected_chunk_ids": [],
                "expected_sources": []
            }
        ]

        result = evaluator.evaluate(test_cases, top_k=3, score_threshold=0.40)
        
        # Verify structure
        self.assertEqual(result["top_k"], 3)
        self.assertEqual(result["summary"]["total_queries"], 2)
        self.assertEqual(result["summary"]["answerable_count"], 1)
        self.assertEqual(result["summary"]["unanswerable_count"], 1)

        # Cleanup collection
        try:
            db.client.delete_collection(collection_name)
        except Exception:
            pass

if __name__ == "__main__":
    unittest.main()

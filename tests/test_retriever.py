import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../engines")))
import unittest
import numpy as np
import Extraction_Engine
import Chunking_Engine
import Embedding_Engine
import Retrieval_Engine

class TestRetrievalConnector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure console to UTF-8 to prevent encoding crashes
        sys.stdout.reconfigure(encoding='utf-8')
        
        # Load and extract test.txt
        file_path = "test.txt"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing '{file_path}' for running semantic retrieval tests.")
            
        text = Extraction_Engine.extract_text(file_path)
        doc_id = "doc_test_setup"
        document = {
            "document_id": doc_id,
            "text": text,
            "metadata": {
                "source_path": os.path.abspath(file_path),
                "file_name": file_path,
                "file_type": ".txt"
            }
        }
        
        # Chunk text
        chunks = Chunking_Engine.split_document(document, chunk_size=300, chunk_overlap=30)
        
        # Initialize Embedder and generate embeddings
        cls.embedder = Embedding_Engine.EmbeddingEngine(model_name='BAAI/bge-small-en-v1.5')
        cls.embedded_chunks = cls.embedder.embed_chunks(chunks)
        cls.dimension = len(cls.embedded_chunks[0]["embedding"])

    def setUp(self):
        # Initialize fresh retriever instance for each test
        self.retriever = Retrieval_Engine.RetrievalConnector(self.embedded_chunks)

    def test_01_basic_retrieval(self):
        """1. Basic retrieval: Query: 'Who is Bhupender Singh Negi?'"""
        query_text = "Who is Bhupender Singh Negi?"
        query_vec = self.embedder.embed_query(query_text)
        
        results = self.retriever.search(query_vec, top_k=3)
        
        print(f"\n[Test 1] Basic Retrieval for query: '{query_text}'")
        for idx, res in enumerate(results):
            print(f"  Rank {idx+1}: Chunk ID: {res['chunk_id']} (Score: {res['score']:.4f})")
            print(f"    Content Preview: {res['text'][:120].strip()}...")
            
        # Verify that Bhupender's name/summary chunk is returned in top results
        found_name = any("BHUPENDER" in res["text"].upper() for res in results)
        self.assertTrue(found_name, "Did not retrieve Bhupender's summary chunk.")

    def test_02_educational_query(self):
        """2. Educational query: Query: 'What is Bhupender's educational background?'"""
        query_text = "What is Bhupender's educational background?"
        query_vec = self.embedder.embed_query(query_text)
        
        results = self.retriever.search(query_vec, top_k=3)
        
        print(f"\n[Test 2] Educational Retrieval for query: '{query_text}'")
        for idx, res in enumerate(results):
            print(f"  Rank {idx+1}: Chunk ID: {res['chunk_id']} (Score: {res['score']:.4f})")
            print(f"    Content Preview: {res['text'][:120].strip()}...")
            
        # Verify that EDUCATION-related content is in the top ranks
        found_edu = any("EDUCATION" in res["text"].upper() or "B.TECH" in res["text"].upper() or "IIT" in res["text"].upper() for res in results)
        self.assertTrue(found_edu, "Did not retrieve educational chunks.")

    def test_03_rag_experience_query(self):
        """3. RAG experience query: Query: 'What experience does Bhupender have with RAG?'"""
        query_text = "What experience does Bhupender have with RAG?"
        query_vec = self.embedder.embed_query(query_text)
        
        results = self.retriever.search(query_vec, top_k=3)
        
        print(f"\n[Test 3] RAG Experience Retrieval for query: '{query_text}'")
        for idx, res in enumerate(results):
            print(f"  Rank {idx+1}: Chunk ID: {res['chunk_id']} (Score: {res['score']:.4f})")
            print(f"    Content Preview: {res['text'][:120].strip()}...")
            
        # Verify that chunks describing RAG experience are ranked highly
        found_rag = any("RAG" in res["text"].upper() or "RETRIEVAL" in res["text"].upper() for res in results)
        self.assertTrue(found_rag, "Did not retrieve chunks discussing RAG experience.")

    def test_04_top_k_behavior(self):
        """4. Top-K behavior: Verify that requesting top_k=3 returns at most 3 results."""
        query_vec = self.embedder.embed_query("Python")
        
        # Test basic limit
        results_3 = self.retriever.search(query_vec, top_k=3)
        self.assertLessEqual(len(results_3), 3)
        
        # Test limit exceeding available chunk count
        large_top_k = len(self.embedded_chunks) + 10
        results_large = self.retriever.search(query_vec, top_k=large_top_k)
        self.assertEqual(len(results_large), len(self.embedded_chunks))

    def test_05_score_ordering(self):
        """5. Score ordering: Verify result[0]["score"] >= result[1]["score"] >= ..."""
        query_vec = self.embedder.embed_query("Machine Learning")
        results = self.retriever.search(query_vec, top_k=10)
        
        for idx in range(len(results) - 1):
            self.assertGreaterEqual(
                results[idx]["score"], 
                results[idx + 1]["score"], 
                f"Scores are not sorted in descending order: {results[idx]['score']} < {results[idx+1]['score']}"
            )

    def test_06_dimension_mismatch(self):
        """6. Dimension mismatch: Pass query vector with incorrect dimension and verify error."""
        bad_query = [0.1] * (self.dimension + 5)
        with self.assertRaises(ValueError) as ctx:
            self.retriever.search(bad_query)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_07_empty_query_vector(self):
        """7. Empty query vector: Verify that invalid empty query is rejected."""
        with self.assertRaises(ValueError) as ctx:
            self.retriever.search([])
        self.assertIn("cannot be empty", str(ctx.exception).lower())

    def test_08_empty_collection(self):
        """8. Empty collection: Verify that retrieval fails clearly on empty vector stores."""
        with self.assertRaises(ValueError) as ctx:
            Retrieval_Engine.RetrievalConnector([])
        self.assertIn("cannot be empty", str(ctx.exception).lower())

if __name__ == "__main__":
    unittest.main()

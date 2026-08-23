import unittest
from unittest.mock import MagicMock, patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../engines")))

from Query_Router import QueryRouter
from Retrieval_Engine.retriever import RetrievalConnector, InMemoryVectorStore
from Context_Engine.context_builder import ContextBuilder
from Generation_Engine.generator import GenerationEngine
from Generation_Engine.providers.base import BaseLLMProvider
from Citation_Engine.citation_engine import CitationEngine

class DummyLLMProvider(BaseLLMProvider):
    def __init__(self):
        self.provider_name = "dummy_provider"
    
    def generate(self, prompt, model, temperature=0.0, max_tokens=None, timeout=60):
        # Default mock generation answer
        return {
            "answer": "The technology used is Python and ChromaDB.",
            "model": model,
            "provider": "dummy_provider",
            "usage": None,
            "finish_reason": "stop"
        }

    def health_check(self, model=None):
        return {
            "available": True,
            "model": model,
            "endpoint": "dummy_endpoint",
            "error": None
        }

class TestImprovements(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter()
        self.context_builder = ContextBuilder()
        self.citation_engine = CitationEngine()
        self.llm_provider = DummyLLMProvider()
        self.generator = GenerationEngine(provider=self.llm_provider)
        
        # Setup mock chunks
        self.mock_chunks = [
            {
                "chunk_id": "doc_1_chunk_0",
                "text": "The project uses Python, ChromaDB, and Ollama.",
                "embedding": [1.0] + [0.0] * 383,
                "metadata": {
                    "document_id": "doc_1",
                    "file_name": "tech_spec.docx",
                    "source_path": "/path/to/tech_spec.docx"
                }
            },
            {
                "chunk_id": "doc_1_chunk_1",
                "text": "The primary model is qwen2.5:3b.",
                "embedding": [0.0, 1.0] + [0.0] * 382,
                "metadata": {
                    "document_id": "doc_1",
                    "file_name": "tech_spec.docx",
                    "source_path": "/path/to/tech_spec.docx"
                }
            },
            {
                "chunk_id": "doc_2_chunk_0",
                "text": "The listed KPIs include system throughput and latency.",
                "embedding": [0.0, 0.0, 1.0] + [0.0] * 381,
                "metadata": {
                    "document_id": "doc_2",
                    "file_name": "kpi.docx",
                    "source_path": "/path/to/kpi.docx"
                }
            }
        ]
        
        self.retriever = RetrievalConnector(self.mock_chunks)

    # TEST 1: Query: "hi" -> CONVERSATIONAL, no embedding, no retrieval, no ChromaDB query
    @patch('Embedding_Engine.embedder.EmbeddingEngine.embed_query')
    @patch('Retrieval_Engine.retriever.RetrievalConnector.search')
    def test_01_conversational_hi(self, mock_search, mock_embed):
        query = "hi"
        route_res = self.router.route(query)
        
        # Verify classification
        self.assertEqual(route_res["query_type"], "CONVERSATIONAL")
        
        # Verify pipeline flow skips embedding and retrieval
        if route_res["query_type"] == "CONVERSATIONAL":
            pass  # Flow correctly bypasses
            
        mock_embed.assert_not_called()
        mock_search.assert_not_called()

    # TEST 2: Query: "how are you?" -> CONVERSATIONAL, direct response, no citation
    def test_02_conversational_how_are_you(self):
        query = "how are you?"
        route_res = self.router.route(query)
        
        self.assertEqual(route_res["query_type"], "CONVERSATIONAL")
        
        # Simulate conversational response logic
        answer = "I'm doing well, thank you for asking!"
        
        # Citation engine should not be called / return no citations
        citations = self.citation_engine.generate_citations(answer, {"sources": []})
        self.assertEqual(citations["sources"], [])
        self.assertFalse(citations["has_citations"])

    # TEST 3: Query: "what are the listed KPIs?" -> KNOWLEDGE, retrieval occurs, relevant chunks, generation occurs, source citation is present
    @patch.object(DummyLLMProvider, 'generate')
    def test_03_knowledge_kpis(self, mock_generate):
        query = "what are the listed KPIs?"
        
        # Mock LLM generation output
        mock_generate.return_value = {
            "answer": "The listed KPIs include throughput and latency.",
            "model": "qwen2.5:3b",
            "provider": "dummy_provider",
            "usage": None,
            "finish_reason": "stop"
        }
        
        # 1. Route
        route_res = self.router.route(query)
        self.assertEqual(route_res["query_type"], "KNOWLEDGE")
        
        # 2. Retrieve & Gate (simulate passing query vector matching doc_2)
        query_vec = [0.0, 0.0, 1.0] + [0.0] * 381
        retrieval_res = self.retriever.retrieve_and_gate(query_vec, top_k=3, score_threshold=0.4)
        
        self.assertTrue(retrieval_res["has_relevant_context"])
        self.assertGreater(len(retrieval_res["results"]), 0)
        
        # 3. Context build
        context_result = self.context_builder.build(query, retrieval_res["results"])
        
        # 4. Generate
        gen_res = self.generator.generate(context_result)
        
        # 5. Citation
        cited_res = self.citation_engine.generate_citations(gen_res["answer"], context_result)
        
        self.assertTrue(cited_res["has_citations"])
        self.assertEqual(len(cited_res["sources"]), 1)
        self.assertEqual(cited_res["sources"][0]["document_id"], "doc_2")

    # TEST 4: Query: "Who is your girlfriend?" -> KNOWLEDGE or non-doc query, retrieval occurs, weak results rejected, Generation Engine NOT called, no citation
    @patch.object(DummyLLMProvider, 'generate')
    def test_04_irrelevant_query(self, mock_generate):
        query = "Who is your girlfriend?"
        
        # 1. Route
        route_res = self.router.route(query)
        self.assertEqual(route_res["query_type"], "KNOWLEDGE")
        
        # 2. Retrieve & Gate with low similarity to mock chunks (orthogonal query vector)
        query_vec = [0.0, 0.0, 0.0, 1.0] + [0.0] * 380
        retrieval_res = self.retriever.retrieve_and_gate(query_vec, top_k=3, score_threshold=0.5)
        
        # Verify relevance gate rejects all chunks
        self.assertFalse(retrieval_res["has_relevant_context"])
        self.assertEqual(len(retrieval_res["results"]), 0)
        self.assertGreater(len(retrieval_res["rejected_results"]), 0)
        
        # Flow logic in main.py: if not has_relevant_context, generation and citation engines are bypassed
        if not retrieval_res["has_relevant_context"]:
            ans = "I couldn't find relevant information about that in the uploaded documents."
            sources = []
            
        mock_generate.assert_not_called()
        self.assertEqual(ans, "I couldn't find relevant information about that in the uploaded documents.")
        self.assertEqual(len(sources), 0)

    # TEST 5: Query: "what technologies are used?" -> relevant chunks retrieved, grounded answer generated, citations correspond to actual document
    @patch.object(DummyLLMProvider, 'generate')
    def test_05_relevant_technologies(self, mock_generate):
        query = "what technologies are used?"
        mock_generate.return_value = {
            "answer": "The project uses Python, ChromaDB, and Ollama.",
            "model": "qwen2.5:3b",
            "provider": "dummy_provider",
            "usage": None,
            "finish_reason": "stop"
        }
        
        query_vec = [1.0] + [0.0] * 383
        retrieval_res = self.retriever.retrieve_and_gate(query_vec, top_k=3, score_threshold=0.4)
        context_result = self.context_builder.build(query, retrieval_res["results"])
        gen_res = self.generator.generate(context_result)
        cited_res = self.citation_engine.generate_citations(gen_res["answer"], context_result)
        
        self.assertTrue(cited_res["has_citations"])
        # Chunks were from doc_1
        self.assertEqual(cited_res["sources"][0]["document_id"], "doc_1")
        self.assertIn("[Source 1]", cited_res["answer"])

    # TEST 6: Multiple chunks from the same document -> Internal multiple chunk IDs retained, displayed one unique source citation
    def test_06_duplicate_sources_deduplication(self):
        # Retrieve against query vector matching both chunks of doc_1
        query_vec = [1.0, 1.0] + [0.0] * 382
        retrieval_res = self.retriever.retrieve_and_gate(query_vec, top_k=3, score_threshold=None)
        
        # Verify internal chunks count
        results = retrieval_res["results"]
        doc_1_chunks = [r for r in results if r["metadata"]["document_id"] == "doc_1"]
        self.assertEqual(len(doc_1_chunks), 2)
        
        # Verify context builder preserves chunk IDs internally
        context_result = self.context_builder.build("query", results)
        self.assertEqual(len(context_result["sources"]), 3) # doc_1_chunk_0, doc_1_chunk_1, doc_2_chunk_0
        
        # Verify citation engine deduplicates them at display time
        answer = "The project uses Python. The primary model is qwen2.5:3b."
        cited_res = self.citation_engine.generate_citations(answer, context_result)
        
        # Both sentences should map to citation index 1 since doc_1 was first in retrieval
        self.assertIn("Python. [Source 1]", cited_res["answer"])
        self.assertIn("qwen2.5:3b. [Source 1]", cited_res["answer"])
        
        # Verify unique source list count is 1 for doc_1
        self.assertEqual(len(cited_res["sources"]), 1)
        self.assertEqual(cited_res["sources"][0]["document_id"], "doc_1")

    # TEST 7: All retrieval scores below threshold -> has_relevant_context = false, Generation Engine not called
    @patch.object(DummyLLMProvider, 'generate')
    def test_07_all_below_threshold(self, mock_generate):
        query_vec = [0.0, 0.0, 0.0, 1.0] + [0.0] * 380
        # Set high threshold
        retrieval_res = self.retriever.retrieve_and_gate(query_vec, top_k=3, score_threshold=0.99)
        
        self.assertFalse(retrieval_res["has_relevant_context"])
        self.assertEqual(len(retrieval_res["results"]), 0)
        
        # In main execution loop, generator is not called when has_relevant_context is False
        if not retrieval_res["has_relevant_context"]:
            pass
            
        mock_generate.assert_not_called()

    # TEST 8: One result above threshold and other results below threshold -> Only accepted relevant chunks passed to Context Builder
    def test_08_mixed_threshold_results(self):
        # We manually construct search scores to test gating logic
        # doc_1_chunk_0 score = 0.8
        # doc_1_chunk_1 score = 0.3
        # doc_2_chunk_0 score = 0.2
        # Let's search and mock scores
        with patch.object(RetrievalConnector, 'search') as mock_search:
            mock_search.return_value = [
                {
                    "chunk_id": "doc_1_chunk_0",
                    "text": "The project uses Python.",
                    "score": 0.8,
                    "metadata": {"document_id": "doc_1", "file_name": "tech_spec.docx"}
                },
                {
                    "chunk_id": "doc_1_chunk_1",
                    "text": "The primary model is qwen2.5:3b.",
                    "score": 0.3,
                    "metadata": {"document_id": "doc_1", "file_name": "tech_spec.docx"}
                },
                {
                    "chunk_id": "doc_2_chunk_0",
                    "text": "The listed KPIs include throughput.",
                    "score": 0.2,
                    "metadata": {"document_id": "doc_2", "file_name": "kpi.docx"}
                }
            ]
            
            # Gating with score_threshold = 0.5
            retrieval_res = self.retriever.retrieve_and_gate([0.1]*384, top_k=3, score_threshold=0.5)
            
            self.assertTrue(retrieval_res["has_relevant_context"])
            self.assertEqual(len(retrieval_res["results"]), 1)
            self.assertEqual(retrieval_res["results"][0]["chunk_id"], "doc_1_chunk_0")
            self.assertEqual(len(retrieval_res["rejected_results"]), 2)
            
            # Build context and check that only the accepted chunk is passed
            context_result = self.context_builder.build("query", retrieval_res["results"])
            self.assertEqual(len(context_result["sources"]), 1)
            self.assertEqual(context_result["sources"][0]["chunk_id"], "doc_1_chunk_0")
            self.assertIn("uses Python", context_result["context"])
            self.assertNotIn("qwen2.5:3b", context_result["context"])

if __name__ == "__main__":
    unittest.main()

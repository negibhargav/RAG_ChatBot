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
import Vector_Store_Engine
import Context_Engine
import Generation_Engine
from Generation_Engine.providers.base import (
    BaseLLMProvider,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    LLMTimeoutError,
    LLMProviderError
)
from Generation_Engine.providers.ollama_provider import OllamaProvider

class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM Provider for unit testing without local dependencies.
    """
    def __init__(self, provider_name="mock_provider", should_fail=False, failure_type="connection"):
        self.provider_name = provider_name
        self.should_fail = should_fail
        self.failure_type = failure_type
        
        # Track parameters passed to generate
        self.last_prompt = None
        self.last_model = None
        self.last_temp = None
        self.last_max_tokens = None
        self.last_timeout = None

    def health_check(self, model=None):
        if self.should_fail:
            return {"available": False, "error": "Mocked health check failure"}
        return {"available": True, "model": model, "endpoint": "http://mock-url"}

    def generate(self, prompt, model, temperature=0.0, max_tokens=None, timeout=60):
        self.last_prompt = prompt
        self.last_model = model
        self.last_temp = temperature
        self.last_max_tokens = max_tokens
        self.last_timeout = timeout

        if self.should_fail:
            if self.failure_type == "connection":
                raise OllamaConnectionError("Mocked connection failure")
            elif self.failure_type == "model_not_found":
                raise OllamaModelNotFoundError("Mocked model not found")
            elif self.failure_type == "timeout":
                raise LLMTimeoutError("Mocked timeout")
            else:
                raise LLMProviderError("Mocked general error")

        return {
            "answer": f"Mock answer for prompt '{prompt[:20]}...'",
            "model": model,
            "provider": self.provider_name,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150
            },
            "finish_reason": "stop"
        }


class TestGenerationEngine(unittest.TestCase):
    def setUp(self):
        sys.stdout.reconfigure(encoding='utf-8')
        self.mock_provider = MockLLMProvider()
        self.engine = Generation_Engine.GenerationEngine(
            provider=self.mock_provider,
            model="qwen2.5:3b",
            temperature=0.0,
            max_tokens=256,
            timeout=30
        )

    def test_01_basic_generation(self):
        """1. Basic generation: Verify GenerationEngine receives prompt and returns an answer."""
        res = self.engine.generate("Hello world")
        self.assertIsNotNone(res["answer"])
        self.assertIn("Mock answer", res["answer"])
        self.assertEqual(res["model"], "qwen2.5:3b")
        self.assertEqual(res["provider"], "mock_provider")

    def test_02_empty_prompt(self):
        """2. Empty prompt: Verify empty prompts are rejected."""
        with self.assertRaises(ValueError):
            self.engine.generate("")
        with self.assertRaises(ValueError):
            self.engine.generate("   ")

    def test_03_provider_abstraction(self):
        """3. Provider abstraction: Verify GenerationEngine runs successfully with abstract provider."""
        another_mock = MockLLMProvider(provider_name="transformer_mock")
        custom_engine = Generation_Engine.GenerationEngine(provider=another_mock)
        res = custom_engine.generate("abstraction test")
        self.assertEqual(res["provider"], "transformer_mock")

    def test_04_provider_failures(self):
        """4. Provider failure: Verify custom exceptions are properly raised."""
        # Connection failure
        fail_conn = MockLLMProvider(should_fail=True, failure_type="connection")
        conn_engine = Generation_Engine.GenerationEngine(provider=fail_conn)
        with self.assertRaises(OllamaConnectionError):
            conn_engine.generate("test connection")

        # Model missing failure
        fail_model = MockLLMProvider(should_fail=True, failure_type="model_not_found")
        model_engine = Generation_Engine.GenerationEngine(provider=fail_model)
        with self.assertRaises(OllamaModelNotFoundError):
            model_engine.generate("test model")

        # Timeout failure
        fail_timeout = MockLLMProvider(should_fail=True, failure_type="timeout")
        time_engine = Generation_Engine.GenerationEngine(provider=fail_timeout)
        with self.assertRaises(LLMTimeoutError):
            time_engine.generate("test timeout")

    def test_05_no_context_behavior(self):
        """5. No-context behavior: Verify SLM is short-circuited when context is missing."""
        no_context_payload = {
            "query": "Who is Bhupender?",
            "context": "",
            "sources": [],
            "prompt": "SYSTEM: ... CONTEXT: None",
            "has_context": False
        }
        res = self.engine.generate(no_context_payload)
        
        self.assertEqual(res["finish_reason"], "no_context")
        self.assertIn("could not find this information", res["answer"])
        self.assertIsNone(res["usage"])
        # Verify provider generate was never invoked
        self.assertIsNone(self.mock_provider.last_prompt)

    def test_06_configuration(self):
        """6. Configuration: Verify parameters are mapped correctly during calls."""
        self.engine.generate("Config test")
        self.assertEqual(self.mock_provider.last_temp, 0.0)
        self.assertEqual(self.mock_provider.last_max_tokens, 256)
        self.assertEqual(self.mock_provider.last_timeout, 30)

    def test_07_response_normalization(self):
        """7. Response normalization: Verify standard completion schema output."""
        res = self.engine.generate("normalize test")
        keys = ["answer", "model", "provider", "usage", "finish_reason"]
        for k in keys:
            self.assertIn(k, res)

    def test_08_usage_handling(self):
        """8. Usage handling: Verify input, output, and total token count parsing."""
        res = self.engine.generate("tokens test")
        self.assertEqual(res["usage"]["input_tokens"], 100)
        self.assertEqual(res["usage"]["output_tokens"], 50)
        self.assertEqual(res["usage"]["total_tokens"], 150)

    def test_09_model_configuration(self):
        """9. Model configuration: Verify model tag swaps work dynamically."""
        self.engine.model = "llama3:8b"
        res = self.engine.generate("model swap test")
        self.assertEqual(res["model"], "llama3:8b")
        self.assertEqual(self.mock_provider.last_model, "llama3:8b")

    def test_10_local_endpoint_configuration(self):
        """10. Local endpoint configuration: Verify endpoint custom ports work."""
        custom_endpoint = "http://127.0.0.1:8000"
        prov = OllamaProvider(base_url=custom_endpoint)
        self.assertEqual(prov.base_url, custom_endpoint)


# ==================================================
# INTEGRATION TEST (Real local Ollama execution)
# ==================================================
class TestOllamaIntegration(unittest.TestCase):
    def test_real_ollama_completion(self):
        print("\n" + "="*40)
        print(" OLLAMA REAL INTEGRATION TEST")
        print("="*40)
        
        provider = OllamaProvider()
        model_name = "qwen2.5:3b"
        
        # Check if local Ollama is running and has the model downloaded
        status = provider.health_check(model_name)
        if not status["available"]:
            print(f"Skipping integration test: {status['error']}")
            print("="*40)
            return

        print("Local Ollama server is ONLINE. Running RAG Integration...")
        
        # 1. Extraction, Chunking, Embedding
        file_path = "test.txt"
        text = Extraction_Engine.extract_text(file_path)
        document = {
            "document_id": "integration_doc",
            "text": text,
            "metadata": {
                "source_path": os.path.abspath(file_path),
                "file_name": file_path,
                "file_type": ".txt"
            }
        }
        
        chunks = Chunking_Engine.split_document(document, chunk_size=300, chunk_overlap=30)
        
        embedder = Embedding_Engine.EmbeddingEngine()
        embedded_chunks = embedder.embed_chunks(chunks)
        
        # 2. Vector DB Storage & Querying
        db_dir = "Chroma_Integration_Test"
        if os.path.exists(db_dir):
            import shutil
            shutil.rmtree(db_dir)
            
        db = Vector_Store_Engine.ChromaVectorStore(db_dir=db_dir)
        db.save_chunks(embedded_chunks)
        
        retriever = Retrieval_Engine.RetrievalConnector(db)
        
        query = "Who is prashant?"
        query_vec = embedder.embed_query(query)
        retrieved_results = retriever.search(query_vec, top_k=3)
        
        # 3. Context Construction
        builder = Context_Engine.ContextBuilder()
        context_result = builder.build(query, retrieved_results)
        
        # 4. Local SLM Generation
        generator = Generation_Engine.GenerationEngine(
            provider=provider,
            model=model_name,
            temperature=0.0
        )
        
        print(f"\nQuery:\n{query}")
        print("\nRetrieved Sources:")
        for r in context_result["sources"]:
            print(f"  - Chunk: {r['chunk_id']} (Score: {r['score']:.4f}) in {r['file_name']}")
            
        print("\nSending Prompt to local model...")
        response = generator.generate(context_result)
        
        print("\nGenerated Answer:")
        print(response["answer"])
        print(f"\nModel: {response['model']}")
        print(f"Provider: {response['provider']}")
        print(f"Usage: {response['usage']}")
        print("="*40)
        
        # Clean up database files
        del db
        del retriever
        import shutil
        if os.path.exists(db_dir):
            try:
                shutil.rmtree(db_dir)
            except Exception:
                pass
                
        # Verification assertions
        self.assertIsNotNone(response["answer"])
        self.assertEqual(response["model"], model_name)
        self.assertEqual(response["provider"], "ollama")
        self.assertTrue(len(response["answer"].strip()) > 0)

if __name__ == "__main__":
    import os
    unittest.main()

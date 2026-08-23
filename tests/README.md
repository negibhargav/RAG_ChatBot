# Unit & Integration Test Suite

This directory contains automated unit and integration tests for all RAG system components.

---

## 📁 File Structure

- `test_vector_db.py`: Vector store creation, collection management, and embedding persistence tests.
- `test_retriever.py`: Similarity search, score calculation, and relevance gating tests.
- `test_context_builder.py`: Context prompt assembly, citation mapping, and token formatting tests.
- `test_generation.py`: Ollama API provider integration and prompt generation tests.
- `test_citation.py`: Citation engine claim attribution and source deduplication tests.
- `test_multi_chat.py`: Multi-chat isolation, chat collection routing, history persistence, and chat deletion tests.
- `test_pipeline.py`: End-to-end ingestion and query pipeline integration tests.
- `test_retrieval_evaluation.py`: Unit tests for retrieval evaluator logic.
- `test_chunking_evaluation.py`: Unit tests for chunking evaluator logic.
- `test_generation_evaluation.py`: Unit tests for answer accuracy evaluator logic.
- `test_grounding_evaluation.py`: Unit tests for grounding & hallucination evaluator logic.
- `test_citation_evaluation.py`: Unit tests for citation evaluator logic.
- `test_improvements.py`: Tests for security, error handling, and robust directory cleanup.

---

## 🧪 Running Tests

Run all unit tests with Python's built-in `unittest` runner:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

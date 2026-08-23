# Context Engine

The **Context Engine** formats retrieved document chunks and chat conversation history into structured prompts for LLM generation.

---

## 📁 File Structure

- `context_builder.py`: Implements `ContextBuilder` class.

---

## ⚙️ Output Package

Returns structured prompt dictionary:
```json
{
  "query": "What is the project about?",
  "context": "[Source 1]\nDocument: report.pdf\nChunk ID: doc_123_chunk_0\nScore: 0.88\n\nContent...",
  "sources": [
    { "citation_id": 1, "document_id": "doc_123", "file_name": "report.pdf" }
  ]
}
```

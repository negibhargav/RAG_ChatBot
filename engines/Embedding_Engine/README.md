# Embedding Engine

The **Embedding Engine** converts plain text chunks and user search queries into dense vector representations.

---

## 📁 File Structure

- `embedder.py`: Main `EmbeddingEngine` class wrapping PyTorch and `SentenceTransformer`.

---

## ⚙️ Model Details & Workflow

- **Default Model**: `BAAI/bge-small-en-v1.5`
- **Output Dimension**: 384 dimensions
- **Device Support**: Auto-selects CUDA GPU if available, else CPU.
- **Methods**:
  - `embed_chunks(chunks)`: Batch embeds list of chunk dictionaries and appends vector embeddings to `chunk["embedding"]`.
  - `embed_query(query_text)`: Generates embedding vector for search queries.

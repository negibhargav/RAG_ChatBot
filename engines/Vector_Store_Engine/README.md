# Vector Store Engine

The **Vector Store Engine** manages vector persistence, database collections, and similarity searches using **ChromaDB**.

---

## 📁 File Structure

- `database.py`: Contains `ChromaVectorStore` wrapper class interacting with `chromadb.PersistentClient`.

---

## ⚙️ Features & Multi-Chat Support

- **Persistence Path**: `Vector_DB/`
- **Dynamic Collections**: Supports dynamic per-chat collection creation (`chat_<chat_id>`).
- **Methods**:
  - `save_chunks(chunks)`: Upserts chunks, embeddings, and metadata into ChromaDB collection.
  - `query(query_embedding, top_k)`: Executes nearest-neighbor search returning top matching records with cosine distance scores.
  - `reset()` / `delete_collection(name)`: Purges vector store data.

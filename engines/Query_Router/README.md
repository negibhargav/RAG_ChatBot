# Query Router Engine

The **Query Router** classifies user queries before hitting vector storage to optimize response speed and avoid unnecessary RAG search.

---

## 📁 File Structure

- `router.py`: Implements `QueryRouter` class with intent classification rules.

---

## ⚙️ Classification Modes

- `CONVERSATIONAL`: Intercepts greetings (`hi`, `hello`), gratitude (`thanks`), and general chitchat. Bypasses vector DB search and returns conversational responses.
- `KNOWLEDGE`: Routes document/factual queries into vector retrieval pipeline.

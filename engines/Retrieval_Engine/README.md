# Retrieval Engine

The **Retrieval Engine** handles semantic candidate retrieval and relevance gating.

---

## 📁 File Structure

- `retriever.py`: Implements `RetrievalConnector` class.

---

## ⚙️ Relevance Gate & Workflow

1. Queries vector store for top-K candidates.
2. Converts distance metrics into similarity scores (`0.0` to `1.0`).
3. Compares scores against `score_threshold` (default `0.40`).
4. Separates candidates into `results` (accepted) and `rejected_results`.
5. Returns `has_relevant_context: False` if no chunks pass the threshold.

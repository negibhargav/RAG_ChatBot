# Citation Engine

The **Citation Engine** provides decoupled sentence-level claim attribution and citation marker (`[Source N]`) injection.

---

## 📁 File Structure

- `citation_engine.py`: Implements `CitationEngine` with source deduplication and sentence similarity matching.

---

## ⚙️ Workflow

1. Deduplicates retrieved source chunks at the document level and assigns sequential `citation_id` integers.
2. Splits generated raw answer into individual sentences.
3. Computes Jaccard token overlap between each sentence and source document content.
4. Appends `[Source N]` tags to supported sentences.
5. Returns final cited string and ordered source metadata list.

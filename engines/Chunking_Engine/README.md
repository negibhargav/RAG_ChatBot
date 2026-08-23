# Chunking Engine

The **Chunking Engine** splits extracted document text into small, semantically meaningful text units ("chunks") suitable for vector embedding and retrieval.

---

## 📁 File Structure

- `recursive_chunker.py`: Implements recursive character splitting with configurable chunk size and overlap parameters.

---

## ⚙️ Parameters & Workflow

- **Chunk Size (`chunk_size`)**: Default `500` characters. Controls the max character limit per chunk.
- **Chunk Overlap (`chunk_overlap`)**: Default `50` characters. Maintains semantic continuity between adjacent chunks.
- **Metadata Propagation**: Attaches metadata (`document_id`, `file_name`, `source_path`, `chunk_index`) to every chunk dict.

```python
# Output Chunk Schema
{
    "chunk_id": "doc_a1b2c3d4_chunk_0",
    "text": "Extracted paragraph content...",
    "metadata": {
        "document_id": "doc_a1b2c3d4",
        "file_name": "resume.pdf",
        "chunk_index": 0,
        "source_path": "/path/to/resume.pdf"
    }
}
```

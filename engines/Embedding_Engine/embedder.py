import torch
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

class EmbeddingEngine:
    """
    RAG Embedding Engine utilizing SentenceTransformers.
    Responsible for converting raw text chunks or queries into float32 numerical vector embeddings.
    """

    def __init__(self, model_name='BAAI/bge-small-en-v1.5', device=None):
        # 10. Handle model/device configuration (GPU if available, else CPU)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 3. Use one embedding model consistently
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[EmbeddingEngine] Loaded '{model_name}' on device '{self.device}'")

    def embed_chunks(self, chunks, normalize_embeddings=True):
        """
        Convert a list of Chunk dictionaries into a list of Chunk dictionaries with embeddings.
        
        1. Input: List of dicts:
           [{"chunk_id": str, "text": str, "metadata": dict}]
           
        5. Output: List of dicts preserving original chunk identity:
           [{"chunk_id": str, "text": str, "embedding": list[float], "metadata": dict}]
        """
        if not chunks:
            return []

        # 9. Handle empty input: Filter empty or whitespace-only chunks
        valid_chunks = []
        valid_texts = []
        
        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            # 12. Don't modify the original chunk text, just validate it is not empty
            if text and text.strip():
                valid_chunks.append(chunk)
                # 2. Only embed the chunk's text (metadata is excluded)
                valid_texts.append(text)
            else:
                print(f"[EmbeddingEngine] Warning: Skipping empty/whitespace chunk at index {idx}")

        if not valid_texts:
            return []

        # 6. Batch embedding (sends all texts in a single batch call)
        # 11. Make normalization configurable
        embeddings_numpy = self.model.encode(
            valid_texts,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False
        )

        # 13. Validate results (number of input chunks == number of embeddings)
        if len(embeddings_numpy) != len(valid_chunks):
            raise RuntimeError(
                f"Embedding validation failed: expected {len(valid_chunks)} vectors, "
                f"but got {len(embeddings_numpy)}"
            )

        # 13. Validate results (every vector must have the same dimension)
        dimensions = [len(v) for v in embeddings_numpy]
        if len(set(dimensions)) > 1:
            raise RuntimeError("Embedding validation failed: inconsistent vector dimensions found.")

        embedded_chunks = []
        # 7. Maintain ordering (chunk_0 -> embedding_0, chunk_1 -> embedding_1)
        for chunk, emb in zip(valid_chunks, embeddings_numpy):
            embedded_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                # 4. One text -> one vector (convert NumPy array to Python float list)
                "embedding": emb.tolist(),
                "metadata": dict(chunk["metadata"])  # Preserve metadata
            })

        return embedded_chunks

    def embed_query(self, query, normalize_embeddings=True):
        """
        14. Keep document embedding and query embedding separate conceptually.
        Embeds a single query string using the exact same model and configuration.
        """
        # 9. Handle empty input
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        # Generate embedding vector
        embedding = self.model.encode(
            query,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False
        )

        return embedding.tolist()

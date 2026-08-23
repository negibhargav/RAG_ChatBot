from abc import ABC, abstractmethod
import numpy as np

class BaseVectorStore(ABC):
    """
    Abstract Base Class for vector stores to decouple retrieval logic from storage.
    """
    @abstractmethod
    def get_all_chunks(self):
        """
        Retrieve all stored text chunks with their metadata and embeddings.
        Returns:
            list[dict]: List of dictionaries containing keys:
                        'chunk_id', 'text', 'embedding', 'metadata'
        """
        pass


class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory concrete vector store provider.
    """
    def __init__(self, embedded_chunks):
        self.chunks = embedded_chunks

    def get_all_chunks(self):
        return self.chunks


class RetrievalConnector:
    """
    Retrieval Connector responsible for evaluating similarities,
    ranking, thresholding, and selecting the top-K relevant chunks.
    """
    def __init__(self, vector_store):
        # Support initializing directly with list of chunks for convenience
        if isinstance(vector_store, list):
            self.vector_store = InMemoryVectorStore(vector_store)
        elif isinstance(vector_store, BaseVectorStore):
            self.vector_store = vector_store
        else:
            raise TypeError("vector_store must be a list of chunks or an instance of BaseVectorStore")

        self.dimension = None
        self._validate_store()

    def _validate_store(self):
        """
        Validate database elements prior to search.
        """
        try:
            chunks = self.vector_store.get_all_chunks()
        except NotImplementedError:
            chunks = []

        # For in-memory collections, we enforce the non-empty check at init
        if isinstance(self.vector_store, InMemoryVectorStore) and not chunks:
            raise ValueError("The vector store/embedded chunks list cannot be empty.")

        for idx, chunk in enumerate(chunks):
            # Check required keys exist
            for key in ["chunk_id", "text", "embedding", "metadata"]:
                if key not in chunk:
                    raise KeyError(f"Chunk at index {idx} is missing required key '{key}'")

            emb = chunk["embedding"]
            if not isinstance(emb, (list, np.ndarray)):
                raise TypeError(f"Embedding at chunk index {idx} must be a list or numpy array.")

            try:
                emb_arr = np.array(emb, dtype=np.float32)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Embedding at chunk index {idx} contains non-numeric values: {e}")

            if not np.isfinite(emb_arr).all():
                raise ValueError(f"Embedding at chunk index {idx} contains NaN or Infinity values.")

            dim = len(emb_arr)
            if dim == 0:
                raise ValueError(f"Embedding at chunk index {idx} is empty (0 dimensions).")

            if self.dimension is None:
                self.dimension = dim
            elif dim != self.dimension:
                raise ValueError(
                    f"Dimension mismatch at chunk index {idx}: "
                    f"expected {self.dimension}, but got {dim}."
                )

    def _validate_query(self, query_vector):
        """
        Validate user query vector dimensionality, finite bounds, and types.
        """
        if query_vector is None:
            raise ValueError("Query vector cannot be None.")

        if not isinstance(query_vector, (list, np.ndarray)):
            raise TypeError("Query vector must be a list or numpy array.")

        try:
            query_arr = np.array(query_vector, dtype=np.float32)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Query vector contains non-numeric values: {e}")

        if len(query_arr) == 0:
            raise ValueError("Query vector cannot be empty.")

        if not np.isfinite(query_arr).all():
            raise ValueError("Query vector contains NaN or Infinity values.")

        if self.dimension is None:
            raise ValueError("The vector store has not been initialized with any embeddings.")

        if len(query_arr) != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, "
                f"but got {len(query_arr)}."
            )

        return query_arr

    def search(self, query_vector, top_k=5, score_threshold=None):
        """
        Perform cosine similarity search across candidates.
        
        Args:
            query_vector (list | numpy.ndarray): Numerical vector representing query.
            top_k (int): Maximum number of results to return.
            score_threshold (float): Optional similarity score lower-bound.
            
        Returns:
            list[dict]: Decoupled matches containing chunk content, score, and inherited metadata.
        """
        # 1. Fetch dimension dynamically if it wasn't captured during init (e.g., store was empty)
        if self.dimension is None:
            if hasattr(self.vector_store, "get_dimension"):
                self.dimension = self.vector_store.get_dimension()
            else:
                try:
                    chunks = self.vector_store.get_all_chunks()
                except NotImplementedError:
                    chunks = []
                if chunks:
                    self.dimension = len(chunks[0]["embedding"])

        # 2. If it is still None, the store is empty
        if self.dimension is None:
            raise ValueError("The vector store is empty.")

        # 3. Validate query vector bounds
        query_arr = self._validate_query(query_vector)

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if score_threshold is not None:
            if not isinstance(score_threshold, (int, float)):
                raise ValueError("score_threshold must be a numeric value.")

        # 4. Check if the vector store has an optimized search implementation (like ChromaDB or FAISS)
        if hasattr(self.vector_store, "search"):
            return self.vector_store.search(query_arr, top_k=top_k, score_threshold=score_threshold)

        # 5. In-memory fallback
        chunks = self.vector_store.get_all_chunks()
        query_norm = np.linalg.norm(query_arr)
        if query_norm == 0:
            query_norm = 1e-10

        results = []
        for chunk in chunks:
            emb_arr = np.array(chunk["embedding"], dtype=np.float32)
            emb_norm = np.linalg.norm(emb_arr)
            if emb_norm == 0:
                emb_norm = 1e-10

            dot_product = np.dot(query_arr, emb_arr)
            # Calculate standard cosine similarity
            score = float(dot_product / (query_norm * emb_norm))

            # Apply similarity threshold
            if score_threshold is not None and score < score_threshold:
                continue

            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": score,
                "metadata": chunk["metadata"]
            })

        # Rank results in descending order of similarity
        results.sort(key=lambda x: x["score"], reverse=True)

        # Slice results to top-K
        return results[:top_k]

    def retrieve_and_gate(self, query_vector, top_k=5, score_threshold=None):
        """
        Retrieves top_k candidates and groups them into relevant or rejected based on score_threshold.
        
        Args:
            query_vector (list | numpy.ndarray): Numerical query representation.
            top_k (int): Maximum candidates to retrieve.
            score_threshold (float): Similarity threshold.
            
        Returns:
            dict: {
                "has_relevant_context": bool,
                "results": list[dict],
                "rejected_results": list[dict]
            }
        """
        # Call search without threshold to get all candidates, then partition them.
        candidates = self.search(query_vector, top_k=top_k, score_threshold=None)
        
        results = []
        rejected_results = []
        
        for candidate in candidates:
            score = candidate.get("score", 0.0)
            if score_threshold is not None:
                if score >= score_threshold:
                    results.append(candidate)
                else:
                    rejected_results.append(candidate)
            else:
                results.append(candidate)
                
        has_relevant_context = len(results) > 0
        
        return {
            "has_relevant_context": has_relevant_context,
            "results": results,
            "rejected_results": rejected_results
        }


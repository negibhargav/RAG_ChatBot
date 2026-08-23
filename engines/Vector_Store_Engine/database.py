import os
# pyrefly: ignore [missing-import]
import chromadb
from Retrieval_Engine import BaseVectorStore

class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB Vector Store implementation.
    Natively supports persisting raw text, metadata, and embeddings.
    """
    def __init__(self, db_dir="Vector_DB", collection_name="rag_bot_collection"):
        self.db_dir = os.path.abspath(db_dir)
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize persistent disk client
        self.client = chromadb.PersistentClient(path=self.db_dir)
        
        # Get or create collection with cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[ChromaVectorStore] Initialized persistent store at '{self.db_dir}'")

    def save_chunks(self, embedded_chunks):
        """
        Save embedded chunks to ChromaDB.
        """
        if not embedded_chunks:
            return

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in embedded_chunks:
            ids.append(chunk["chunk_id"])
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])
            # Copy metadata dictionary
            meta = dict(chunk["metadata"])
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[ChromaVectorStore] Saved {len(embedded_chunks)} chunks to collection 'rag_bot_collection'.")

    def get_dimension(self):
        """
        Returns the dimension of the embeddings in the database (or None if empty).
        """
        count = self.collection.count()
        if count == 0:
            return None
        # Retrieve one item to inspect embedding dimension
        peek = self.collection.peek(limit=1)
        if peek and peek["embeddings"]:
            return len(peek["embeddings"][0])
        return None

    def get_all_chunks(self):
        """
        Implements BaseVectorStore.get_all_chunks().
        Retrieves all chunks from the ChromaDB collection and formats them as standard chunks.
        """
        count = self.collection.count()
        if count == 0:
            return []

        # Retrieve all items with their embeddings, texts, and metadata
        data = self.collection.get(include=["embeddings", "documents", "metadatas"])
        
        chunks = []
        for idx in range(len(data["ids"])):
            chunks.append({
                "chunk_id": data["ids"][idx],
                "text": data["documents"][idx],
                "embedding": data["embeddings"][idx],
                "metadata": data["metadatas"][idx]
            })
        return chunks

    def search(self, query_vector, top_k=5, score_threshold=None):
        """
        Perform native ChromaDB query utilizing cosine similarity.
        """
        count = self.collection.count()
        if count == 0:
            raise ValueError("The vector store is empty.")

        # Ensure query_vector is a list of floats (Chroma client constraint)
        if hasattr(query_vector, "tolist"):
            query_list = query_vector.tolist()
        else:
            query_list = list(query_vector)

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        formatted_results = []
        for idx in range(len(ids)):
            # Distance returned is cosine distance: D = 1 - similarity
            # Convert back to standard similarity score
            score = 1.0 - float(distances[idx])

            if score_threshold is not None and score < score_threshold:
                continue

            formatted_results.append({
                "chunk_id": ids[idx],
                "text": documents[idx],
                "score": score,
                "metadata": metadatas[idx]
            })

        return formatted_results

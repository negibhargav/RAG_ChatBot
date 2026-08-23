import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../engines")))
import shutil
import Extraction_Engine
import Chunking_Engine
import Embedding_Engine
import Retrieval_Engine
import Vector_Store_Engine

def main():
    # Configure UTF-8 to prevent encoding crashes
    sys.stdout.reconfigure(encoding='utf-8')
    print("=== STARTING VECTOR DB TEST ===")

    db_dir = "Vector_DB_Test"
    # Clean up previous runs on startup (files are unlocked now)
    if os.path.exists(db_dir):
        print(f"Cleaning up old database directory at '{db_dir}'...")
        shutil.rmtree(db_dir)

    # 1. Pipeline: Extraction, Provenance, Chunking, Embedding
    print("\n[Step 1] Running RAG Pipeline over 'test.txt'...")
    file_path = "test.txt"
    text = Extraction_Engine.extract_text(file_path)
    
    document = {
        "document_id": "doc_db_test_id",
        "text": text,
        "metadata": {
            "source_path": os.path.abspath(file_path),
            "file_name": os.path.basename(file_path),
            "file_type": ".txt"
        }
    }
    
    chunks = Chunking_Engine.split_document(document, chunk_size=300, chunk_overlap=30)
    print(f"  Generated {len(chunks)} chunks.")
    
    embedder = Embedding_Engine.EmbeddingEngine(model_name='BAAI/bge-small-en-v1.5')
    embedded_chunks = embedder.embed_chunks(chunks)
    print(f"  Generated {len(embedded_chunks)} embeddings.")

    # 2. Vector DB: Save chunks to ChromaDB
    print(f"\n[Step 2] Initializing persistent ChromaDB at '{db_dir}'...")
    db = Vector_Store_Engine.ChromaVectorStore(db_dir=db_dir)
    db.save_chunks(embedded_chunks)
    
    # 3. Retrieval: Search using RetrievalConnector
    print("\n[Step 3] Querying ChromaDB via RetrievalConnector...")
    retriever = Retrieval_Engine.RetrievalConnector(db)
    
    query_text = "Who is Bhupender Singh Negi?"
    query_vec = embedder.embed_query(query_text)
    
    results = retriever.search(query_vec, top_k=3)
    print(f"  Query: '{query_text}'")
    for idx, res in enumerate(results):
        print(f"    Rank {idx+1}: Chunk ID: {res['chunk_id']} (Score: {res['score']:.4f})")
        print(f"      Content Preview: {res['text'][:100].strip()}...")

    assert any("BHUPENDER" in res["text"].upper() for res in results), "Error: Retrieval did not match name!"
    print("  [Success] Retrieval matched semantic search criteria.")

    # 4. Persistence Check: Reload from disk
    print("\n[Step 4] Simulating program restart and reloading from disk...")
    # Delete old Python references
    del db
    del retriever
    
    # Reinitialize a new ChromaVectorStore on the same directory
    reloaded_db = Vector_Store_Engine.ChromaVectorStore(db_dir=db_dir)
    reloaded_retriever = Retrieval_Engine.RetrievalConnector(reloaded_db)
    
    reloaded_count = reloaded_db.collection.count()
    print(f"  Reloaded database. Contains {reloaded_count} chunks.")
    assert reloaded_count == len(embedded_chunks), f"Error: Count mismatch! Expected {len(embedded_chunks)}, got {reloaded_count}"
    
    # Run a test query on reloaded database
    reloaded_results = reloaded_retriever.search(query_vec, top_k=1)
    print(f"  Query on reloaded database returned: Chunk ID: {reloaded_results[0]['chunk_id']} (Score: {reloaded_results[0]['score']:.4f})")
    
    print("\n=== VECTOR DB TEST COMPLETED SUCCESSFULY ===")

if __name__ == "__main__":
    main()

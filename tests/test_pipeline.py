import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../engines")))
import hashlib
import pprint
import Extraction_Engine
import Chunking_Engine
import Embedding_Engine

def main():
    # Set console encoding to UTF-8 to prevent Windows cp1252 encoding crashes
    sys.stdout.reconfigure(encoding='utf-8')
    print("=== STARTING PIPELINE TEST ===")
    
    file_path = "test.txt"
    file_name = os.path.basename(file_path)
    
    # 0. Extract and Normalize Text
    print("\n[Step 0] Extracting and Normalizing Text...")
    text = Extraction_Engine.extract_text(file_path)
    print(f"Extracted raw text length: {len(text)} characters.")
    
    # 1. Step 1: Document Metadata/Provenance Assignment
    print("\n[Step 1] Document Metadata/Provenance Assignment:")
    hasher = hashlib.md5(file_name.encode('utf-8'))
    doc_id = f"doc_{hasher.hexdigest()[:8]}"
    
    document = {
        "document_id": doc_id,
        "text": text,
        "metadata": {
            "source_path": os.path.abspath(file_path),
            "file_name": file_name,
            "file_type": ".txt"
        }
    }
    pprint.pprint(document, depth=2)
    
    # 2. Step 2: Text Chunking
    print("\n[Step 2] Text Chunking (chunk_size=300, chunk_overlap=30):")
    chunks = Chunking_Engine.split_document(document, chunk_size=300, chunk_overlap=30)
    print(f"Generated {len(chunks)} chunks.")
    
    # 3. Step 3: Generating Embeddings
    print("\n[Step 3] Initializing Embedding Engine & Generating Chunks Embeddings...")
    embedder = Embedding_Engine.EmbeddingEngine(model_name='BAAI/bge-small-en-v1.5')
    
    # Generate batch embeddings for chunks (defaults to normalize_embeddings=True)
    embedded_chunks = embedder.embed_chunks(chunks)
    
    print("\n=== Validating Chunk Embeddings ===")
    # Validation 1: Check number of output chunks matches input count
    print(f"Number of input chunks: {len(chunks)}")
    print(f"Number of generated embeddings: {len(embedded_chunks)}")
    assert len(chunks) == len(embedded_chunks), "Error: Chunk count mismatch!"
    print("[Success] Output count matches input count.")
    
    # Validation 2: Verify all vectors have the same dimension
    if len(embedded_chunks) > 0:
        first_emb_len = len(embedded_chunks[0]["embedding"])
        print(f"Vector dimensions of first chunk: {first_emb_len}")
        for idx, ec in enumerate(embedded_chunks):
            current_dim = len(ec["embedding"])
            assert current_dim == first_emb_len, f"Error: Dimension mismatch at index {idx}!"
        print(f"[Success] All {len(embedded_chunks)} vectors have identical dimensions of {first_emb_len}.")
    
    # Preview a couple of embedded chunks
    print("\n=== Preview of Embedded Chunks ===")
    preview_limit = min(2, len(embedded_chunks))
    for idx in range(preview_limit):
        ec = embedded_chunks[idx]
        print(f"\n- Chunk ID: {ec['chunk_id']}")
        print(f"  Metadata: {ec['metadata']}")
        print(f"  Content: \"{ec['text'][:80]}...\"")
        print(f"  Vector Shape: {len(ec['embedding'])} dimensions")
        print(f"  Vector Preview (First 5 values): {ec['embedding'][:5]}")
        print("-" * 50)
        
    # 4. Step 4: Query Embedding
    print("\n=== Test Query Embedding ===")
    test_query = "Who is Bhupender Singh Negi?"
    print(f"Generating embedding for query: '{test_query}'")
    query_vector = embedder.embed_query(test_query)
    
    print(f"  Query Vector Shape: {len(query_vector)} dimensions")
    print(f"  Query Vector Preview (First 5 values): {query_vector[:5]}")
    
    # Validate query dimensions match chunk dimensions
    if len(embedded_chunks) > 0:
        assert len(query_vector) == first_emb_len, "Error: Query dimension mismatch!"
        print("[Success] Query vector dimension matches chunk vector dimension.")
        
    print("\n=== PIPELINE TEST COMPLETED SUCCESSFULY ===")

if __name__ == "__main__":
    main()

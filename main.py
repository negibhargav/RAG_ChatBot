import os
import sys
import hashlib

# Ensure imports work from engines directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "engines")))

import upload_files
import Extraction_Engine
import Chunking_Engine
import Embedding_Engine
import Vector_Store_Engine
import Retrieval_Engine
import Context_Engine
import Generation_Engine
from Generation_Engine.providers.ollama_provider import OllamaProvider
import Citation_Engine
from Query_Router import QueryRouter

# Configuration parameters (Fully configurable)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
VECTOR_DB_DIR = "Vector_DB"
OLLAMA_MODEL = "qwen2.5:3b"
SCORE_THRESHOLD = 0.40  # Configurable relevance threshold

def generate_document_id(file_name):
    """Generates a deterministic 8-character document ID based on the file name."""
    hasher = hashlib.md5(file_name.encode('utf-8'))
    return f"doc_{hasher.hexdigest()[:8]}"

def main():
    # Set console encoding to UTF-8 to prevent Windows cp1252 encoding crashes
    sys.stdout.reconfigure(encoding='utf-8')
    print("="*60)
    print("                   LOCAL RAG BOT APPLICATION")
    print("="*60)
    
    # Reset Vector DB and Uploaded Document folders on startup
    print("\nResetting Vector Database and Uploaded Document folders...")
    for path in [VECTOR_DB_DIR, upload_files.UPLOAD_DIR]:
        if os.path.exists(path):
            try:
                import shutil
                shutil.rmtree(path)
                print(f"  [Reset] Cleaned and removed: '{path}/'")
            except Exception as e:
                print(f"  [Warning] Could not reset '{path}': {e}")
                
    # 1. Initialize DB and Embedder
    print(f"Initializing Vector Database at '{VECTOR_DB_DIR}'...")
    db = Vector_Store_Engine.ChromaVectorStore(db_dir=VECTOR_DB_DIR)
    
    print("Initializing Embedding Engine (model=BAAI/bge-small-en-v1.5)...")
    embedder = Embedding_Engine.EmbeddingEngine()
    
    # Ask if user wants to upload new files
    user_choice = input("\nDo you want to upload new documents? (y/n) [default: y]: ").strip().lower()
    
    if user_choice not in ['n', 'no']:
        # Run the uploader (opens file selection dialog)
        uploaded_files = upload_files.upload_documents()
        
        if uploaded_files:
            print("\n=== Ingesting Uploaded Files ===")
            
            for file in uploaded_files:
                print(f"\nProcessing file: '{file}'")
                
                # Construct paths and extensions
                full_path = os.path.join(upload_files.UPLOAD_DIR, file)
                file_name = os.path.basename(file)
                _, ext = os.path.splitext(file)
                ext = ext.lower()
                
                # Extraction Stage
                extracted_text = None
                try:
                    if ext == '.txt':
                        extracted_text = Extraction_Engine.extract_text(full_path)
                    elif ext == '.docx':
                        extracted_text = Extraction_Engine.extract_docx(full_path)
                    elif ext == '.pdf':
                        extracted_text = Extraction_Engine.extract_pdf(full_path)
                    elif ext in {'.png', '.jpg', '.jpeg'}:
                        extracted_text = Extraction_Engine.extract_image(full_path)
                    else:
                        print(f"  [Info] Format '{ext}' is unsupported.")
                        continue
                except Exception as e:
                    print(f"  [Error] Text extraction failed: {e}")
                    continue
                    
                if not extracted_text:
                    print("  [Warning] Extracted text is empty. Skipping.")
                    continue

                # Step 1: Document Metadata/Provenance Assignment
                doc_id = generate_document_id(file_name)
                document = {
                    "document_id": doc_id,
                    "text": extracted_text,
                    "metadata": {
                        "source_path": full_path,
                        "file_name": file_name,
                        "file_type": ext
                    }
                }
                
                print(f"  [Step 1: Provenance Assigned] Document ID: {document['document_id']}")

                # Step 2: Text Chunking
                chunks = Chunking_Engine.split_document(
                    document, 
                    chunk_size=CHUNK_SIZE, 
                    chunk_overlap=CHUNK_OVERLAP
                )
                print(f"  [Step 2: Splitting Text] Generated {len(chunks)} chunks.")

                # Step 3: Generate Vector Embeddings
                print("  [Step 3: Generating Numerical Embeddings]...")
                embedded_chunks = embedder.embed_chunks(chunks)
                
                # Save to database
                print(f"  [Step 4: Storing in Vector DB] Saving to database collection...")
                db.save_chunks(embedded_chunks)
                print("  [Ingestion Successful]")
        else:
            print("\nNo new files were selected for upload.")
    
    # 2. Check current collection count
    collection_count = db.collection.count()
    print(f"\nVector database currently contains {collection_count} total text chunks.")
    
    has_database_data = True
    if collection_count == 0:
        print("\n[Error] The database is completely empty.")
        retry_choice = input("Would you like to upload documents now? (y/n) [default: y]: ").strip().lower()
        if retry_choice not in ['n', 'no']:
            uploaded_files = upload_files.upload_documents()
            if uploaded_files:
                print("\n=== Ingesting Uploaded Files ===")
                for file in uploaded_files:
                    print(f"\nProcessing file: '{file}'")
                    full_path = os.path.join(upload_files.UPLOAD_DIR, file)
                    file_name = os.path.basename(file)
                    _, ext = os.path.splitext(file)
                    ext = ext.lower()
                    
                    extracted_text = None
                    try:
                        if ext == '.txt':
                            extracted_text = Extraction_Engine.extract_text(full_path)
                        elif ext == '.docx':
                            extracted_text = Extraction_Engine.extract_docx(full_path)
                        elif ext == '.pdf':
                            extracted_text = Extraction_Engine.extract_pdf(full_path)
                        elif ext in {'.png', '.jpg', '.jpeg'}:
                            extracted_text = Extraction_Engine.extract_image(full_path)
                    except Exception as e:
                        print(f"  [Error] Text extraction failed: {e}")
                        continue
                        
                    if not extracted_text:
                        print("  [Warning] Extracted text is empty. Skipping.")
                        continue

                    doc_id = generate_document_id(file_name)
                    document = {
                        "document_id": doc_id,
                        "text": extracted_text,
                        "metadata": {
                            "source_path": full_path,
                            "file_name": file_name,
                            "file_type": ext
                        }
                    }
                    
                    chunks = Chunking_Engine.split_document(
                        document, 
                        chunk_size=CHUNK_SIZE, 
                        chunk_overlap=CHUNK_OVERLAP
                    )
                    print(f"  [Step 2: Splitting Text] Generated {len(chunks)} chunks.")

                    print("  [Step 3: Generating Numerical Embeddings]...")
                    embedded_chunks = embedder.embed_chunks(chunks)
                    
                    print(f"  [Step 4: Storing in Vector DB] Saving to database collection...")
                    db.save_chunks(embedded_chunks)
                    print("  [Ingestion Successful]")
                
                # Check collection count again
                collection_count = db.collection.count()
                print(f"\nVector database now contains {collection_count} total text chunks.")
        
        if collection_count == 0:
            print("\nContinuing in standard chatbot mode (queries will be answered directly by Qwen).")
            has_database_data = False
        
    # 3. Initialize RAG query engines
    print("\nInitializing RAG retrieval and generation layers...")
    retriever = Retrieval_Engine.RetrievalConnector(db)
    context_builder = Context_Engine.ContextBuilder()
    
    provider = OllamaProvider()
    generator = Generation_Engine.GenerationEngine(
        provider=provider,
        model=OLLAMA_MODEL,
        temperature=0.0
    )
    citation_engine = Citation_Engine.CitationEngine()
    
    # Check Ollama server connectivity
    status = provider.health_check(OLLAMA_MODEL)
    if not status["available"]:
        print(f"\n[Warning] Local Ollama is offline or model is missing: {status['error']}")
        print("Note: Prompts will return mock/grounded fallback responses until Ollama is running.")
    else:
        print(f"Local Ollama online: '{OLLAMA_MODEL}' is active and ready.")
        
    print("\n" + "="*50)
    print(" RAG BOT PROMPT LOOP - Type 'exit' or 'quit' to close")
    print("="*50)
    
    # 4. Interactive loop
    router = QueryRouter()
    verbose_logging = True  # Diagnostic logging (can be disabled later)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']:
                print("Closing RAG Bot. Goodbye!")
                break
                
            # Step 0: Route the query
            route_res = router.route(query)
            query_type = route_res["query_type"]
            
            if verbose_logging:
                print(f"\n[Query Router]\nQuery type: {query_type}")
                
            if query_type == "CONVERSATIONAL":
                lower_query = query.lower()
                if any(w in lower_query for w in ["hi", "hello", "hey", "hola", "greetings"]):
                    answer = "Hello! How can I help you today?"
                elif any(w in lower_query for w in ["thanks", "thank you", "thx"]):
                    answer = "You're welcome! Let me know if you need anything else."
                elif any(w in lower_query for w in ["how are you", "how's it going"]):
                    answer = "I'm doing well, thank you for asking! How can I help you?"
                elif any(w in lower_query for w in ["who are you", "what is your name"]):
                    answer = "I am a local document-grounded AI assistant."
                else:
                    answer = "Hello! Let me know if you have any questions about the documents."
                
                print("\nAnswer:")
                print(answer)
                print("\nSources:")
                print("  [No supporting context found in uploaded documents]")
                print("-"*50)
                continue
                
            # If the database is empty, bypass retrieval and call the generator directly with the raw query
            if not has_database_data:
                if verbose_logging:
                    print("\n[Retrieval]\nDatabase is empty. Skipping retrieval.")
                    print("\n[Generation]\nCalling Ollama qwen2.5:3b directly")
                
                print("Thinking...")
                response = generator.generate(query)
                
                print("\nAnswer:")
                print(response["answer"])
                print("\nSources:")
                print("  [No supporting context found in uploaded documents]")
                print("-"*50)
                continue
                
            # Step 1: Embed Query
            query_vec = embedder.embed_query(query)
            
            # Step 2: Retrieve and Relevance Gate
            retrieval_res = retriever.retrieve_and_gate(query_vec, top_k=3, score_threshold=SCORE_THRESHOLD)
            
            if verbose_logging:
                candidates_count = len(retrieval_res["results"]) + len(retrieval_res["rejected_results"])
                print(f"\n[Retrieval]\nCandidates retrieved: {candidates_count}")
                print("\n[Relevance Gate]")
                all_candidates = sorted(
                    retrieval_res["results"] + retrieval_res["rejected_results"],
                    key=lambda x: x["score"],
                    reverse=True
                )
                for idx, c in enumerate(all_candidates):
                    status = "ACCEPT" if c in retrieval_res["results"] else "REJECT"
                    print(f"Chunk {idx+1}: score={c['score']:.4f} → {status}")
                print(f"\n[Context]\nAccepted chunks: {len(retrieval_res['results'])}")
                
            if not retrieval_res["has_relevant_context"]:
                print("\nAnswer:")
                print("I couldn't find relevant information about that in the uploaded documents.")
                print("\nSources:")
                print("  [No supporting context found in uploaded documents]")
                print("-"*50)
                continue
                
            # Step 3: Construct Grounded Context
            context_result = context_builder.build(query, retrieval_res["results"])
            
            # Step 4: Generate SLM answer
            print("Thinking...")
            if verbose_logging:
                print(f"\n[Generation]\nCalling Ollama {OLLAMA_MODEL}")
            response = generator.generate(context_result)
            
            # Step 5: Inject Citations
            cited_result = citation_engine.generate_citations(response["answer"], context_result)
            
            # Step 6: Render cited answer
            print("\nAnswer:")
            print(cited_result["answer"])
            
            if verbose_logging:
                print(f"\n[Citation]\nUnique sources: {len(cited_result['sources'])}")
                
            print("\nSources:")
            if cited_result["sources"]:
                for s in cited_result["sources"]:
                    print(f"  [{s['citation_id']}] {s['file_name']}")
            else:
                print("  [No supporting context found in uploaded documents]")
            print("-"*50)
            
        except KeyboardInterrupt:
            print("\nClosing RAG Bot. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error] Failed to process query: {e}")

if __name__ == "__main__":
    main()


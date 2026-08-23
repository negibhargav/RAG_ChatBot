import os
import sys
import hashlib
import shutil
import uuid
import json
import gc
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from flask import Flask, request, jsonify, render_template

def secure_rmtree(path):
    """Safely removes a directory on Windows by forcing garbage collection to release file handles and retrying."""
    gc.collect()
    for attempt in range(5):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return True
        except PermissionError:
            time.sleep(0.2)
            gc.collect()
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
        return True
    except Exception as e:
        print(f"[Warning] secure_rmtree failed for {path}: {e}")
        return False

def prune_orphaned_vector_segments():
    """Scans Vector_DB and removes subdirectories that no longer correspond to active database segments."""
    db_path = os.path.join("Vector_DB", "chroma.sqlite3")
    vector_db_dir = "Vector_DB"
    if not os.path.exists(db_path) or not os.path.exists(vector_db_dir):
        return
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM segments")
        active_ids = set(row[0] for row in cursor.fetchall())
        conn.close()

        for entry in os.listdir(vector_db_dir):
            full_path = os.path.join(vector_db_dir, entry)
            if os.path.isdir(full_path) and entry != "chat_metadata" and entry not in active_ids:
                secure_rmtree(full_path)
    except Exception as e:
        print(f"[Warning] Prune segments failed: {e}")

# Set console encoding to UTF-8 to prevent Windows cp1252 encoding crashes
sys.stdout.reconfigure(encoding='utf-8')

# Ensure imports work from RAG_bot parent dir and engines directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
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

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
VECTOR_DB_DIR = "Vector_DB"
OLLAMA_MODEL = "qwen2.5:3b"
SCORE_THRESHOLD = 0.40

app = Flask(__name__)

# ── Logging Setup ──────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
_log_handler = RotatingFileHandler(
    "logs/app.log",
    maxBytes=5 * 1024 * 1024,  # 5 MB per file
    backupCount=3,
    encoding="utf-8"
)
_log_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logger = logging.getLogger("rag_bot")
logger.setLevel(logging.INFO)
logger.addHandler(_log_handler)
# Also mirror to console so existing terminal output is unchanged
logger.addHandler(logging.StreamHandler(sys.stdout))

# Initialize Global Engines
print("[Web Backend] Initializing Vector Database...")
db = Vector_Store_Engine.ChromaVectorStore(db_dir=VECTOR_DB_DIR)

print("[Web Backend] Loading Embedding Model BAAI/bge-small-en-v1.5...")
embedder = Embedding_Engine.EmbeddingEngine()

print("[Web Backend] Initializing RAG Pipeline Layer Connectors...")
retriever = Retrieval_Engine.RetrievalConnector(db)
context_builder = Context_Engine.ContextBuilder()
provider = OllamaProvider()
generator = Generation_Engine.GenerationEngine(provider=provider, model=OLLAMA_MODEL, temperature=0.0)
citation_engine = Citation_Engine.CitationEngine()
router = QueryRouter()

logger.info("Server started. Model=%s | VectorDB=%s", OLLAMA_MODEL, VECTOR_DB_DIR)

# Local storage configuration for chat metadata and histories
METADATA_DIR = os.path.join(VECTOR_DB_DIR, "chat_metadata")
os.makedirs(METADATA_DIR, exist_ok=True)

def get_chat_metadata_path(chat_id):
    return os.path.join(METADATA_DIR, chat_id, "metadata.json")

def get_chat_history_path(chat_id):
    return os.path.join(METADATA_DIR, chat_id, "history.json")

def load_chat_history(chat_id):
    path = get_chat_history_path(chat_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_history(chat_id, history):
    path = get_chat_history_path(chat_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def list_all_chats():
    if not os.path.exists(METADATA_DIR):
        return []
    chats = []
    for chat_id in os.listdir(METADATA_DIR):
        meta_path = get_chat_metadata_path(chat_id)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    chats.append(json.load(f))
            except Exception:
                pass
    chats.sort(key=lambda x: x.get("created_at", ""))
    return chats

def rename_chat(chat_id, first_query):
    meta_path = get_chat_metadata_path(chat_id)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            clean_name = first_query.strip().split("\n")[0]
            if len(clean_name) > 30:
                clean_name = clean_name[:27] + "..."
            
            meta["name"] = clean_name
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

def generate_document_id(file_name):
    """Generates a deterministic 8-character document ID based on the file name."""
    hasher = hashlib.md5(file_name.encode('utf-8'))
    return f"doc_{hasher.hexdigest()[:8]}"

@app.route('/')
def index():
    """Serves the primary user interface."""
    return render_template('index.html')

@app.route('/chats', methods=['GET'])
def get_chats():
    """Returns a list of all active chats."""
    return jsonify({"chats": list_all_chats()})

@app.route('/chats', methods=['POST'])
def create_chat():
    """Creates a new chat session."""
    chat_id = str(uuid.uuid4())
    os.makedirs(os.path.join(METADATA_DIR, chat_id), exist_ok=True)
    
    meta = {
        "chat_id": chat_id,
        "name": "New Chat",
        "created_at": datetime.now().isoformat()
    }
    with open(get_chat_metadata_path(chat_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        
    save_chat_history(chat_id, [])
    logger.info("Chat created | chat_id=%s", chat_id)
    return jsonify({"success": True, "chat": meta})

@app.route('/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Deletes a chat, its history, uploaded files, and ChromaDB collection."""
    try:
        col_name = f"chat_{chat_id}"
        db.client.delete_collection(col_name)
    except Exception:
        pass
        
    chat_upload_dir = os.path.join(upload_files.UPLOAD_DIR, chat_id)
    secure_rmtree(chat_upload_dir)
            
    chat_meta_dir = os.path.join(METADATA_DIR, chat_id)
    secure_rmtree(chat_meta_dir)
    
    prune_orphaned_vector_segments()
    
    logger.info("Chat deleted | chat_id=%s", chat_id)
    return jsonify({"success": True, "message": f"Chat {chat_id} deleted."})

@app.route('/chats/<chat_id>/history', methods=['GET'])
def get_chat_history_route(chat_id):
    """Returns the conversation history for a chat."""
    return jsonify({"history": load_chat_history(chat_id)})


@app.route('/documents', methods=['GET'])
def get_documents():
    """Returns a list of all uploaded documents currently in the upload directory for a chat."""
    chat_id = request.args.get("chat_id", "default")
    uploaded_files_list = []
    chat_upload_dir = os.path.join(upload_files.UPLOAD_DIR, chat_id)
    
    if os.path.exists(chat_upload_dir):
        for root, dirs, files in os.walk(chat_upload_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), chat_upload_dir)
                rel_path = rel_path.replace("\\", "/")
                size_bytes = os.path.getsize(os.path.join(root, file))
                size_kb = round(size_bytes / 1024, 2)
                uploaded_files_list.append({
                    "name": file,
                    "rel_path": rel_path,
                    "size_kb": size_kb
                })
    return jsonify({"documents": uploaded_files_list})

@app.route('/upload', methods=['POST'])
def upload():
    """Handles document upload and ingests them into the RAG system for a specific chat."""
    if 'files' not in request.files:
        return jsonify({"error": "No files found in upload request."}), 400
        
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected."}), 400

    chat_id = request.form.get("chat_id", "default")
    
    # Ensure chat workspace directory registered
    os.makedirs(os.path.join(METADATA_DIR, chat_id), exist_ok=True)
    if not os.path.exists(get_chat_metadata_path(chat_id)):
        meta = {
            "chat_id": chat_id,
            "name": "Default Chat" if chat_id == "default" else "New Chat",
            "created_at": datetime.now().isoformat()
        }
        with open(get_chat_metadata_path(chat_id), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # Ensure Chat-specific Upload Dir exists
    chat_upload_dir = os.path.join(upload_files.UPLOAD_DIR, chat_id)
    os.makedirs(chat_upload_dir, exist_ok=True)
    
    saved_files = []
    ingested_count = 0
    errors = []
    
    for file in files:
        filename = file.filename
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        if ext not in upload_files.ALLOWED_EXTENSIONS:
            errors.append(f"Skipped: '{filename}' - Unsupported format ({ext}).")
            continue
            
        subdir = upload_files.EXTENSION_TO_SUBDIR.get(ext, '')
        target_dir = os.path.join(chat_upload_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)
        
        destination_path = os.path.join(target_dir, filename)
        file.save(destination_path)
        rel_path = f"{subdir}/{filename}"
        saved_files.append(rel_path)
        
        # Ingest file immediately
        try:
            extracted_text = None
            if ext == '.txt':
                extracted_text = Extraction_Engine.extract_text(destination_path)
            elif ext == '.docx':
                extracted_text = Extraction_Engine.extract_docx(destination_path)
            elif ext == '.pdf':
                extracted_text = Extraction_Engine.extract_pdf(destination_path)
            elif ext in {'.png', '.jpg', '.jpeg'}:
                extracted_text = Extraction_Engine.extract_image(destination_path)
                
            if not extracted_text:
                errors.append(f"Failed '{filename}': Text content could not be extracted.")
                continue

            doc_id = generate_document_id(filename)
            document = {
                "document_id": doc_id,
                "text": extracted_text,
                "metadata": {
                    "source_path": destination_path,
                    "file_name": filename,
                    "file_type": ext
                }
            }
            
            # Chunking
            chunks = Chunking_Engine.split_document(
                document, 
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP
            )
            
            # Embeddings
            embedded_chunks = embedder.embed_chunks(chunks)
            
            # Save to Chroma DB inside chat collection
            chat_db = Vector_Store_Engine.ChromaVectorStore(db_dir=VECTOR_DB_DIR, collection_name=f"chat_{chat_id}")
            chat_db.save_chunks(embedded_chunks)
            ingested_count += 1
            
        except Exception as e:
            errors.append(f"Error processing '{filename}': {str(e)}")
            logger.error("Ingest error | file=%s | chat_id=%s | error=%s", filename, chat_id, str(e))

    if ingested_count:
        logger.info("Upload complete | chat_id=%s | files=%s | ingested=%d",
                    chat_id,
                    [os.path.basename(f) for f in saved_files],
                    ingested_count)
    if errors:
        logger.warning("Upload errors | chat_id=%s | errors=%s", chat_id, errors)
            
    # Re-validate DB dimension in retriever
    try:
        chat_retriever = Retrieval_Engine.RetrievalConnector(
            Vector_Store_Engine.ChromaVectorStore(db_dir=VECTOR_DB_DIR, collection_name=f"chat_{chat_id}")
        )
        chat_retriever._validate_store()
    except Exception:
        pass
        
    return jsonify({
        "success": True,
        "files_uploaded": saved_files,
        "ingested_count": ingested_count,
        "errors": errors
    })

@app.route('/query', methods=['POST'])
def query():
    """Processes user query and handles conversational routing, gating, and citations."""
    data = request.json or {}
    query_text = data.get("query", "").strip()
    score_threshold = float(data.get("score_threshold", SCORE_THRESHOLD))
    top_k = int(data.get("top_k", 3))
    chat_id = data.get("chat_id", "default")
    
    if not query_text:
        return jsonify({"error": "Query cannot be empty."}), 400

    logger.info("Query received | chat_id=%s | query='%s'", chat_id, query_text[:120])
        
    # Load conversation history for this chat
    history = load_chat_history(chat_id)
    
    # Step 0: Query Router
    route_res = router.route(query_text)
    query_type = route_res["query_type"]
    
    # 1. Conversational Query Routing
    if query_type == "CONVERSATIONAL":
        lower_query = query_text.lower()
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
            
        # Update history
        history.append({"role": "user", "content": query_text})
        history.append({
            "role": "assistant",
            "content": answer,
            "sources": []
        })
        save_chat_history(chat_id, history)
        logger.info("Conversational reply | chat_id=%s | route=CONVERSATIONAL", chat_id)
        return jsonify({
            "query_type": "CONVERSATIONAL",
            "has_relevant_context": False,
            "answer": answer,
            "sources": []
        })
        
    # Get chat vector store & check if DB has data
    chat_db = Vector_Store_Engine.ChromaVectorStore(db_dir=VECTOR_DB_DIR, collection_name=f"chat_{chat_id}")
    collection_count = chat_db.collection.count()
    
    # 2. No database data - Direct chatbot mode
    if collection_count == 0:
        print(f"[Query Pipeline] Database empty for chat {chat_id}. Direct chatbot invocation.")
        try:
            # Build conversation history block
            history_block = ""
            if history:
                history_lines = []
                for msg in history[-10:]:
                    role_name = "User" if msg["role"] == "user" else "Assistant"
                    history_lines.append(f"{role_name}: {msg['content']}")
                history_block = "CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"
                
            prompt = (
                f"SYSTEM:\nYou are a helpful conversational AI assistant.\n\n"
                f"{history_block}"
                f"USER QUESTION:\n{query_text}"
            )
            response = generator.generate(prompt)
            answer = response["answer"]
            
            history.append({"role": "user", "content": query_text})
            history.append({
                "role": "assistant",
                "content": answer,
                "sources": [],
                "diagnostic_logs": {
                    "router": "KNOWLEDGE",
                    "retrieval": "Database is empty. Skipped retrieval.",
                    "gating": "No chunks evaluated.",
                    "generation": "Called Ollama directly."
                }
            })      
            save_chat_history(chat_id, history)
            
            if len(history) <= 2:
                rename_chat(chat_id, query_text)
                
            return jsonify({
                "query_type": "KNOWLEDGE",
                "has_relevant_context": False,
                "answer": answer,
                "sources": [],
                "diagnostic_logs": {
                    "router": "KNOWLEDGE",
                    "retrieval": "Database is empty. Skipped retrieval.",
                    "gating": "No chunks evaluated.",
                    "generation": "Called Ollama directly."
                }
            })
        except Exception as e:
            return jsonify({"error": f"Failed calling LLM: {str(e)}"}), 500
 
    # 3. Standard RAG pipeline execution
    try:
        # Step 1: Embed Query
        query_vec = embedder.embed_query(query_text)
        
        # Step 2: Retrieve and Relevance Gate
        chat_retriever = Retrieval_Engine.RetrievalConnector(chat_db)
        retrieval_res = chat_retriever.retrieve_and_gate(query_vec, top_k=top_k, score_threshold=score_threshold)
        
        # Collect gate diagnostic metadata
        gate_logs = []
        all_candidates = sorted(
            retrieval_res["results"] + retrieval_res["rejected_results"],
            key=lambda x: x["score"],
            reverse=True
        )
        for idx, c in enumerate(all_candidates):
            status = "ACCEPT" if c in retrieval_res["results"] else "REJECT"
            gate_logs.append({
                "chunk_id": c["chunk_id"],
                "score": round(c["score"], 4),
                "status": status
            })
            
        diagnostics = {
            "router": "KNOWLEDGE",
            "candidates_retrieved": len(all_candidates),
            "gate_logs": gate_logs,
            "accepted_chunks": len(retrieval_res["results"])
        }
        
        # Gating Check
        if not retrieval_res["has_relevant_context"]:
            answer = "I couldn't find relevant information about that in the uploaded documents."
            
            history.append({"role": "user", "content": query_text})
            history.append({
                "role": "assistant",
                "content": answer,
                "sources": [],
                "diagnostic_logs": diagnostics
            })  
            save_chat_history(chat_id, history)
            
            if len(history) <= 2:
                rename_chat(chat_id, query_text)
            logger.info("RAG gated-out | chat_id=%s | no relevant context", chat_id)
            return jsonify({
                "query_type": "KNOWLEDGE",
                "has_relevant_context": False,
                "answer": answer,
                "sources": [],
                "diagnostic_logs": diagnostics
            })
            
        # Step 3: Construct Grounded Context
        context_result = context_builder.build(query_text, retrieval_res["results"])
        
        # Build prompt incorporating conversation history
        system_prompt = context_builder.system_prompt
        context_text = context_result["context"]
        
        history_block = ""
        if history:
            history_lines = []
            for msg in history[-10:]:
                role_name = "User" if msg["role"] == "user" else "Assistant"
                history_lines.append(f"{role_name}: {msg['content']}")
            history_block = "CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"
            
        prompt = (
            f"SYSTEM:\n{system_prompt}\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"{history_block}"
            f"USER QUESTION:\n{query_text}"
        )
        
        # Step 4: Generate LLM Answer
        response = generator.generate(prompt)
        answer = response["answer"]
        
        # Step 5: Inject Citations
        cited_result = citation_engine.generate_citations(answer, context_result)
        final_answer = cited_result["answer"]
        
        # Save query & answer to history
        # Save query, answer, citations and diagnostics to history
        history.append({
            "role": "user",
            "content": query_text
        })

        history.append({
            "role": "assistant",
            "content": final_answer,
            "sources": cited_result.get("sources", []),
            "diagnostic_logs": diagnostics
        })

        save_chat_history(chat_id, history)
        
        # Auto-rename chat name if it's the first turn
        if len(history) <= 2:
            rename_chat(chat_id, query_text)

        logger.info("RAG answer | chat_id=%s | route=KNOWLEDGE | chunks=%d | has_citations=%s",
                    chat_id, len(retrieval_res["results"]), cited_result["has_citations"])
        return jsonify({
            "query_type": "KNOWLEDGE",
            "has_relevant_context": True,
            "answer": final_answer,
            "sources": cited_result["sources"],
            "diagnostic_logs": diagnostics
        })
        
    except Exception as e:
        logger.error("Query pipeline error | chat_id=%s | error=%s", chat_id, str(e))
        return jsonify({"error": f"Query processing failed: {str(e)}"}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Resets all multi-chat workspaces and removes all ingested document files."""
    errors = []
    
    # 1. Reset vector DB (delete all collections in client)
    try:
        for collection in db.client.list_collections():
            db.client.delete_collection(collection.name)
        # Re-create empty default collection for backward compatibility
        db.collection = db.client.get_or_create_collection(
            name="rag_bot_collection",
            metadata={"hnsw:space": "cosine"}
        )
        print("[Web Backend] All collections successfully reset.")
        logger.info("All ChromaDB collections reset.")
    except Exception as e:
        errors.append(f"DB reset error: {str(e)}")
        logger.error("DB reset error: %s", str(e))
        
    # 2. Reset uploaded files directory
    upload_dir = upload_files.UPLOAD_DIR
    if not secure_rmtree(upload_dir):
        errors.append("File system reset failed.")
    os.makedirs(upload_dir, exist_ok=True)
            
    # 3. Reset metadata directory
    if not secure_rmtree(METADATA_DIR):
        errors.append("Metadata directory reset failed.")
    os.makedirs(METADATA_DIR, exist_ok=True)
            
    # Reset retriever dimension state
    retriever.dimension = None
    logger.info("System reset complete | errors=%d", len(errors))
    return jsonify({
        "success": len(errors) == 0,
        "message": "System database and files reset successfully.",
        "errors": errors
    })

if __name__ == '__main__':
    # Launch local development server
    app.run(host='127.0.0.1', port=5000, debug=True)

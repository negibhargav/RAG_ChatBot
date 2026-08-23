# Document-Aware AI Chatbot using RAG

A local, privacy-focused **Retrieval-Augmented Generation (RAG)** system designed to ground Small Language Model (SLM) responses directly in private uploaded documents. Built with modular python engines, Flask backend, ChromaDB vector database, local embedding models, and an interactive dark-themed web UI.

---

## 🌟 Key Features

- **🔒 Private Multi-Chat Workspaces**:
  - Every chat session operates as an isolated workspace with a unique `chat_id`.
  - Independent ChromaDB collection (`chat_<chat_id>`) per session ensures documents in Chat A are never retrieved in Chat B.
  - Per-chat document upload management and persisted chat history.

- **📄 Multi-Format Document Ingestion**:
  - Supports **PDF** (`.pdf`), **Word** (`.docx`), **Plain Text** (`.txt`), and **Images** (`.png`, `.jpg`, `.jpeg`).
  - OCR integration using `pytesseract` for image text extraction.

- **⚡ Local Embedding & SLM Generation**:
  - Embeddings generated via `BAAI/bge-small-en-v1.5` (`sentence-transformers`).
  - LLM inference powered locally by **Ollama** running `qwen2.5:3b`.

- **🛡️ Query Routing & Relevance Gating**:
  - **Query Router**: Automatically distinguishes conversational chitchat (greetings/thanks) from knowledge-seeking questions.
  - **Relevance Gate**: Filters out low-relevance chunks using a customizable cosine similarity threshold (default: `0.40`). Refuses to answer if no chunks pass the gate.

- **📌 Grounded Citations & Diagnostics**:
  - Automatic sentence-level claim attribution with source badges `[Source N]`.
  - Collapsible diagnostic logs showing candidate scores, gating decisions, and routing metadata in the UI.

- **📊 Comprehensive 5-Phase Evaluation Framework**:
  - Includes automated test runners for Retrieval Gate Sweep, Chunking Configuration, Answer Accuracy, Groundedness, and Citation Evaluation.

- **🪵 Rotating System Logging**:
  - Logging configured in `logs/app.log` (5MB max size, 3 backup files).
  - Logs system startup, query routing, chat events, and uploaded file names (**document text/content is never saved in logs**).

---

## 📁 System Architecture

```text
RAG_bot/
├── app.py                      # Flask REST API backend & endpoint handlers
├── upload_files.py             # File upload helper & extension validation
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── engines/                    # Modular RAG Pipeline Components
│   ├── Extraction_Engine/      # PDF, DOCX, TXT & Image OCR extractors
│   ├── Chunking_Engine/        # Recursive document text splitter
│   ├── Embedding_Engine/       # SentenceTransformer vector embedder
│   ├── Vector_Store_Engine/    # ChromaDB persistent store wrapper
│   ├── Retrieval_Engine/       # Top-K retrieval & relevance score gate
│   ├── Context_Engine/         # Context prompt builder & history formatting
│   ├── Query_Router/           # Conversational vs. Knowledge query classification
│   ├── Generation_Engine/      # Ollama SLM provider & prompt execution
│   ├── Citation_Engine/        # Decoupled sentence-level citation engine
│   └── Evaluation_Engine/      # 5-Phase RAG evaluation benchmark suite
│
├── static/                     # Web UI Frontend Assets
│   ├── css/style.css           # Premium dark theme stylesheet
│   └── js/main.js              # Frontend controller & API communicator
│
├── templates/
│   └── index.html              # Single Page Application HTML interface
│
├── evaluation/                 # Benchmark Datasets & Reports
│   ├── datasets/               # Golden evaluation test cases
│   ├── reports/                # Generated evaluation evaluation logs
│   └── run_evaluation.py       # Full evaluation suite runner script
│
├── tests/                      # Unit & integration test suite (120+ tests)
└── logs/                       # System application logs
```

---

## 🚀 Quick Start Guide

### Prerequisites

1. **Python 3.10+** installed.
2. **Ollama** installed and running locally.
   - Pull the default model:
     ```bash
     ollama pull qwen2.5:3b
     ```
3. *(Optional for Image OCR)* **Tesseract OCR** installed on system PATH.

---

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/RAG_bot.git
   cd RAG_bot
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

### Running the Application

1. **Start the Flask Backend Server**:
   ```bash
   python app.py
   ```

2. **Open the Web UI**:
   Navigate to `http://127.0.0.1:5000` in your web browser.

---

## 🧪 Running Tests & Evaluation

- **Run Unit & Integration Tests**:
  ```bash
  python -m unittest discover -s tests -p "test_*.py"
  ```

- **Run full 5-Phase RAG Evaluation Suite**:
  ```bash
  python evaluation/run_evaluation.py
  ```

---

## 🔧 Pipeline Configuration Settings

Adjustable parameters in `app.py` / Web UI:
- `CHUNK_SIZE`: Default `500` characters.
- `CHUNK_OVERLAP`: Default `50` characters.
- `SCORE_THRESHOLD`: Default `0.40` relevance gate cutoff.
- `TOP_K`: Retrieve top 3 relevant chunks.
- `OLLAMA_MODEL`: Default `qwen2.5:3b`.

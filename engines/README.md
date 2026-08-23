# RAG Engines Architecture

This directory houses the core decoupled modules ("Engines") that form the RAG pipeline. Each engine adheres to a single responsibility principle to allow isolated unit testing, modular swapping, and evaluation.

---

## 🧩 Overview of Modules

| Engine Module | Primary Responsibility | Key Dependency / Provider |
|---|---|---|
| [`Extraction_Engine`](Extraction_Engine/README.md) | Extracts raw text from PDF, DOCX, TXT, and Image (OCR) files | `pypdf`, `python-docx`, `PIL`, `pytesseract` |
| [`Chunking_Engine`](Chunking_Engine/README.md) | Splits large documents into overlapping text chunks | Recursive character splitting |
| [`Embedding_Engine`](Embedding_Engine/README.md) | Converts text chunks and queries into dense vector embeddings | `BAAI/bge-small-en-v1.5` (`sentence-transformers`) |
| [`Vector_Store_Engine`](Vector_Store_Engine/README.md) | Manages vector persistence and collection operations | `chromadb` |
| [`Retrieval_Engine`](Retrieval_Engine/README.md) | Performs top-K similarity search and relevance score gating | Cosine similarity & threshold filtering |
| [`Context_Engine`](Context_Engine/README.md) | Formats retrieved chunks & history into structured system prompts | Custom prompt templates |
| [`Query_Router`](Query_Router/README.md) | Classifies queries as Conversational vs. Knowledge RAG | Intent regex classifier |
| [`Generation_Engine`](Generation_Engine/README.md) | Executes LLM generation against local Ollama SLM server | `qwen2.5:3b` via Ollama API |
| [`Citation_Engine`](Citation_Engine/README.md) | Performs claim matching and appends `[Source N]` citations | Deduplication & Jaccard overlap |
| [`Evaluation_Engine`](Evaluation_Engine/README.md) | 5-Phase evaluation framework for offline quality benchmarking | Custom metric calculators |

---

## 🔄 End-to-End Execution Flow

1. **Ingestion Workflow**:
   `Document File` ➔ `Extraction_Engine` ➔ `Chunking_Engine` ➔ `Embedding_Engine` ➔ `Vector_Store_Engine` (Saved per `chat_id`)

2. **Query & Retrieval Workflow**:
   `User Query` ➔ `Query_Router` ➔ `Embedding_Engine` ➔ `Retrieval_Engine` (Gated) ➔ `Context_Engine` ➔ `Generation_Engine` ➔ `Citation_Engine` ➔ `Response with Sources`

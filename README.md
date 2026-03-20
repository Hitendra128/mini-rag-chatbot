# Mini RAG Chatbot 

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for answering queries using internal documents. The system retrieves relevant document chunks and generates grounded responses using an LLM.

---

## Features
- Semantic search using embeddings
- FAISS-based vector retrieval
- LLM-based answer generation (OpenRouter)
- Streamlit chatbot interface
- Transparent display of retrieved context

---

## Tech Stack
- Python
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS (vector search)
- OpenRouter (LLM API)
- Streamlit (frontend)

---

## How It Works

### 1. Document Processing
- Documents are loaded from `.md` files
- Split into chunks (size = 500, overlap = 50)

### 2. Embeddings
- Each chunk is converted into vector embeddings using Sentence Transformers

### 3. Vector Search
- FAISS is used to store embeddings
- Top-k relevant chunks are retrieved using similarity search

### 4. Answer Generation
- Retrieved chunks are passed as context to the LLM
- The model is instructed to answer ONLY from context
- Prevents hallucination

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py

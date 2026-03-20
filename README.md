# Mini RAG Chatbot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on custom documents.

## Features

* Semantic search using embeddings
* FAISS vector database
* LLM-based answer generation (OpenRouter API)
* Streamlit UI

## Tech Stack

* Python
* Sentence Transformers
* FAISS
* OpenRouter (LLM)
* Streamlit

## How It Works

1. Documents are loaded and split into chunks
2. Each chunk is converted into embeddings
3. FAISS index is created for similarity search
4. User query → converted to embedding
5. Relevant chunks are retrieved
6. LLM generates answer based on context

## How to Run

```bash
pip install -r requirements.txt
set OPENROUTER_API_KEY=your_api_key
streamlit run app.py
```

## Example Query

"What does Indecimal provide?"

## Output

The chatbot retrieves relevant document chunks and generates a contextual answer.

## Note

API keys are managed securely using environment variables.

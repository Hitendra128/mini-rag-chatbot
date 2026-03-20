from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(data_path="data"):
    documents = []

    print("Files:", os.listdir(data_path))

    for file in os.listdir(data_path):
        if file.endswith(".md"):
            file_path = os.path.join(data_path, file)
            print("Loading:", file_path)

            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            documents.extend(docs)

    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    return chunks

def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts)

    return embeddings, model

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index

def search(query, model, index, chunks, k=3):
    query_embedding = model.encode([query])

    D, I = index.search(query_embedding, k)

    results = [chunks[i].page_content for i in I[0]]

    return results

def generate_answer(query, context):
    prompt = f"""
Answer the question ONLY using the context below.
If answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    embeddings, model = create_embeddings(chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    index = create_faiss_index(embeddings)
    print("FAISS index created successfully")

    query = "What does Indecimal provide?"

    results = search(query, model, index, chunks)

    print("\nQuery:", query)
    print("\nTop Retrieved Chunks:\n")

    for i, res in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(res[:300])
        print()

    context = "\n".join(results)

    answer = generate_answer(query, context)

    print("\nFinal Answer:\n")
    print(answer)

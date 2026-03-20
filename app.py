import streamlit as st
from rag_pipeline import load_documents, chunk_documents, create_embeddings, create_faiss_index, search, generate_answer

st.title("Mini RAG Chatbot 🧠")

# Load pipeline once
@st.cache_resource
def setup():
    docs = load_documents()
    chunks = chunk_documents(docs)
    embeddings, model = create_embeddings(chunks)
    index = create_faiss_index(embeddings)
    return chunks, model, index

chunks, model, index = setup()

# User input
query = st.text_input("Ask your question:")

if query:
    results = search(query, model, index, chunks)

    st.subheader("Retrieved Context:")
    for i, res in enumerate(results):
        with st.expander(f"Result {i+1}"):
            st.write(res)

    context = "\n".join(results)
    answer = generate_answer(query, context)

    st.subheader("Final Answer:")
    st.write(answer)
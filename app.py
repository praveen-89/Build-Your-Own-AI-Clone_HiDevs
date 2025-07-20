import streamlit as st
from rag_pipeline import load_documents, create_vector_store, get_rag_chain
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="GenAI Chatbot")

st.title("💬 GenAI Chatbot with Groq")

# Load and create chain
docs = load_documents()
vectorstore = create_vector_store(docs)
rag_chain = get_rag_chain(vectorstore)

# Chat UI
user_question = st.text_input("Ask a question from the document:")

if user_question:
    with st.spinner("Generating answer..."):
        response = rag_chain.run(user_question)
        st.success(response)

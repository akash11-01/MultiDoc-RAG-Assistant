import streamlit as st
import hashlib

from src.vectorstore import get_index, delete_namespace
from src.rag_pipeline import ingest_files, answer_question
from src.config import MAX_FILES

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Chat with Your Documents")

index = get_index()

if "namespace" not in st.session_state:
    st.session_state.namespace = None

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.sidebar:
    files = st.file_uploader(
        "Upload documents (max 5)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if not files:
            st.warning("Please upload at least one file.")
        elif len(files) > MAX_FILES:
            st.warning(f"You can upload a maximum of {MAX_FILES} files.")
        else:
            sig = "".join(f"{f.name}-{f.size}" for f in files)
            ns = hashlib.md5(sig.encode()).hexdigest()

            with st.spinner("Processing documents..."):
                ingest_files(files, index, ns)

            st.session_state.namespace = ns
            st.session_state.chat = []
            st.success("Documents processed successfully.")

    if st.button("Clear Chat & Docs"):
        if st.session_state.namespace:
            delete_namespace(index, st.session_state.namespace)
        st.session_state.chat = []
        st.session_state.namespace = None
        st.success("Session cleared.")

query = st.text_input("Ask a question about your uploaded documents")

if st.button("Send"):
    if not st.session_state.namespace:
        st.warning("Please upload and process documents first.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            result = answer_question(
                question=query,
                index=index,
                namespace=st.session_state.namespace,
                chat_history=st.session_state.chat
            )

        st.session_state.chat.append({
            "question": query,
            "answer": result["answer"],
            "sources": result["sources"]
        })

for item in reversed(st.session_state.chat):
    st.markdown(f"**Q:** {item['question']}")
    st.markdown(f"**A:** {item['answer']}")
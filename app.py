"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from rag import Retriever, answer, chunk_text, load_text

st.set_page_config(page_title="DocuQuery", page_icon="📄")
st.title("📄 DocuQuery: ask questions about your document")
st.caption("Retrieval-Augmented Generation: retrieve relevant chunks, then let an LLM answer from them.")

with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size (words)", 50, 300, 120, 10)
    overlap = st.slider("Chunk overlap (words)", 0, 100, 30, 5)
    top_k = st.slider("Chunks retrieved (k)", 1, 6, 3)

upload = st.file_uploader("Upload a PDF or .txt file", type=["pdf", "txt"])
if upload is None:
    st.info("Upload a document to begin. Try sample_docs/company_handbook.txt")
    st.stop()

key = (upload.name, upload.size, chunk_size, overlap)
if st.session_state.get("key") != key:
    text = load_text(upload.name, upload.getvalue())
    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        st.error("No readable text found (scanned PDFs need OCR).")
        st.stop()
    st.session_state.update(key=key, retriever=Retriever(chunks), n=len(chunks), history=[])
st.success(f"Indexed {st.session_state.n} chunks from {upload.name}")

for q, a, hits in st.session_state.history:
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        st.write(a)
        for n, h in enumerate(hits, 1):
            with st.expander(f"Source {n} (similarity {h.score:.2f})"):
                st.write(h.text)

if question := st.chat_input("Ask a question about the document"):
    with st.spinner("Searching and answering..."):
        reply, hits = answer(question, st.session_state.retriever, top_k)
    st.session_state.history.append((question, reply, hits))
    st.rerun()

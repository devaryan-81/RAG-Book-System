import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

st.title("Debug - PDF Loading")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    st.write(f"Pages loaded: {len(docs)}")
    st.write(f"First page preview: {docs[0].page_content[:200] if docs else 'EMPTY'}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    st.write(f"Chunks created: {len(chunks)}")

    # Filter out empty chunks
    chunks = [c for c in chunks if c.page_content.strip()]
    st.write(f"Non-empty chunks: {len(chunks)}")

    if chunks:
        st.success("PDF loaded correctly!")
    else:
        st.error("All chunks are empty — PDF may be scanned/image-based!")
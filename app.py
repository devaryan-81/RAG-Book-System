import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.title("Debug")

mistral_key = os.getenv("MISTRAL_API_KEY")
st.write("MISTRAL_API_KEY found:", bool(mistral_key))
st.write("Key preview:", mistral_key[:8] + "..." if mistral_key else "NOT FOUND")

# Test embedding directly
try:
    from langchain_mistralai import MistralAIEmbeddings
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    result = embeddings.embed_query("hello world")
    st.success(f"Embeddings working! Vector length: {len(result)}")
except Exception as e:
    st.error(f"Embedding failed: {e}")
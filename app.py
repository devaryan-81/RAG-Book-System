import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import base64
import fitz  # pymupdf

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from mistralai import Mistral

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")
st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")


def extract_text_with_ocr(file_path: str) -> list[Document]:
    """Convert each PDF page to image and extract text via Mistral vision."""
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    pdf = fitz.open(file_path)
    documents = []

    progress = st.progress(0, text="Extracting text from pages...")

    for i, page in enumerate(pdf):
        # Render page to image
        mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR quality
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

        # Send to Mistral vision
        response = client.chat.complete(
            model="mistral-small-2506",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_b64}"
                        },
                        {
                            "type": "text",
                            "text": "Extract all the text from this page exactly as it appears. Output only the extracted text, nothing else."
                        }
                    ]
                }
            ]
        )

        text = response.choices[0].message.content.strip()
        if text:
            documents.append(Document(
                page_content=text,
                metadata={"page": i + 1}
            ))

        progress.progress((i + 1) / len(pdf), text=f"Processing page {i + 1} of {len(pdf)}...")

    pdf.close()
    progress.empty()
    return documents


# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):
        with st.spinner("Running OCR and building vector database..."):

            docs = extract_text_with_ocr(file_path)

            if not docs:
                st.error("Could not extract any text from the PDF.")
            else:
                st.info(f"Extracted text from {len(docs)} pages.")

                embeddings = MistralAIEmbeddings(model="mistral-embed")

                vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=embeddings,
                )

                st.session_state["vectorstore"] = vectorstore
                st.success("Vector database created! Ask a question below.")


# ---------- Q&A ----------
if "vectorstore" in st.session_state:

    llm = ChatMistralAI(model="mistral-small-2506")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context, say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ])

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:
        retriever = st.session_state["vectorstore"].as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
        )

        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.invoke({"context": context, "question": query})
        response = llm.invoke(final_prompt)

        st.write("### AI Answer")
        st.write(response.content)
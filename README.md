# RAG Project 

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on the content of those documents.

## 🚧 Deployment Status

Live Demo : https://rag-book-system-2g6itzlebmfawedq6u5aeh.streamlit.app/

## ✨ Features

- Upload PDF files
- Split documents into chunks
- Generate embeddings using Hugging Face
- Store and retrieve relevant context using Chroma
- Ask questions about uploaded documents through a Streamlit interface

## 📁 Project Structure

```bash
RAG_Project_2/
├── app.py
├── create_database.py
├── main.py
├── requirements.txt
├── .env
├── chroma_db/
├── document loaders/
├── retrievers/
└── vector_store/

📦 Installation
Install the required packages:
pip install -r requirements.txt

🔐 Environment Variables
Create a .env file in the project root and add:
MISTRAL_API_KEY=your_mistral_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token


▶️ Run the App Locally
Start the Streamlit app
streamlit run app.py

How to use
Upload a PDF file
Click Create Vector Database
Ask your question in the input area


🧠 Tech Stack
Python
Streamlit
LangChain
Chroma
Hugging Face Embeddings
Mistral AI

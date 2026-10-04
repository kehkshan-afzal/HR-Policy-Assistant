# 📋 HR Policy Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG) application that allows users to upload an HR Policy PDF and ask questions about the document.

The application retrieves the most relevant sections of the uploaded HR policy and uses Google's Gemini API to generate an answer based only on the retrieved content.

## 🚀 Live Demo

Add your Streamlit Community Cloud URL here after deployment.

Example:

https://your-app-name.streamlit.app/

## 📌 Features

- Upload an HR Policy PDF
- Extract PDF text using PyMuPDF
- Split document text into chunks
- Generate semantic embeddings using Sentence Transformers
- Store embeddings in FAISS
- Search for the most relevant policy sections
- Generate answers using Gemini
- Answers are based only on retrieved HR policy content
- Secure Gemini API key using Streamlit Secrets
- Simple and beginner-friendly interface

## 🧠 How RAG Works

The application follows this process:

PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Search
    ↓
Relevant Policy Sections
    ↓
Gemini
    ↓
Final Answer

## 🛠️ Technologies Used

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- NumPy
- Google Gemini API

## 📁 Project Structure

```text
HR-Policy-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

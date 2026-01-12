# 📚 Multimodal RAG (Retrieval-Augmented Generation) System

A **Retrieval-Augmented Generation (RAG)** based project that allows users to upload **audio, images, text, and PDF files**, extract knowledge from them, and ask questions using a Large Language Model.

The system combines **document parsing, embeddings, vector search, and LLM reasoning** to deliver accurate, context-aware answers.

---

## 🚀 Features

- Upload and process multiple file types:
  - 📄 PDF
  - 📝 Text files
  - 🖼️ Images (OCR)
  - 🔊 Audio (Speech-to-Text)
- Automatic content extraction
- Text chunking & embeddings
- Vector search using FAISS
- Hybrid retrieval (semantic + keyword)
- Question answering using LLM
- FastAPI backend
- Modular and extensible architecture

---

## 🧠 Architecture Overview

User Uploads File  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ 

File Type Detection  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ 

Content Extraction (PDF / Image/ Audio / Text) 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ 

Chunking & Preprocessing  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ 

Embedding Generation  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ 

Vector Store (FAISS)  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ 

Retriever  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ 

RAG Prompt Construction  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ 

LLM Inference  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ 

Final Answer  


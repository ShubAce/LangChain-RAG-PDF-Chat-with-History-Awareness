# Conversational RAG with PDF Uploads and Session-based Chat History

This project is a Streamlit-based application that enables users to chat with the contents of uploaded PDF documents using a **Retrieval-Augmented Generation (RAG)** approach. It combines the capabilities of **LangChain**, **Chroma vector store**, **HuggingFace embeddings**, and **Groq’s Gemma2-9b-It** language model to deliver accurate, context-aware answers.

Users can upload one or multiple PDFs, and the app extracts and splits the content into chunks for vector-based retrieval. The model uses **KMeans clustering-enhanced embeddings** (`all-MiniLM-L6-v2`) to efficiently organize information and retrieve relevant sections during conversations.

The system supports **multi-turn dialogues with memory**, maintaining session-based chat history using LangChain's message history tools. It also reformulates follow-up questions into standalone queries to ensure relevance and clarity before retrieving answers.

## Features
- Upload and process multiple PDF files
- RAG-based question answering using Groq LLM
- Session-aware memory for consistent conversations
- Intelligent context reformulation for better retrieval
- User-friendly Streamlit interface

## Tech Stack
- **LangChain** for RAG pipelines and prompt handling  
- **Chroma** for vector storage and document retrieval  
- **HuggingFace Transformers** for semantic embeddings  
- **Groq (Gemma2-9b-It)** as the underlying LLM  
- **Streamlit** for the frontend UI

This tool is ideal for researchers, students, and professionals who need to extract and interact with knowledge from large documents efficiently.

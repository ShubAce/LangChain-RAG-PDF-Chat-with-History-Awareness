# LangChain-RAG-PDF-Chat-with-History-Awareness
This project is a Streamlit-based application that enables interactive, context-aware conversations over PDF documents using Retrieval-Augmented Generation (RAG). Built with LangChain, Chroma, HuggingFace embeddings, and Groq’s Gemma2-9b-It LLM, it allows users to upload one or more PDF files and engage in multi-turn Q&A backed by retrieval from document content and persistent session memory.
## Key Features
-Multi-PDF Upload: Supports uploading multiple PDF files for content extraction and retrieval.
-Retrieval-Augmented Generation (RAG): Combines question reformulation with contextual document retrieval for accurate answers.
-Session-based Memory: Maintains chat history across sessions for personalized and context-aware interaction.
-KMeans-based Vector Embeddings: Utilizes all-MiniLM-L6-v2 embeddings from HuggingFace to build a robust vector index.
-Streaming Interface: User-friendly interface built with Streamlit for real-time interaction and visualization.

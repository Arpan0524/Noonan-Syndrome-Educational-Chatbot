# Noonan-Syndrome-Educational-Chatbot

An AI-powered medical educational chatbot using Python, Streamlit, and Hugging Face Sentence Transformers. The chatbot uses semantic search and cosine similarity to understand user questions contextually and retrieve the most relevant response from a CSV-based medical dataset. I deployed the application publicly using Hugging Face Spaces, creating a fully functional web-based AI healthcare assistant.

COMPLETE ARCHITECTURE:-
User
 ↓
Streamlit Web UI
 ↓
Sentence Transformer Model
 ↓
Convert Question → Embedding
 ↓
Cosine Similarity Search
 ↓
Medical CSV Dataset
 ↓
Best Matching Answer
 ↓
Display Response

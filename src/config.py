import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(key: str, default: str = ""):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_secret("PINECONE_INDEX_NAME", "multi-doc-rag")
PINECONE_CLOUD = get_secret("PINECONE_CLOUD", "aws")
PINECONE_REGION = get_secret("PINECONE_REGION", "us-east-1")

GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

MAX_FILES = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 7
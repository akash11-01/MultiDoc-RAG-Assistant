import streamlit as st
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

@st.cache_resource
def get_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def embed_texts(texts):
    model = get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()

def embed_query(query):
    model = get_model()
    return model.encode(query, normalize_embeddings=True).tolist()
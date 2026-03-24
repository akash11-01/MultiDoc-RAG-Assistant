import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSION
)

@st.cache_resource
def get_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [i["name"] for i in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )

    return pc.Index(PINECONE_INDEX_NAME)

def upsert_chunks(index, chunks, embeddings, namespace):
    vectors = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{namespace}-{i}",
            "values": emb,
            "metadata": {
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "chunk_id": chunk["metadata"]["chunk_id"]
            }
        })

    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100], namespace=namespace)

def query_chunks(index, query_embedding, namespace, top_k):
    res = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    return res.matches if hasattr(res, "matches") else res["matches"]

def delete_namespace(index, namespace):
    index.delete(delete_all=True, namespace=namespace)
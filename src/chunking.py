from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []

    for doc in docs:
        text = doc["text"]
        metadata = doc["metadata"]

        split_texts = splitter.split_text(text)

        for i, chunk in enumerate(split_texts):
            chunks.append({
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_id": i
                }
            })

    return chunks
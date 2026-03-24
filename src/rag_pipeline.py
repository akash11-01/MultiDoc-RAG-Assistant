from src.loaders import parse_uploaded_file
from src.chunking import chunk_documents
from src.embeddings import embed_texts, embed_query
from src.vectorstore import upsert_chunks, query_chunks
from src.llm import rewrite_question_with_history, generate_answer
from src.config import TOP_K

def ingest_files(files, index, namespace):
    docs = []

    for f in files:
        parsed = parse_uploaded_file(f)
        docs.extend(parsed)

    if not docs:
        raise ValueError("No text could be extracted from the uploaded files.")

    chunks = chunk_documents(docs)

    if not chunks:
        raise ValueError("Text was extracted, but no chunks were created.")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    upsert_chunks(index, chunks, embeddings, namespace)

    return {
        "num_docs": len(docs),
        "num_chunks": len(chunks)
    }


def format_chat_history(chat_history, max_turns=4):
    if not chat_history:
        return ""

    recent_history = chat_history[-max_turns:]
    history_lines = []

    for item in recent_history:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()

        if q:
            history_lines.append(f"User: {q}")
        if a:
            history_lines.append(f"Assistant: {a}")

    return "\n".join(history_lines)


def answer_question(question, index, namespace, chat_history=None):
    history_text = format_chat_history(chat_history or [])
    standalone_question = rewrite_question_with_history(question, history_text)

    q_emb = embed_query(standalone_question)
    matches = query_chunks(index, q_emb, namespace, TOP_K)

    if not matches:
        return {
            "answer": "I could not find relevant information in the uploaded documents.",
            "sources": []
        }

    context_parts = []
    sources = []

    for m in matches:
        meta = m.metadata if hasattr(m, "metadata") else m["metadata"]

        text = meta.get("text", "").strip()
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")

        if text:
            context_parts.append(f"[Source: {source}, Page: {page}]\n{text}")
            sources.append((source, page))

    if not context_parts:
        return {
            "answer": "I could not find readable text in the retrieved document chunks.",
            "sources": sources
        }

    final_context = "\n\n".join(context_parts)

    answer = generate_answer(
        question=question,
        standalone_question=standalone_question,
        context=final_context,
        chat_history=history_text
    )

    return {
        "answer": answer,
        "sources": sources
    }
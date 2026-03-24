# Multi-Doc RAG Chatbot

A lightweight **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit**.
Upload multiple PDF/TXT files, index them in Pinecone, and ask questions grounded in the uploaded content.

---

## Features

- Upload up to 5 documents (`.pdf`, `.txt`)
- Extracts PDF text page-wise (PyMuPDF)
- Chunks text using recursive splitting
- Creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Stores and retrieves vectors in Pinecone (namespace per upload session)
- Uses Groq LLM for:
  - follow-up question rewriting (chat-history aware)
  - final grounded answer generation
- Session-level chat flow in Streamlit

---

## Tech Stack

- **Frontend/UI:** Streamlit
- **Document parsing:** PyMuPDF (`fitz`)
- **Chunking:** LangChain Text Splitters
- **Embeddings:** SentenceTransformers
- **Vector DB:** Pinecone
- **LLM:** Groq

---

## Environment Variables

Create a `.env` file in project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=multi-doc-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Notes:
- The app also supports `st.secrets` in Streamlit deployments.
- Default values for index/model/cloud/region are defined in `src/config.py`.

---

## Installation

```bash
python -m venv venv
```

### Windows (PowerShell)

```bash
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

---

## How It Works

1. Upload documents in sidebar.
2. On **Process Documents**:
	- files are parsed (`src/loaders.py`)
	- text is chunked (`src/chunking.py`)
	- chunks are embedded (`src/embeddings.py`)
	- vectors are upserted into Pinecone namespace (`src/vectorstore.py`)
3. Ask a question:
	- recent chat is formatted
	- question is rewritten to standalone form (`src/llm.py`)
	- similar chunks are retrieved from Pinecone
	- final answer is generated from retrieved context only

---

## Current Limits / Defaults

- `MAX_FILES = 5`
- `CHUNK_SIZE = 1000`
- `CHUNK_OVERLAP = 200`
- `TOP_K = 7`
- Embedding dimension: `384`

These are configurable in `src/config.py`.

---

## Common Issues

### 1) `No text could be extracted from the uploaded files.`
- Ensure PDFs contain selectable text (not scanned image-only pages).
- Try with TXT to validate pipeline.

### 2) Pinecone connection/index errors
- Verify `PINECONE_API_KEY`, cloud, region, and index settings.

### 3) Groq errors / empty responses
- Verify `GROQ_API_KEY` and `GROQ_MODEL`.

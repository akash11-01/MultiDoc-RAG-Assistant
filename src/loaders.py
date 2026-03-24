import fitz

def load_pdf(file_bytes: bytes, file_name: str):
    docs = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text("text")
        text = text.strip()

        if text:
            docs.append({
                "text": text,
                "metadata": {
                    "source": file_name,
                    "page": page_num
                }
            })

    pdf.close()
    return docs

def load_txt(file_bytes: bytes, file_name: str) -> list[dict]:
    text = file_bytes.decode("utf-8", errors="ignore")
    if not text.strip():
        return []
    return [{
        "text": text,
        "metadata": {
            "source": file_name,
            "page": 1
        }
    }]

def parse_uploaded_file(uploaded_file) -> list[dict]:
    file_name = uploaded_file.name
    file_bytes = uploaded_file.read()

    if file_name.lower().endswith(".pdf"):
        return load_pdf(file_bytes, file_name)

    if file_name.lower().endswith(".txt"):
        return load_txt(file_bytes, file_name)

    raise ValueError(f"Unsupported file type: {file_name}")
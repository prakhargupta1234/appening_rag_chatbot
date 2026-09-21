from pathlib import Path
import requests
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from .config import PDF_URL, PDF_PATH, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

def download_pdf():
    path = Path(PDF_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    print("Downloading knowledge-base PDF...")
    r = requests.get(PDF_URL, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)

def chunk_text(text):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks

def main():
    download_pdf()
    reader = PdfReader(PDF_PATH)
    documents, metadatas, ids = [], [], []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk_no, chunk in enumerate(chunk_text(text)):
            documents.append(chunk)
            metadatas.append({"source": "Agentic AI eBook", "page": page_no, "chunk": chunk_no})
            ids.append(f"p{page_no}-c{chunk_no}")

    if not documents:
        raise RuntimeError("No text could be extracted from the PDF.")

    print(f"Extracted {len(documents)} chunks.")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    batch = 100
    for i in range(0, len(documents), batch):
        collection.add(
            ids=ids[i:i+batch],
            documents=documents[i:i+batch],
            metadatas=metadatas[i:i+batch],
            embeddings=embeddings[i:i+batch],
        )

    print(f"Indexed {collection.count()} chunks in Chroma.")
    print("Ingestion complete.")

if __name__ == "__main__":
    main()

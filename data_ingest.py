# data_ingest.py
import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

PDF_PATH = "manual.pdf"
PERSIST_DIR = "./chroma_db"

def load_pdf_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": i, "text": text})
    return pages

def chunk_pages(pages, chunk_size=400, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = []
    for p in pages:
        # create a pseudo document with page-level metadata
        docs.extend(splitter.split_text(p["text"]))
    # We'll keep metadata mapping when adding to DB
    return docs

def build_vector_db(docs, persist_dir=PERSIST_DIR, collection_name="manual_collection"):
    # Embedding model
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Chroma client with persistence
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name="manual_collection")

    texts = [d for d in docs]
    embeddings = emb_model.encode(texts, show_progress_bar=True).tolist()

    # Generate simple unique ids
    ids = [f"doc_{i}" for i in range(len(texts))]
    metadatas = [{"source": f"manual_page_chunk_{i}"} for i in range(len(texts))]

    # Add to chroma (if collection already has documents you may want to clear or skip)
    collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
    # client.persist()
    print(f"Persisted {len(texts)} chunks to Chroma at {persist_dir}")

if __name__ == "__main__":
    pages = load_pdf_text(PDF_PATH)
    docs = chunk_pages(pages)
    build_vector_db(docs)

import re
from pathlib import Path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import chromadb

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "manual_collection"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"


def load_manual(path):
    """Load text from a PDF file."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def split_by_sections(text):
    """
    Split manual by numbered sections (e.g., 3.3.1 Latte, 2.4 Closing Checklist).
    Each section becomes one doc.
    """
    pattern = r"^(\d+(\.\d+)*)(\s+[A-Za-z].*)$"
    sections = re.split(r"(?=^\d+(\.\d+)*\s+[A-Za-z].*$)", text, flags=re.MULTILINE)
    grouped = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        lines = sec.splitlines()
        if lines:
            matches = re.match(pattern, lines[0])
            if matches:
                section_num = matches.group(1)
                title = matches.group(3).strip()
                structured_line = f"{section_num} {title}"
            else:
                structured_line = lines[0]
        else:
            structured_line = "Untitled Section"
        grouped.append({"title": structured_line, "content": sec})
    return grouped


def build_vector_db(docs, persist_dir=PERSIST_DIR, collection_name=COLLECTION_NAME):
    emb_model = SentenceTransformer(EMB_MODEL_NAME)
    client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_dir
    )
)

    
    # --- ADD THIS LINE ---
    try:
        client.delete_collection(name=collection_name)
    except:
        pass  # Collection might not exist yet, which is fine
        
    collection = client.get_or_create_collection(name=collection_name)

    texts = [d["content"] for d in docs]
    embeddings = emb_model.encode(texts, show_progress_bar=True).tolist()
    ids = [f"section_{i}" for i in range(len(texts))]
    metadatas = [{"title": d["title"]} for d in docs]

    collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
    print(f"✅ Persisted {len(texts)} sections into Chroma at {persist_dir}")

if __name__ == "__main__":
    manual_path = "manual_chat.pdf"   # update filename if needed
    text = load_manual(manual_path)
    docs = split_by_sections(text)
    build_vector_db(docs)

import re
from pathlib import Path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import chromadb

PERSIST_DIR = "./chroma_db"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"



def load_pdf(path):
    """Load text from a PDF file."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_txt(path):
    """Load text from a TXT file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
def load_manual(path):
    """Auto-detect file type and load content."""
    if path.lower().endswith(".pdf"):
        return load_pdf(path)
    elif path.lower().endswith(".txt"):
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

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


def build_vector_db(docs, collection_name, persist_dir=PERSIST_DIR):
    client = chromadb.PersistentClient(path=persist_dir)    
    # --- ADD THIS LINE ---
    try:
        client.delete_collection(name=collection_name)
    except:
        pass  # Collection might not exist yet, which is fine
        
    collection = client.get_or_create_collection(name=collection_name)

    texts = [d["content"] for d in docs]
    ids = [f"{collection_name}_section_{i}" for i in range(len(texts))]
    metadatas = [{"title": d["title"]} for d in docs]

    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    print(f"✅ Persisted {len(texts)} sections into Chroma (collection='{collection_name}')")

if __name__ == "__main__":
    manuals = {
        "arai": "manual_chat.pdf",
        "jai": "career_manual.txt",
        "kai": "knowledge_manual.txt"
    }
    for name, filepath in manuals.items():
        if not Path(filepath).exists():
            print(f"⚠️ Skipping {filepath} (file not found)")
            continue
        print(f"\n📖 Ingesting {filepath} into collection '{name}_collection'")
        text = load_manual(filepath)
        docs = split_by_sections(text)
        build_vector_db(docs, collection_name=f"{name}_collection", persist_dir=PERSIST_DIR)

# arai_rag.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import pipeline, PipelineException
import os
import textwrap

# Settings
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "manual_collection"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"   # small & fast; swap if you use cloud API
TOP_K = 3

# init
emb_model = SentenceTransformer(EMB_MODEL_NAME)
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR))
collection = client.get_collection(name=COLLECTION_NAME)

try:
    generator = pipeline("text2text-generation", model=GEN_MODEL)
except Exception as e:
    print("Local generator load failed:", e)
    generator = None

def retrieve(query, top_k=TOP_K):
    qvec = emb_model.encode([query])[0].tolist()
    res = collection.query(query_embeddings=[qvec], n_results=top_k, include=["documents","metadatas","distances"])
    docs = []
    for idx in range(len(res["documents"][0])):
        docs.append({
            "text": res["documents"][0][idx],
            "meta": res["metadatas"][0][idx],
            "score": res["distances"][0][idx]
        })
    return docs

PROMPT_TEMPLATE = """
You are a clear, concise assistant. Use the following retrieved excerpts (numbered). Provide a short step-by-step answer to the user's question and reference which excerpt(s) you used by number.

EXCERPTS:
{excerpts}

Question: {question}

Answer (short, step-by-step). Then add "SOURCES:" and list excerpt numbers used.
"""

def answer_question(query, top_k=TOP_K):
    hits = retrieve(query, top_k=top_k)
    # build excerpt block
    excerpts = []
    for i, h in enumerate(hits, start=1):
        text = h["text"]
        text = textwrap.shorten(text, width=800, placeholder="...")
        excerpts.append(f"[{i}] {text}")

    prompt = PROMPT_TEMPLATE.format(excerpts="\n\n".join(excerpts), question=query)

    # generate
    if generator:
        out = generator(prompt, max_length=256)[0]["generated_text"]
    else:
        # fallback: naive extractive answer (very simple) -> return concatenated hits
        out = " (Fallback) See excerpts: " + " | ".join([f"[{i}]" for i in range(1, len(hits)+1)])

    # Prepare sources list to return to UI
    sources = [{"id": h.get("meta", {}).get("source","unknown"), "text": h["text"]} for h in hits]
    return out, sources

if __name__ == "__main__":
    q = "How do I process a refund?"
    ans, sources = answer_question(q)
    print("ANSWER:\n", ans)
    print("SOURCES:\n", sources)

# arai_rag.py
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline
import textwrap

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "manual_collection"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"   # fast, local model
TOP_K = 3

# Init embeddings & DB
emb_model = SentenceTransformer(EMB_MODEL_NAME)
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

# Init generator (local HF model)
try:
    generator = pipeline("text2text-generation", model=GEN_MODEL)
except Exception as e:
    print("Generator load failed:", e)
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
You are a clear, concise assistant. Use the following retrieved excerpts (numbered).
Provide a short step-by-step answer to the user's question and reference which excerpt(s) you used.

EXCERPTS:
{excerpts}

Question: {question}

Answer:
"""

def answer_question(query, top_k=TOP_K):
    hits = retrieve(query, top_k=top_k)
    excerpts = []
    for i, h in enumerate(hits, start=1):
        text = textwrap.shorten(h["text"], width=500, placeholder="...")
        excerpts.append(f"[{i}] {text}")

    prompt = PROMPT_TEMPLATE.format(excerpts="\n\n".join(excerpts), question=query)

    if generator:
        out = generator(prompt, max_length=256)[0]["generated_text"]
    else:
        out = "Fallback: Sources only.\n" + "\n".join(excerpts)

    sources = [{"meta": h["meta"], "text": h["text"]} for h in hits]
    return out, sources

if __name__ == "__main__":
    question = input("Enter your question: ")
    # q = "How do I process a refund?"
    ans, sources = answer_question(question)
    print("ANSWER:\n", ans)
    # print("SOURCES:\n", sources)

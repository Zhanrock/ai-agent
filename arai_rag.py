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

def answer_question(query, style="bullet", top_k=TOP_K):
    """
    Answer user question based on retrieved chunks.
    style = "bullet" or "sentence"
    """
    hits = retrieve(query, top_k=top_k)
    excerpts = []
    for i, h in enumerate(hits, start=1):
        text = textwrap.shorten(h["text"], width=600, placeholder="...")
        excerpts.append(f"[{i}] {text}")

    if style == "bullet":
        format_instructions = "Write the answer in 3–6 concise bullet points."
    else:
        format_instructions = "Write the answer in 2–3 short paragraphs."

    prompt = f"""
    You are a precise assistant for employees.
    Only use excerpts directly relevant to the question. 
    Ignore anything unrelated (even if retrieved).

    Question: {query}

    Excerpts:
    {chr(10).join(excerpts)}

    Answer format: {format_instructions}
    """

    if generator:
        out = generator(prompt, max_new_tokens=300)[0]["generated_text"]
    else:
        out = "Fallback: " + "\n".join(excerpts)

    sources = [{"meta": h["meta"], "text": h["text"]} for h in hits]
    return out, sources


# Test run
if __name__ == "__main__":
    q = input("Enter your question: ")
    style_choice = input("Answer style? (bullet/sentence): ").strip().lower()
    ans, sources = answer_question(q, style=style_choice)
    print("\nANSWER:\n", ans)
    # print("\nSOURCES:\n", sources)
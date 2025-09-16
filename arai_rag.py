from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline, AutoTokenizer
import re

# ---------------- CONFIG ----------------
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "manual_collection"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"   # small but decent
TOP_K = 5
# ----------------------------------------

# Init embeddings & DB
emb_model = SentenceTransformer(EMB_MODEL_NAME)
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

# Init generator + tokenizer
try:
    generator = pipeline("text2text-generation", model=GEN_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, use_fast=True)
    print("Device set to use", generator.device)
except Exception as e:
    print("❌ Generator load failed:", e)
    generator, tokenizer = None, None


# ---------------- HELPERS ----------------
def retrieve(query, top_k=TOP_K):
    qvec = emb_model.encode([query])[0].tolist()
    res = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = []
    for idx in range(len(res["documents"][0])):
        docs.append({
            "text": res["documents"][0][idx],
            "meta": res["metadatas"][0][idx],
            "score": res["distances"][0][idx]
        })
    return docs


def split_sentences(text):
    sents = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


def extract_relevant_sentences(hits, query_keywords, max_sentences_per_hit=3):
    extracted = []
    for i, h in enumerate(hits):
        sents = split_sentences(h["text"])
        for s in sents:
            extracted.append((i, s))
            if sum(1 for x in extracted if x[0] == i) >= max_sentences_per_hit:
                break
    return extracted


# ---------------- MAIN ANSWER ----------------
def answer_question(query, style="bullet", top_k=TOP_K, max_prompt_tokens_margin=50):
    kws = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    hits = retrieve(query, top_k=top_k)

    # Find the most relevant section (e.g., "Latte")
    target_section = None
    for h in hits:
        title = h["meta"].get("title", "").lower()
        if "latte" in query.lower() and "latte" in title:
            target_section = h
            break
    if not target_section:
        target_section = hits[0] if hits else None

    # Only extract sentences from the target section
    extracted = []
    if target_section:
        sents = split_sentences(target_section["text"])
        for s in sents[:6]:  # up to 6 sentences
            extracted.append((0, s))

    if not extracted:
        extracted = [(0, split_sentences(hits[0]["text"])[0])] if hits else []

    pieces = []
    seen = set()
    for hit_idx, s in extracted:
        if s not in seen:
            pieces.append((hit_idx, s))
            seen.add(s)

    if style == "bullet":
        style_instr = "Write the answer as 3-6 concise bullet points. Keep exact numbers and steps."
    else:
        style_instr = "Write the answer in 2-3 short paragraphs. Keep exact numbers and steps."

    context_text = "\n".join(f"- {s}" for _, s in pieces)

    prompt = f"""You are an accurate assistant for employees.
ONLY use the excerpts below to answer the question.
Do NOT invent extra steps.

You are a helpful assistant. Answer ONLY from the provided sources. 
- If multiple recipes are retrieved, identify the section most relevant to the user’s question.
- Do not mix instructions from other drinks or sections.
- If the user asks 'How to make latte?', only return the steps from the 'Latte' section.
- Stop your answer at the last step in that section, even if more text is retrieved.
- If unsure, only answer with the exact section text.
- Answer in the style requested (bullet or sentence).


Question: {query}

Excerpts:
{context_text}

Answer format: {style_instr}
"""

    if generator:
        out = generator(prompt, max_new_tokens=220)[0]["generated_text"]
    else:
        out = "⚠️ Generator not available."

    # Collect source section titles
    source_ids = []
    title = target_section["meta"].get("title", "Unknown Section") if target_section else "Unknown Section"
    preview = target_section["text"][:200] if target_section else ""
    source_ids.append({"section": title, "preview": preview})

    return out.strip(), source_ids


# ---------------- INTERACTIVE ----------------
if __name__ == "__main__":
    q = input("Enter your question: ")
    style_choice = input("Answer style? (bullet/sentence): ").strip().lower()
    ans, sources = answer_question(q, style=style_choice)
    print("\nANSWER:\n", ans)
    print("\nSOURCES:")
    for s in sources:
        print(f"- {s['section']} → {s['preview'][:80]}...\n")

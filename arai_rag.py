from sentence_transformers import SentenceTransformer
import chromadb
chromadb.config.telemetry = False

import chromadb.segment.impl.metadata.sqlite as sqlite_module

def safe_decode_seq_id(seq_id_bytes):
    if isinstance(seq_id_bytes, int):
        return seq_id_bytes
    if len(seq_id_bytes) == 8:
        return int.from_bytes(seq_id_bytes, "big")
    elif len(seq_id_bytes) == 24:
        return int.from_bytes(seq_id_bytes, "big")
    else:
        raise ValueError(f"Unexpected seq_id_bytes: {seq_id_bytes}")

sqlite_module._decode_seq_id = safe_decode_seq_id

from chromadb.utils import embedding_functions
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
# Init Chroma client + embedding function
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMB_MODEL_NAME
)
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
    res = collection.query(
        query_texts=[query],
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
    # Split on punctuation or "Step n:"
    sents = re.split(r'(?<=[\.\!\?])\s+|(?=Step \d+:)', text.strip())
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

    # If no hits or the top hit is not relevant, return custom message
    if not hits or hits[0]["score"] > 1.0:  # adjust threshold as needed
        return "Sorry, that question is unrelated to the manual and cannot be answered.", []

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
        # Remove section title if present
        title = target_section["meta"].get("title", "").strip()
        sents = [s for s in sents if s.strip() and s.strip() != title]
        for s in sents:
            extracted.append((0, s))

    if not extracted:
        extracted = [(0, split_sentences(hits[0]["text"])[0])] if hits else []

    pieces = []
    seen = set()
    for hit_idx, s in extracted:
        if s not in seen:
            pieces.append((hit_idx, s))
            seen.add(s)

    title = target_section["meta"].get("title", "").strip() if target_section else ""
    if style == "bullet":
        style_instr = (
            "Write the answer as Markdown bullet points. "
            "Each item should be on a new line and listed only once. "
            "Do not repeat or truncate items. "
            "Do not invent extra steps."
        )
        context_text = "\n".join(f"• {s}" for _, s in pieces if s.strip() and s.strip() != title)
    else:
        style_instr = "Write the answer in 2-3 short paragraphs. Keep exact numbers and steps."
        context_text = " ".join(s for _, s in pieces if s.strip() and s.strip() != title)

    prompt = f"""You are an accurate assistant for employees.
ONLY use the excerpts below to answer the question.
Do NOT invent extra steps.

Question: {query}

Excerpts:
{context_text}

Answer format: {style_instr}
"""

    if generator:
        out = generator(prompt, max_new_tokens=220)[0]["generated_text"]
    else:
        out = "⚠️ Generator not available."

    # Try to split steps if model outputs in a single line
    if style == "bullet":
        # Split on any "Step n:" or "•"
        out = re.sub(r"(Step \d+:)", r"\n• \1", out)
        out = re.sub(r"(•)", r"\n•", out)
        lines = [line.strip() for line in out.split("\n") if line.strip()]
        title = target_section["meta"].get("title", "").strip() if target_section else ""
        lines = [line for line in lines if title.lower() not in line.lower()]
        # Stop at first line that looks like a new section header (starts with number and dot)
        section_header_pattern = re.compile(r"^\d+(\.\d+)*\s*[\.:]")
        filtered = []
        for x in lines:
            if section_header_pattern.match(x):
                break
            # Only keep lines that look like bullets or steps
            if x.startswith("•") or x.startswith("-") or re.match(r"Step \d+:", x):
                filtered.append(x)
        # Add bullet if missing
        for i, line in enumerate(filtered):
            if not line.startswith("•") and not line.startswith("-"):
                filtered[i] = "• " + line
        # Improved deduplication: match by the start of the line (first 30 chars, lowercased)
        seen = set()
        deduped = []
        for x in filtered:
            norm = re.sub(r'\s+', ' ', x[:30].lower()).strip()
            if norm not in seen and len(x) > 10:
                deduped.append(x)
                seen.add(norm)
        out = "\n".join(deduped)
    elif style == "sentence":
        title = target_section["meta"].get("title", "").strip() if target_section else ""
        out_lines = [line for line in out.split("\n") if title.lower() not in line.lower()]
        out = " ".join(out_lines).strip()

    # Collect source section titles
    source_ids = []
    title = target_section["meta"].get("title", "Unknown Section") if target_section else "Unknown Section"
    preview = target_section["text"][:200] if target_section else ""
    source_ids.append({"section": title, "preview": preview})

    # If the answer is empty, show the full source section text as the answer
    if not out.strip():
        # Use the entire target section text, skipping the title if present
        title = target_section["meta"].get("title", "").strip() if target_section else ""
        section_text = target_section["text"] if target_section else ""
        # Remove the title from the start if present
        if section_text.startswith(title):
            section_text = section_text[len(title):].strip()
        out = section_text.strip()

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

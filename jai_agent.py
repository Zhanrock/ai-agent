# jai_agent.py
import os
import csv
import json
import re
from pathlib import Path
from openai import OpenAI
import chromadb

# Config
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "jai_collection"
OPENAI_MODEL = "gpt-4o-mini"
TOP_K = 4
MOCK_PERF = "mock_performance.csv"      # provide sample CSV locally
CAREER_PATH = "career_path.json"        # provide career_path.json locally
NUDGE_LIB = "nudge_library.json"        # provide nudge library json locally

# Init clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# Utilities
def load_json(path):
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

career_map = load_json(CAREER_PATH)
nudge_lib = load_json(NUDGE_LIB)

def retrieve_docs(query, top_k=TOP_K):
    res = collection.query(query_texts=[query], n_results=top_k, include=["documents","metadatas","distances"])
    docs = []
    for i in range(len(res["documents"][0])):
        docs.append({
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "score": res["distances"][0][i]
        })
    return docs

def get_employee_row(employee_id):
    if not Path(MOCK_PERF).exists():
        return None
    with open(MOCK_PERF, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            if r.get("employee_id") == str(employee_id) or r.get("name","").lower()==str(employee_id).lower():
                return r
    return None

# Core: determine a single growth nudge
def compute_growth_nudge(employee_id):
    row = get_employee_row(employee_id)
    if not row:
        return "No performance data found for this employee."

    current_role = row.get("role")
    skills_unlocked = json.loads(row.get("skills_unlocked") or "[]") if row.get("skills_unlocked") else []
    next_role_info = career_map.get(current_role, {})
    next_role = next_role_info.get("next_role")
    skills_required = next_role_info.get("skills_required", []) if next_role_info else []

    # find first missing required skill:
    growth_skill = None
    for s in skills_required:
        if s not in skills_unlocked:
            growth_skill = s
            break

    # fallback if none
    if not growth_skill and skills_required:
        growth_skill = skills_required[0]  # suggest refinement

    # lookup nudge text
    nudge_text = nudge_lib.get(growth_skill, None)
    if nudge_text:
        return nudge_text

    # If no explicit nudge, create small prompt using retrieved docs
    prompt_context = f"Employee: {row.get('name')} Role: {current_role} NextRole: {next_role} GrowthSkill: {growth_skill}\n\nUse the following excerpts to craft a specific 'next step' nudge for this employee (1 concise actionable step)."
    docs = retrieve_docs(growth_skill or current_role)
    context_text = "\n".join([d["text"][:800] for d in docs])
    prompt = f"{prompt_context}\n\nExcerpts:\n{context_text}\n\nOutput a single short actionable nudge (one sentence)."

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"You are an encouraging career coach. Provide one specific actionable step."},
                {"role":"user","content":prompt}
            ],
            max_tokens=150, temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

# Public function: ask Jai questions (RAG + persona)
def ask_jai(query, employee_id=None, style="sentence"):
    # Retrieve relevant manual sections first
    docs = retrieve_docs(query)
    context_text = "\n".join([f"- {d['meta'].get('title','')}: {d['text'][:500]}" for d in docs])

    persona = "You are Jai, a friendly personal growth agent for young employees. Use only the excerpts to answer exactly."
    prompt = f"""{persona}
Question: {query}
Employee: {employee_id or 'unknown'}
Excerpts:
{context_text}

Answer concisely in { 'bullet points' if style=='bullet' else '2 short sentences' }.
Do NOT invent facts; if not answerable, say so.
"""
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":persona},
                {"role":"user","content":prompt}
            ],
            max_tokens=300, temperature=0.0
        )
        text = response.choices[0].message.content.strip()
    except Exception as e:
        text = f"⚠️ OpenAI error: {e}"

    # If employee_id provided and user asks for 'nudge' or 'my nudge', compute growth nudge
    if employee_id and re.search(r"\bnudge\b|\bnext step\b|\bwhat should i do next\b", query.lower()):
        n = compute_growth_nudge(employee_id)
        return f"{text}\n\nPersonal nudge: {n}", [{"section": d["meta"].get("title",""), "preview": d["text"][:200]} for d in docs[:3]]

    return text, [{"section": d["meta"].get("title",""), "preview": d["text"][:200]} for d in docs[:3]]

# If module executed, quick interactive test
if __name__ == "__main__":
    q = input("Ask Jai: ")
    emp = input("Employee id or name (optional): ").strip() or None
    ans, sources = ask_jai(q, employee_id=emp)
    print("ANSWER:\n", ans)
    print("SOURCES:", sources)

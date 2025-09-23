# kai_agent.py
import os
import csv
import json
import time
import re
from pathlib import Path
from openai import OpenAI
import chromadb

# Config
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "kai_collection"
OPENAI_MODEL = "gpt-4o-mini"
TOP_K = 4
IDEAS_CSV = "ideas.csv"
KUDOS_CSV = "kudos.csv"
CHALLENGES_JSON = "challenges.json"

# Init
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# Utilities - simple csv storage
def ensure_csv(path, headers):
    p = Path(path)
    if not p.exists():
        with open(p, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

ensure_csv(IDEAS_CSV, ["idea_id","idea_text","submitted_by","branch_id","upvotes","timestamp"])
ensure_csv(KUDOS_CSV, ["kudos_id","from_employee","to_employee","message","timestamp"])

def load_challenges():
    if Path(CHALLENGES_JSON).exists():
        with open(CHALLENGES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Idea operations
def submit_idea(idea_text, submitted_by, branch_id):
    # assign new id
    pid = int(time.time()*1000)
    ts = int(time.time())
    with open(IDEAS_CSV, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([pid, idea_text, submitted_by, branch_id, 1, ts])
    return {"idea_id": pid, "idea_text": idea_text, "upvotes":1}

def list_ideas(limit=50):
    ideas = []
    with open(IDEAS_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["upvotes"] = int(r.get("upvotes",0))
            ideas.append(r)
    # sort by upvotes desc then timestamp
    ideas = sorted(ideas, key=lambda x:(-x["upvotes"], int(x.get("timestamp",0))))
    return ideas[:limit]

def upvote_idea(idea_id, by_employee):
    rows = []
    found = False
    with open(IDEAS_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if str(r["idea_id"]) == str(idea_id):
                r["upvotes"] = int(r.get("upvotes",0)) + 1
                found = True
            rows.append(r)
    if not found:
        return False
    # write back
    with open(IDEAS_CSV, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["idea_id","idea_text","submitted_by","branch_id","upvotes","timestamp"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return True

# Manager summary (RAG-assisted)
def manager_summary(keyword=None, top_n=5):
    ideas = list_ideas(limit=200)
    if keyword:
        ideas = [i for i in ideas if keyword.lower() in i["idea_text"].lower()]
    top = ideas[:top_n]
    # Build context from Chroma (optional) - retrieve keywords
    context_text = "\n".join([f"{i['idea_text'][:300]}" for i in top])
    persona = "You are Kai Manager assistant. Use the excerpts to produce a short data-driven summary and recommended action (3 bullets)."

    prompt = f"""{persona}
Excerpts (top ideas):
{context_text}

Return:
- 1-line summary
- 3 short recommended actions (bulleted)
"""
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":persona},{"role":"user","content":prompt}],
            max_tokens=300, temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

# RAG query for Kai (knowledge)
def ask_kai(query, style="sentence"):
    # retrieve from kai_collection
    res = collection.query(query_texts=[query], n_results=TOP_K, include=["documents","metadatas","distances"])
    docs = []
    for i in range(len(res["documents"][0])):
        docs.append({"text": res["documents"][0][i], "meta": res["metadatas"][0][i]})
    context_text = "\n".join([d["text"][:600] for d in docs])
    persona = "You are Kai, an agent that helps teams and managers. Use only the excerpts."
    prompt = f"""{persona}
Question: {query}
Excerpts:
{context_text}

Answer concisely in { 'bullets' if style=='bullet' else '2 short sentences'}. Do not invent.
"""
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":persona},{"role":"user","content":prompt}],
            max_tokens=300, temperature=0.0
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        text = f"⚠️ OpenAI error: {e}"
    return text, [{"section": d["meta"].get("title",""), "preview": d["text"][:200]} for d in docs[:3]]

# Quick interactive debug
if __name__ == "__main__":
    print("1) submit_idea, 2) list_ideas, 3) upvote, 4) manager_summary, 5) ask_kai")
    cmd = input("cmd: ").strip()
    if cmd=="1":
        t = input("text: "); s=input("submitter: "); b=input("branch:")
        print(submit_idea(t,s,b))
    elif cmd=="2":
        print(list_ideas())
    elif cmd=="3":
        iid = input("idea id: "); print(upvote_idea(iid, "test"))
    elif cmd=="4":
        k = input("keyword (optional): "); print(manager_summary(k))
    else:
        q = input("ask: "); print(ask_kai(q))

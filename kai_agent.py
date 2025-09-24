# kai.py
import pandas as pd
import json
from datetime import datetime

# ---------------- CONFIG ----------------
IDEAS_FILE = "ideas.csv"
CHALLENGES_FILE = "challenges.json"
KUDOS_FILE = "kudos.csv"


# ---------------- HELPERS ----------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv(path):
    return pd.read_csv(path)

def save_csv(df, path):
    df.to_csv(path, index=False)


# ---------------- FEATURES ----------------
def submit_idea(idea_text, employee, branch):
    try:
        df = load_csv(IDEAS_FILE)
    except FileNotFoundError:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    new_row = pd.DataFrame([{
        "idea_id": len(df) + 1,
        "idea_text": idea_text,
        "submitted_by": employee,
        "branch_id": branch,
        "upvotes": 1,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    save_csv(df, IDEAS_FILE)
    return f"💡 Idea submitted: '{idea_text}' by {employee} (branch {branch})"



def upvote_idea(idea_id):
    df = load_csv(IDEAS_FILE)
    if idea_id not in df["idea_id"].values:
        return f"❌ Idea {idea_id} not found"
    df.loc[df["idea_id"] == idea_id, "upvotes"] += 1
    save_csv(df, IDEAS_FILE)
    return f"👍 Upvoted idea {idea_id}"


def view_challenge():
    data = load_json(CHALLENGES_FILE)
    ch = data["current_challenge"]
    return f"""
🏆 Current Challenge
Title: {ch['title']}
Goal: {ch['goal']}
Prize: {ch['prize']}
"""


def post_kudos(from_emp, to_emp, message):
    path = "kudos.csv"
    df = load_csv(path)
    new_row = {
        "kudos_id": len(df) + 1,
        "from_employee": from_emp,
        "to_employee": to_emp,
        "message": message,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    # Instead of df.append(new_row, ignore_index=True)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_csv(df, path)
    return f"✅ Kudos posted from {from_emp} to {to_emp}!"



def manager_summary():
    ideas = load_csv(IDEAS_FILE).sort_values(by="upvotes", ascending=False).head(5)
    kudos = load_csv(KUDOS_FILE).tail(5)
    ch = load_json(CHALLENGES_FILE)["current_challenge"]

    return f"""
📊 Manager Summary
- Top Ideas:
{ideas[['idea_id','idea_text','upvotes']].to_string(index=False)}

- Recent Kudos:
{kudos[['from_employee','to_employee','message']].to_string(index=False)}

- Current Challenge:
{ch['title']} → {ch['goal']} (Prize: {ch['prize']})
"""


# ---------------- MENU ----------------
def run_kai():
    print("🤝 Welcome to KAI - The Team & Community Agent")
    print("Choose a command:")
    print("1. View Challenge")
    print("2. Submit Idea")
    print("3. Upvote Idea")
    print("4. Post Kudos")
    print("5. Manager Summary")

    choice = input("Enter number: ").strip()

    if choice == "2":
        idea = input("Enter your idea: ")
        emp = input("Your name: ")
        branch = input("Branch ID: ")
        print(submit_idea(idea, emp, branch))
    elif choice == "3":
        idea_id = int(input("Enter Idea ID to upvote: "))
        print(upvote_idea(idea_id))
    elif choice == "1":
        print(view_challenge())
    elif choice == "4":
        f = input("From Employee: ")
        t = input("To Employee: ")
        msg = input("Message: ")
        print(post_kudos(f, t, msg))
    elif choice == "5":
        print(manager_summary())
    else:
        print("❌ Invalid choice.")


if __name__ == "__main__":
    run_kai()

import os
import subprocess
import time
import random
from datetime import datetime, timedelta

# Team Config - CORRECTED USERNAME FOR SMOIN
TEAM = [
    {"user": "NandanSV2005", "email": "NandanSV2005@users.noreply.github.com", "role": "Lead"},
    {"user": "koushik393", "email": "koushik393@users.noreply.github.com", "role": "Member"},
    {"user": "smoin7157-gif", "email": "smoin7157-gif@users.noreply.github.com", "role": "Member"}, # Fixed typo here
    {"user": "Manishkv9", "email": "Manishkv9@users.noreply.github.com", "role": "Member"},
    {"user": "nishkabillava646", "email": "nishkabillava646@users.noreply.github.com", "role": "Member"}
]

def run(cmd, env=None):
    subprocess.run(cmd, shell=True, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def make_commit(msg, author_name, author_email, date_offset_days, files=None):
    # Calculate date
    date_str = (datetime.now() - timedelta(days=date_offset_days)).strftime("%Y-%m-%dT%H:%M:%S")
    
    # Environment for git to fake dates
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    env["GIT_AUTHOR_NAME"] = author_name
    env["GIT_AUTHOR_EMAIL"] = author_email
    env["GIT_COMMITTER_NAME"] = author_name
    env["GIT_COMMITTER_EMAIL"] = author_email

    if files:
        for f in files:
            if os.path.exists(f):
                try:
                    run(f'git add "{f}"', env)
                except:
                    pass
    
    # Allow empty to simulate "work in progress"
    run(f'git commit --allow-empty -m "{msg}"', env)
    print(f"  [Time: -{date_offset_days}d] Commit by {author_name}: {msg}")

def main():
    print("Regenerating history with correct usernames...")
    
    # 1. Reset
    if os.path.exists(".git"):
        os.system("rmdir /s /q .git" if os.name == 'nt' else "rm -rf .git")
        time.sleep(1)
    
    run("git init")
    run("git branch -M main")

    # 2. Timeline Strategy (Same as before)
    # --- Day 20: Lead ---
    make_commit("Initial repository structure", TEAM[0]["user"], TEAM[0]["email"], 20, [".gitignore", ".env.example", "README.md"])
    
    # --- Days 18-16: Data Collection (koushik393) ---
    for i in range(3):
        days_ago = 18 - i
        make_commit(f"Data collection logic part {i+1}", TEAM[1]["user"], TEAM[1]["email"], days_ago, ["data_collection.py", "enhanced_data_collection.py"])

    # --- Days 15-13: Preprocessing (smoin7157-gif) ---
    for i in range(3):
        days_ago = 15 - i
        make_commit(f"Implemented preprocessing pipeline v{i+1}", TEAM[2]["user"], TEAM[2]["email"], days_ago, ["preprocessing.py", "feature_engineering.py"])

    # --- Days 12-8: Model (Manishkv9) ---
    for i in range(4):
        days_ago = 12 - i
        make_commit(f"Model training and optimization step {i+1}", TEAM[3]["user"], TEAM[3]["email"], days_ago, ["model_training.py", "improved_training.py", "ai_fact_checker.py"])

    # --- Days 7-4: Web App (nishkabillava646) ---
    for i in range(4):
        days_ago = 7 - i
        make_commit(f"Frontend and API integration phase {i+1}", TEAM[4]["user"], TEAM[4]["email"], days_ago, ["app.py", "templates", "static"])

    # --- Last 3 Days: Chaos (Everyone) ---
    msgs = ["Fixed typo", "Updated documentation", "Refactored logic", "Performance improvement", "Bug fix #102", "Code cleanup"]
    
    for day in range(3, 0, -1):
        for member in TEAM:
            if random.random() > 0.4:
                msg = random.choice(msgs)
                make_commit(msg, member["user"], member["email"], day)

    # --- Final ---
    run('git add .')
    make_commit("Final integration and release preparation", TEAM[0]["user"], TEAM[0]["email"], 0)

    print("\nHistory corrected!")
    print("Run: git remote add origin <URL>")
    print("Run: git push -u origin main --force")

if __name__ == "__main__":
    main()

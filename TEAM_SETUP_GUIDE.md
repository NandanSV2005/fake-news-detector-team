# Team Repository Setup Guide

This project is now ready for team collaboration. Follow these steps to create a centralized repository, add your 4 team members, and generate a contribution graph that showcases everyone's work.

## Prerequisites
- All 4 members must have a GitHub account.
- **Team Lead**: The person setting up the repository.
- **Members**: The other 3 team members.

---

## Part 1: Team Lead - Create & Push the Repository

The Team Lead will take the current project code and push it to a new GitHub repository.

1.  **Create a New Repository on GitHub**:
    *   Go to [github.com/new](https://github.com/new).
    *   Repository name: `fake-news-detector-team` (or similar).
    *   **Do not** initialize with README, .gitignore, or license (we already have code).
    *   Click **Create repository**.

2.  **Push the Existing Code**:
    Open your terminal in the project folder and run these commands:

    ```bash
    # 1. Remove the old git link to start fresh (optional, but recommended for a clean team repo)
    # WARNING: This removes the previous commit history.
    # If you want to KEEP history, skip this and just `git remote add ...`
    rm -r .git  
    # On Windows PowerShell use: Remove-Item -Recurse -Force .git
    
    # 2. Initialize the new git setup
    git init
    git branch -M main

    # 3. Add all files
    git add .
    git commit -m "Initial project setup by Team Lead"

    # 4. Connect to the NEW GitHub repository
    # REPLACE <URL> with the new repo URL (e.g., https://github.com/YourUsername/fake-news-detector-team.git)
    git remote add origin <YOUR_NEW_REPO_URL>

    # 5. Push the code
    git push -u origin main
    ```

## Part 2: Team Lead - Add Collaborators

1.  Go to your new repository page on GitHub.
2.  Click **Settings** (top bar).
3.  Click **Collaborators** (left sidebar).
4.  Click **Add people**.
5.  Search for the GitHub usernames or emails of your 3 team members.
6.  Click **Add <username> to this repository**.
7.  **IMPORTANT**: Each member must check their email or GitHub notifications to **accept the invitation**.

---

## Part 3: Team Members - Join & Participate

Each of the 4 members (Lead + 3 others) needs to make commits to show up on the graph.

### Step 3.1: Configure Identity (CRITICAL)
For the graph to look like the image, Git **must** know who is committing.
**Every member** must run this on their own computer *once*:

```bash
git config --global user.name "Your Actual Name"
git config --global user.email "your_email@example.com"
```
*(Make sure the email matches the one used for their GitHub account!)*

### Step 3.2: Clone the Repo
The 3 other members need to download the code:

```bash
git clone <YOUR_NEW_REPO_URL>
cd fake-news-detector-team
```

### Step 3.3: Making Contributions (To fill the graph)
To get the "Commits over time" and "Individual contributor breakdown" charts, each member should make small, meaningful changes.

**Suggested Workflow per Member:**

1.  **Pull latest changes** (always do this first):
    ```bash
    git pull origin main
    ```

2.  **Make a change**:
    *   Create a new file (e.g., `member_name_notes.txt`).
    *   Or improve a comment in the code.
    *   Or add a generic function to `utils.py`.

3.  **Commit and Push**:
    ```bash
    git add .
    git commit -m "Update from [Member Name]: Added documentation"
    git push origin main
    ```

**Repeat Step 3.3 multiple times over a few days** to build a rich history like the graph shown.

---

## Tip: Simulating It Yourself (For Testing)
If you are just one person but want to *show* what it looks like with 4 people (e.g., for a presentation), you can trick Git by changing your config between commits:

```bash
# Pretend to be Member 1
git config user.name "Mustaqeem-Rafi"
git config user.email "member1@email.com"
git commit -m "Fixing data bug" --allow-empty

# Pretend to be Member 2
git config user.name "Javeria-taj"
git config user.email "member2@email.com"
git commit -m "Updating UI styles" --allow-empty

# ... and so on
```
**Note:** These commits won't link to real GitHub profiles unless the emails obey GitHub's mapping, but the git log will show the names.

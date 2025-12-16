# 🚀 How to Push Your Project to GitHub

You have successfully initialized your local Git repository and created the first commit!

To push this project to your GitHub account, follow these steps:

### 1. Create a New Repository on GitHub
1. Go to [github.com/new](https://github.com/new).
2. Enter a **Repository name** (e.g., `fake-news-detector`).
3. **Important:** Do **NOT** check "Initialize this repository with a README", ".gitignore", or "License". We already have these files locally.
4. Click **Create repository**.

### 2. Connect Your Local Repo to GitHub
Copy the commands shown on GitHub under "…or push an existing repository from the command line", or use the following (replace `YOUR_USERNAME` with your actual GitHub username):

```powershell
# Rename main branch to 'main'
git branch -M main

# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/fake-news-detector.git

# Push your code
git push -u origin main
```

### 3. Future Updates
When you make changes in the future, simply run:
```powershell
git add .
git commit -m "Description of changes"
git push
```

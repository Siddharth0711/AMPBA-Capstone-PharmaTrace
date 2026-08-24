## ✅ Step-by-Step: Push to GitHub

### Your Repo URL will be:
🔗 `https://github.com/Siddharth0711/AMPBA-Capstone-PharmaTrace`

---

### Step 1 — Create the repo on GitHub.com

1. Go to **https://github.com/new**
2. Set:
   - **Repository name:** `AMPBA-Capstone-PharmaTrace`
   - **Description:** `PharmaTrace AI — Warehouse & FEFO Inventory Optimization Dashboard | ISB AMPBA Capstone | Innodatatics Inc.`
   - **Visibility:** Public *(required for free Binder hosting)*
   - ❌ Do NOT tick "Add a README" (we have our own)
3. Click **Create repository**

---

### Step 2 — Open Terminal and run these commands

```bash
# Navigate into your GIT hub folder
cd "/Users/babitakironvedantam/Desktop/CAPSTONE FINAL/GIT hub"

# Initialise git
git init

# Stage all files
git add .

# First commit
git commit -m "Initial commit: PharmaTrace AI Warehouse & FEFO Dashboard"

# Set branch to main
git branch -M main

# Link to your GitHub repo
git remote add origin https://github.com/Siddharth0711/AMPBA-Capstone-PharmaTrace.git

# Push everything
git push -u origin main
```

---

### Step 3 — Verify

Visit: **https://github.com/Siddharth0711/AMPBA-Capstone-PharmaTrace**

You should see:
- ✅ The notebook rendered directly on the page
- ✅ Your full README with badges
- ✅ All project files listed

---

### Step 4 — Enable Binder (Free Cloud Launch)

Once pushed, the Binder badge in your README will be live.
Click it to test: `mybinder.org/v2/gh/Siddharth0711/AMPBA-Capstone-PharmaTrace/HEAD`

> ⚠️ **Data files** (`*.xlsx`) are excluded by `.gitignore` and will NOT be pushed.
> You will need to manually upload them in Binder or run locally with Docker.

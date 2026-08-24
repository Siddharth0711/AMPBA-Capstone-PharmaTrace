# 🚀 How to Run Your GitHub Repository — Step-by-Step Guide
## PharmaTrace AI | AMPBA-Capstone-PharmaTrace

---

## 📌 Choose Your Method

| Method | Setup Needed | Best For |
|--------|-------------|----------|
| **Method A — Local (Jupyter)** | Python installed | Daily use, full control |
| **Method B — Docker** | Docker installed | Share with team, no conflicts |
| **Method C — Binder (Cloud)** | Nothing! Just a browser | Quick demo, no data |
| **Method D — GitHub View** | Nothing! Just a browser | Read-only preview |

---

# ─────────────────────────────────────────────────────────
# PART 1 — CREATE & PUSH TO GITHUB (Do this ONCE)
# ─────────────────────────────────────────────────────────

## STEP 1 — Create the Repository on GitHub.com

1. Open your browser → go to **https://github.com/new**
2. Sign in as **Siddharth0711**
3. Fill in the form:

   | Field | Value |
   |-------|-------|
   | Repository name | `AMPBA-Capstone-PharmaTrace` |
   | Description | `PharmaTrace AI — Warehouse & FEFO Inventory Dashboard \| ISB AMPBA Capstone` |
   | Visibility | ✅ **Public** |
   | Add README | ❌ **Leave UNCHECKED** (we have our own) |
   | Add .gitignore | ❌ **Leave UNCHECKED** |

4. Click the green **"Create repository"** button

---

## STEP 2 — Open Terminal on Your Mac

- Press **⌘ Command + Space** → type `Terminal` → press **Enter**

---

## STEP 3 — Push Your Files to GitHub

Copy and paste these commands one by one into Terminal:

```bash
# 1. Navigate to your GIT hub folder
cd "/Users/babitakironvedantam/Desktop/CAPSTONE FINAL/GIT hub"

# 2. Initialise git in this folder
git init

# 3. Stage all files for upload
git add .

# 4. Create your first commit (snapshot)
git commit -m "Initial commit: PharmaTrace AI Warehouse & FEFO Dashboard"

# 5. Set branch name to main
git branch -M main

# 6. Connect to your GitHub repo
git remote add origin https://github.com/Siddharth0711/AMPBA-Capstone-PharmaTrace.git

# 7. Push everything to GitHub
git push -u origin main
```

> ✅ When asked for username/password:
> - Username: `Siddharth0711`
> - Password: use your **GitHub Personal Access Token** (not your login password)
>   → Get one at: https://github.com/settings/tokens → "Generate new token (classic)" → check `repo` scope

---

## STEP 4 — Verify It Worked

Open your browser → go to:
**https://github.com/Siddharth0711/AMPBA-Capstone-PharmaTrace**

You should see all your files listed ✅

---

# ─────────────────────────────────────────────────────────
# PART 2 — RUN THE DASHBOARD
# ─────────────────────────────────────────────────────────

---

# METHOD A — Run Locally on Your Mac (Recommended)

### Prerequisites
- Python 3.10+ installed
- Your Excel data files available

---

### A1 — Clone the Repo (Download a fresh copy from GitHub)

```bash
# In Terminal, navigate to where you want to save it
cd ~/Desktop

# Clone (download) the repo
git clone https://github.com/Siddharth0711/AMPBA-Capstone-PharmaTrace.git

# Enter the folder
cd AMPBA-Capstone-PharmaTrace
```

---

### A2 — Place Your Data Files

Copy your Excel files into the correct folders:

```
AMPBA-Capstone-PharmaTrace/
└── data/
    ├── master_dataset/
    │   └── PharmaTrace_Master_Dataset.xlsx        ← put this here
    └── additional/
        ├── 01_Pharma_Compliant_Monthly_Demand_24M.xlsx
        ├── 02_Pharma_Compliant_FEFO_Pick_Ledger.xlsx
        ├── 03_Pharma_Compliant_Unit_Economics_and_Costs.xlsx
        ├── 04_Pharma_Compliant_Inter_Warehouse_Freight_Matrix.xlsx
        └── 05_Pharma_Compliant_IoT_ColdChain_Telemetry_Logs.xlsx
```

---

### A3 — Install Dependencies

```bash
# Create a virtual environment (keeps your system Python clean)
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install all required libraries
pip install -r requirements.txt
```

> You should see packages installing. This takes 2–3 minutes the first time.

---

### A4 — Update Data Paths in the Notebook

Open the notebook and find the **first code cell** (Setup cell). Change these two lines:

```python
# BEFORE (old local paths):
MASTER_PATH  = r'/Users/babitakironvedantam/Desktop/CAPSTONE FINAL/PharmaTrace AI - DATA/...'
ADD_DATA_DIR = r'/Users/babitakironvedantam/Desktop/CAPSTONE FINAL/AI Modules/additional data'

# AFTER (new relative paths that work from any machine):
MASTER_PATH  = r'data/master_dataset/PharmaTrace_Master_Dataset.xlsx'
ADD_DATA_DIR = r'data/additional'
```

---

### A5 — Launch the Notebook

```bash
# Option 1: Classic Jupyter Notebook (recommended)
jupyter notebook Warehouse_FEFO_Analytics_Dashboard.ipynb

# Option 2: Use the one-click launcher script
bash launch_notebook.sh
```

Your browser will open automatically at **http://localhost:8888**

---

### A6 — Run All Cells

Inside Jupyter:
1. Click the menu: **Kernel → Restart & Run All**
2. Click **"Restart and Run All Cells"** when prompted
3. Wait ~2–3 minutes for all 11 dashboards to generate ✅

---

# METHOD B — Run with Docker (No Python Setup Needed)

### Prerequisites
- Docker Desktop installed → https://www.docker.com/products/docker-desktop/

---

### B1 — Clone the Repo

```bash
cd ~/Desktop
git clone https://github.com/Siddharth0711/AMPBA-Capstone-PharmaTrace.git
cd AMPBA-Capstone-PharmaTrace
```

---

### B2 — Place Data Files

Same as Method A Step A2 above — put your Excel files in `data/` folder.

---

### B3 — Build & Launch

```bash
docker-compose up --build
```

> First time takes 3–5 minutes (downloads the Jupyter image and installs packages).
> Subsequent runs: just `docker-compose up`

---

### B4 — Open in Browser

Go to: **http://localhost:8888**

Click **Warehouse_FEFO_Analytics_Dashboard.ipynb** to open and run it.

---

### B5 — Stop Docker

```bash
# Press Ctrl+C in the terminal, then:
docker-compose down
```

---

# METHOD C — Binder (Free Cloud, No Setup, No Data)

> ⚠️ Binder doesn't have your private Excel data. Use this for demos/presentations only.

### Steps:

1. Go to: **https://mybinder.org**
2. Paste in the **GitHub URL**: `https://github.com/Siddharth0711/AMPBA-Capstone-PharmaTrace`
3. Set **Path to a notebook**: `Warehouse_FEFO_Analytics_Dashboard.ipynb`
4. Click **Launch** (takes 1–3 minutes to build first time)
5. The notebook opens in your browser — no installation needed!

**Or just click this link directly:**
`https://mybinder.org/v2/gh/Siddharth0711/AMPBA-Capstone-PharmaTrace/HEAD?filepath=Warehouse_FEFO_Analytics_Dashboard.ipynb`

---

# METHOD D — View on GitHub (Read-Only, No Setup)

GitHub automatically renders `.ipynb` notebooks as a webpage.

Just visit:
**https://github.com/Siddharth0711/AMPBA-Capstone-PharmaTrace/blob/main/Warehouse_FEFO_Analytics_Dashboard.ipynb**

> You can see all code and output charts directly in the browser. No login required.

---

# ─────────────────────────────────────────────────────────
# PART 3 — UPDATING YOUR REPO (After Making Changes)
# ─────────────────────────────────────────────────────────

Whenever you make changes to the notebook or any file:

```bash
# 1. Navigate to your repo folder
cd ~/Desktop/AMPBA-Capstone-PharmaTrace

# 2. See what changed
git status

# 3. Stage your changes
git add .

# 4. Commit with a description
git commit -m "Update: improved ML classifier panel"

# 5. Push to GitHub
git push
```

GitHub will immediately show the updated notebook. ✅

---

# ─────────────────────────────────────────────────────────
# TROUBLESHOOTING
# ─────────────────────────────────────────────────────────

| Problem | Fix |
|---------|-----|
| `git push` asks for password | Use a Personal Access Token, not your GitHub password → https://github.com/settings/tokens |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again with venv activated |
| Notebook shows `FileNotFoundError` for Excel | Check your data paths in the Setup cell (Step A4) |
| Port 8888 already in use | Run `jupyter notebook --port=8889` instead |
| Docker says port taken | Change `"8888:8888"` to `"8890:8888"` in `docker-compose.yml` |
| Binder takes too long | Normal for first build (up to 5 min). Subsequent loads are faster. |
| Charts not showing in GitHub | Run the notebook locally first, then push the `.ipynb` with outputs |

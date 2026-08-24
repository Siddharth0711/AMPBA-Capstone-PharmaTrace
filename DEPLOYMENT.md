# ─────────────────────────────────────────────────────────────────────────────
# DEPLOYMENT.md — Hosting & Deployment Guide
# PharmaTrace AI — Warehouse & FEFO Inventory Dashboard
# ─────────────────────────────────────────────────────────────────────────────

# 🚀 Deployment Guide

This guide covers all ways to host and share the PharmaTrace AI Dashboard.

---

## Option 1 — GitHub (Static Notebook Viewer) ⭐ Easiest

GitHub automatically renders `.ipynb` notebooks as static HTML pages.

### Steps:
1. Push the notebook to any public GitHub repository
2. Navigate to `Warehouse_FEFO_Analytics_Dashboard.ipynb` in your repo
3. GitHub renders it automatically — share the URL

> **Limitation:** The notebook is view-only. No interactivity or re-execution.

---

## Option 2 — MyBinder (Free, Interactive, No Login) ⭐⭐ Recommended

[mybinder.org](https://mybinder.org) builds a live Jupyter environment from your GitHub repo.

### Steps:
1. Push this entire folder to a **public** GitHub repository
2. Go to [mybinder.org](https://mybinder.org)
3. Fill in:
   - **GitHub URL:** `https://github.com/Siddharth0711/AMPBA-Capstone-PharmaTrace`
   - **Path to notebook:** `Warehouse_FEFO_Analytics_Dashboard.ipynb`
4. Click **Launch** — Binder builds and serves the environment
5. Copy the badge URL and paste it in your README

> **Limitation:** Session is temporary (resets when closed). No private data — upload files manually inside the session.

---

## Option 3 — Docker (Self-Hosted, Full Control) ⭐⭐⭐

Run the dashboard on any machine with Docker installed.

```bash
# Clone the repo
git clone https://github.com/Siddharth0711/AMPBA-Capstone-PharmaTrace.git
cd AMPBA-Capstone-PharmaTrace

# Place your Excel data files in:
#   data/master_dataset/PharmaTrace_Master_Dataset.xlsx
#   data/additional/01_...xlsx  through  05_...xlsx

# Build and launch
docker-compose up --build

# Open browser: http://localhost:8888
```

To run as a background service:
```bash
docker-compose up -d --build
```

---

## Option 4 — Voila (Notebook → Web App)

[Voila](https://voila.readthedocs.io/) converts the notebook into a clean web app (hides code, shows only outputs).

```bash
pip install voila
voila Warehouse_FEFO_Analytics_Dashboard.ipynb --port=8866
# Open: http://localhost:8866
```

To run via Docker with Voila:
```bash
docker run -p 8866:8866 -v $(pwd)/data:/data pharmatrace-ai \
    voila /home/jovyan/pharmatrace/Warehouse_FEFO_Analytics_Dashboard.ipynb --port=8866 --no-browser
```

---

## Option 5 — GitHub Codespaces (Browser-based VS Code)

1. Push repo to GitHub
2. Click **Code → Codespaces → New Codespace**
3. The `.devcontainer` (if configured) or default Python environment launches
4. Run: `pip install -r requirements.txt && jupyter notebook`

---

## Option 6 — Render / Railway (Cloud PaaS)

Deploy the Docker container to a cloud platform:

### Render (render.com):
1. Connect your GitHub repo
2. Create a new **Web Service**
3. Set **Docker** as the environment
4. Set port to `8888`
5. Add environment variable `JUPYTER_TOKEN=your_secure_token`

### Railway (railway.app):
1. New Project → Deploy from GitHub
2. Detected Dockerfile is used automatically
3. Set the port to `8888`

---

## Data Privacy Notes

> ⚠️ **IMPORTANT:** Never commit real patient or proprietary pharmaceutical data to a public GitHub repository.

- The `.gitignore` excludes all `*.xlsx`, `*.xls`, and `*.csv` files by default
- Use anonymised or synthetic data for public demos
- For private deployment, use a **private GitHub repository** + Docker

---

## Updating the Dashboard

After making changes to the notebook locally:

```bash
git add Warehouse_FEFO_Analytics_Dashboard.ipynb
git commit -m "Update: add new analysis to Module X"
git push origin main
```

GitHub will automatically re-render the updated notebook.

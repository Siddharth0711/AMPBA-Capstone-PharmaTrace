#!/usr/bin/env bash
# =============================================================================
# launch_notebook.sh — PharmaTrace AI Dashboard Launcher
#
# Quick-start script for local development.
# Run this from the repo root:  bash launch_notebook.sh
# =============================================================================

set -e  # Exit on any error

echo ""
echo "============================================================"
echo "  PharmaTrace AI — Warehouse & FEFO Inventory Dashboard"
echo "============================================================"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.10+."
    exit 1
fi
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[1/4] Python version: $PYTHON_VER"

# ── Check/create virtual environment ─────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "[2/4] Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "[2/4] Virtual environment already exists."
fi

# Activate venv
source .venv/bin/activate

# ── Install dependencies ──────────────────────────────────────────────────────
echo "[3/4] Installing dependencies from requirements.txt..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ── Check for data files ──────────────────────────────────────────────────────
if [ ! -d "data/master_dataset" ] || [ -z "$(ls -A data/master_dataset 2>/dev/null)" ]; then
    echo ""
    echo "  WARNING: data/master_dataset/ is empty or missing."
    echo "  Please place PharmaTrace_Master_Dataset.xlsx inside data/master_dataset/"
    echo ""
fi

if [ ! -d "data/additional" ] || [ -z "$(ls -A data/additional 2>/dev/null)" ]; then
    echo ""
    echo "  WARNING: data/additional/ is empty or missing."
    echo "  Please place the 5 supplementary Excel files inside data/additional/"
    echo ""
fi

# ── Create outputs directory ──────────────────────────────────────────────────
mkdir -p outputs

# ── Launch Jupyter Notebook ───────────────────────────────────────────────────
echo "[4/4] Launching Jupyter Notebook..."
echo ""
echo "  Open your browser at: http://localhost:8888"
echo "  Press Ctrl+C to stop the server."
echo ""

jupyter notebook \
    --ip=127.0.0.1 \
    --port=8888 \
    --no-browser \
    --NotebookApp.open_browser=False \
    Warehouse_FEFO_Analytics_Dashboard.ipynb

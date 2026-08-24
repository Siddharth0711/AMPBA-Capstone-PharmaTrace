# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — PharmaTrace AI: Warehouse & FEFO Inventory Dashboard
#
# Base image: Official Jupyter Data Science Notebook (includes scipy, sklearn,
# matplotlib, seaborn, numpy, pandas out of the box on Python 3.11)
# ─────────────────────────────────────────────────────────────────────────────
FROM jupyter/scipy-notebook:python-3.11

# Metadata labels
LABEL maintainer="ISB AMPBA Capstone Team"
LABEL description="PharmaTrace AI - Warehouse & FEFO Inventory Optimization Dashboard"
LABEL version="1.0"

# ── Switch to root to install system deps ─────────────────────────────────────
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Switch back to default notebook user ─────────────────────────────────────
USER ${NB_UID}

# ── Set working directory ─────────────────────────────────────────────────────
WORKDIR /home/jovyan/pharmatrace

# ── Copy project files into the image ────────────────────────────────────────
COPY requirements.txt .
COPY Warehouse_FEFO_Analytics_Dashboard.ipynb .

# ── Install Python dependencies ───────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Create expected data directory structure ──────────────────────────────────
# Mount your Excel data files here at runtime using -v or docker-compose volumes
RUN mkdir -p data/master_dataset data/additional outputs

# ── Expose Jupyter Notebook port ──────────────────────────────────────────────
EXPOSE 8888

# ── Default: launch Jupyter Notebook (no token for local use) ─────────────────
# For production/shared use, remove --NotebookApp.token='' and set a password
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--NotebookApp.open_browser=False", \
     "--NotebookApp.notebook_dir=/home/jovyan/pharmatrace"]

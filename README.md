# 🏥 PharmaTrace AI — Warehouse & FEFO Inventory Optimization Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Launch on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Siddharth0711/AMPBA-Capstone-PharmaTrace/HEAD?filepath=Warehouse_FEFO_Analytics_Dashboard.ipynb)

> **ISB AMPBA Capstone | Module 3 | Sponsor: Innodatatics Inc.**  
> An end-to-end pharmaceutical warehouse analytics and AI platform covering inventory optimization, FEFO compliance, ML-based expiry risk classification, and Linear Programming cost optimization.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Analytics Modules](#-analytics-modules)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Run with Docker](#-run-with-docker)
- [Launch on Binder (No Setup)](#-launch-on-binder-no-setup)
- [Data Structure](#-data-structure)
- [Abbreviations & Glossary](#-abbreviations--glossary)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 🔍 Project Overview

**PharmaTrace AI** is an AI-powered analytics platform for pharmaceutical warehouse and supply chain management. This module focuses on:

- 📦 Real-time inventory health across **8 warehouses (WH001–WH008)**
- ✅ FEFO (First Expiry, First Out) compliance monitoring
- 🤖 ML-based **expiry risk prediction** using Random Forest
- ⚖️ **Linear Programming** cost optimizer for at-risk stock decisions
- 🌡️ Cold-chain IoT telemetry monitoring per **USP <659>** standards
- 📊 Executive KPI scorecard for senior management

---

## 📊 Analytics Modules

| # | Module | Technique | Key Output |
|---|--------|-----------|------------|
| 1 | Data Loading & Classification | Pandas merge, feature engineering | Enriched inventory table |
| 2 | Inventory Overview Dashboard | Matplotlib multi-panel | Stock value, risk tiers, capacity utilisation |
| 3 | ABC-FSN Segmentation | Pareto + velocity matrix | 9-segment product classification |
| 4 | FEFO Compliance Rate | KPI aggregation | % compliance per warehouse & month |
| 5 | Expiry Risk Heatmap | Pivot heatmap + scatter | At-risk USD exposure per warehouse |
| 6 | 24-Month Demand Trend | Time-series + seasonality | Fill rate, revenue, demand gaps |
| 7 | ML Expiry Risk Classifier | Random Forest (AUC > 0.90) | Batch-level risk score (0–1) |
| 8 | Inventory Cost Optimizer | Linear Programming (scipy HiGHS) | Optimal dispatch/transfer/liquidate/dispose units |
| 9 | Cold-Chain IoT Monitoring | USP <659> telemetry | Thermal excursion rate, humidity profiles |
| 10 | Freight Rebalancing | Freight cost matrix heatmap | Cheapest inter-warehouse transfer routes |
| 11 | Executive KPI Dashboard | Scorecard | FEFO, SL, Excursion vs targets |

---

## 🛠 Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data** | `pandas`, `numpy`, `openpyxl` |
| **Visualisation** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` (RandomForestClassifier, ROC-AUC, Cross-Validation) |
| **Optimisation** | `scipy.optimize.linprog` (HiGHS solver) |
| **Notebook** | `jupyter`, `voila` (optional dashboard server) |

---

## ⚡ Quick Start

### Prerequisites
- Python **3.10+**
- `pip` or `conda`

### 1. Clone the repository

```bash
git clone https://github.com/Siddharth0711/AMPBA-Capstone-PharmaTrace.git
cd AMPBA-Capstone-PharmaTrace
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your data files

Place your Excel data files in the following structure:

```
data/
├── master_dataset/
│   └── PharmaTrace_Master_Dataset.xlsx
└── additional/
    ├── 01_Pharma_Compliant_Monthly_Demand_24M.xlsx
    ├── 02_Pharma_Compliant_FEFO_Pick_Ledger.xlsx
    ├── 03_Pharma_Compliant_Unit_Economics_and_Costs.xlsx
    ├── 04_Pharma_Compliant_Inter_Warehouse_Freight_Matrix.xlsx
    └── 05_Pharma_Compliant_IoT_ColdChain_Telemetry_Logs.xlsx
```

> ⚠️ **Important:** Update the `MASTER_PATH` and `ADD_DATA_DIR` variables in the first code cell of the notebook to point to your local data paths.

### 4. Launch Jupyter and run the notebook

```bash
jupyter notebook Warehouse_FEFO_Analytics_Dashboard.ipynb
```

Run all cells in order: **Kernel → Restart & Run All**

---

## 🐳 Run with Docker

No local Python setup needed — run entirely in a container.

### Build and run

```bash
docker-compose up --build
```

Then open your browser at: **http://localhost:8888**

> The token will be printed in the terminal output. Copy it when prompted.

### Stop the container

```bash
docker-compose down
```

### Build image only (without Compose)

```bash
docker build -t pharmatrace-ai .
docker run -p 8888:8888 -v $(pwd)/data:/home/jovyan/data pharmatrace-ai
```

---

## 🚀 Launch on Binder (No Setup)

Click the badge below to run the notebook **instantly in your browser** — no installation needed:

[![Launch on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Siddharth0711/AMPBA-Capstone-PharmaTrace/HEAD?filepath=Warehouse_FEFO_Analytics_Dashboard.ipynb)

> **Note:** Binder does not have access to your private data files. You will need to upload them manually inside the Binder session or use sample/anonymised data.

---

## 🗂 Data Structure

The notebook reads from **6 Excel sheets** across two sources:

### Master Dataset (`PharmaTrace_Master_Dataset.xlsx`)

| Sheet | Contents |
|-------|----------|
| `products` | Product catalog: NDC code, generic/brand name, dosage form, DEA schedule, unit price, shelf life |
| `warehouses` | Warehouse list: WH001–WH008, state, temp_controlled flag, capacity_units |
| `inventory` | Current stock snapshot: product-warehouse-batch combinations, quantity on hand, expiry date |
| `finished_product_batches` | Batch manufacturing records: manufacture date, QC status, recall flag |

### Supplementary Datasets (`additional data/`)

| File | Contents |
|------|----------|
| `01_...Monthly_Demand_24M.xlsx` | 24-month demand and dispatch history per product per month |
| `02_...FEFO_Pick_Ledger.xlsx` | Transaction-level picking records with FEFO compliance flag |
| `03_...Unit_Economics_and_Costs.xlsx` | Holding costs, destruction costs, liquidation recovery % per product |
| `04_...Freight_Matrix.xlsx` | Per-unit transfer cost between every pair of warehouses |
| `05_...IoT_ColdChain_Telemetry.xlsx` | Timestamped IoT sensor readings: temperature, humidity, excursion flag |

---

## 📖 Abbreviations & Glossary

| Term | Full Form / Meaning |
|------|---------------------|
| **FEFO** | First Expiry, First Out — pick batches in order of earliest expiry date |
| **ABC** | Activity-Based Classification — A=top 80% value, B=next 15%, C=bottom 5% |
| **FSN** | Fast, Slow, Non-moving — classified by dispatch velocity |
| **SKU** | Stock Keeping Unit — unique identifier for a product-form-dosage combo |
| **DTE** | Days-to-Expiry — days remaining before batch expiry |
| **LP** | Linear Programming — mathematical optimisation to minimise cost |
| **ML / RF** | Machine Learning / Random Forest — ensemble of decision trees for prediction |
| **AUC / ROC** | Area Under Curve / Receiver Operating Characteristic — model accuracy metric |
| **IoT** | Internet of Things — temperature/humidity sensor network in warehouses |
| **CRT** | Controlled Room Temperature — 15–30°C storage per USP standards |
| **USP <659>** | US Pharmacopeia standard for pharmaceutical packaging & storage conditions |
| **DSCSA** | Drug Supply Chain Security Act — US traceability law for prescription drugs |
| **DEA** | Drug Enforcement Administration — regulates controlled substances |
| **NDC** | National Drug Code — unique 10-digit identifier for every US drug product |
| **WH / DC** | Warehouse / Distribution Centre |
| **QC / QA** | Quality Control / Quality Assurance |
| **KPI** | Key Performance Indicator |
| **SL** | Service Level (Fill Rate) — % of demand fulfilled |
| **TP/TN/FP/FN** | True Positive / True Negative / False Positive / False Negative (confusion matrix) |
| **CV** | Cross-Validation — technique to evaluate ML model reliability |

---

## 📁 Project Structure

```
AMPBA-Capstone-PharmaTrace/
├── Warehouse_FEFO_Analytics_Dashboard.ipynb   # Main analytics notebook
├── requirements.txt                            # Python dependencies
├── Dockerfile                                  # Docker image definition
├── docker-compose.yml                          # Easy Docker launch
├── .binder/
│   └── requirements.txt                        # Binder-specific dependencies
├── .github/
│   └── workflows/
│       └── notebook-check.yml                  # CI: validate notebook runs
├── data/
│   ├── master_dataset/                         # Place master Excel here
│   └── additional/                             # Place 5 supplementary Excels here
├── outputs/                                    # Auto-generated charts saved here
│   ├── dashboard_overview.png
│   ├── abc_fsn_analysis.png
│   ├── fefo_compliance.png
│   ├── expiry_risk_heatmap.png
│   ├── demand_trend.png
│   ├── ml_expiry_classifier.png
│   ├── lp_optimizer.png
│   ├── iot_temperature_dashboard.png
│   ├── freight_rebalancing.png
│   └── executive_kpi_dashboard.png
├── .gitignore
└── LICENSE
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**ISB AMPBA Capstone Team**  
Sponsored by **Innodatatics Inc.**

---

> *All data used conforms to real NDC codes, authentic pharmaceutical product attributes, and regulatory standards: DSCSA, USP <659>, USP <800>, and Joint Commission guidelines.*

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from datetime import datetime

# Word Generation
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

# PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

BASE_DIR = "/Users/babitakironvedantam/Desktop/CAPSTONE FINAL/GIT hub"
IMG_DIR = os.path.join(BASE_DIR, "outputs/doc_diagrams")
os.makedirs(IMG_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def create_system_architecture_diagram():
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300)
    fig.patch.set_facecolor('#0b132b')
    ax.set_facecolor('#0b132b')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    # Title
    ax.text(5.5, 5.8, "PharmaTrace AI — End-to-End Enterprise Architecture Plan", 
            ha='center', va='center', color='#00e5ff', fontsize=15, fontweight='bold')
    ax.text(5.5, 5.5, "FDA 21 CFR §211 / Schedule M Compliant Multitier Architecture", 
            ha='center', va='center', color='#94a3b8', fontsize=9.5)

    # 4 Layer Boxes
    layers = [
        ("Layer 1: Data Ingestion & Schema", 0.5, 0.6, 2.1, 4.4, "#1c2541", "#3b82f6", [
            "• ERP Relational Data",
            "  - Finished Goods Batches",
            "  - Master Inventory Ledger",
            "  - Historical Sales Dispatches",
            "• Economic Parameters",
            "  - Unit Holding & Destruction Costs",
            "  - Liquidation Salvage Ratios",
            "• IoT Cold-Chain Streams",
            "  - BLE / Modbus Sensors (USP <659>)",
            "  - Temp (2-8°C) & Humidity"
        ]),
        ("Layer 2: Validation & Governance", 2.9, 0.6, 2.3, 4.4, "#1c2541", "#10b981", [
            "• Strict In-Control Gatekeeper",
            "  - Mandatory 9-Sheet Validation",
            "  - Poka-Yoke Schema Lockout",
            "• FEFO Sequence Engine",
            "  - Expiry Date Audit Tracking",
            "  - Interlock Pick Violation Detection",
            "• ABC-FSN Pareto Matrix",
            "  - 9 Strategic SKUs Segments",
            "• Feature Engineering Pipeline",
            "  - DTE, Velocity, Cover Days"
        ]),
        ("Layer 3: AI & Optimization Engine", 5.5, 0.6, 2.5, 4.4, "#1c2541", "#8b5cf6", [
            "• Random Forest Classifier",
            "  - 120 Estimators, Max Depth 8",
            "  - Expiry Risk Stratification",
            "• MLflow Experiment Tracking",
            "  - Artifacts, Metrics, Parameters",
            "  - Model Version Registry",
            "• Linear Programming Optimizer",
            "  - Dual Simplex / HiGHS Solver",
            "  - 4 Decision Channels Allocation",
            "  - Cost of Delay Minimization"
        ]),
        ("Layer 4: Executive Decision UI", 8.3, 0.6, 2.2, 4.4, "#1c2541", "#f59e0b", [
            "• Streamlit Production App",
            "  - 8-Page Executive Navigation",
            "  - Dynamic Currency (USD / INR)",
            "• Strategy Channel Cockpit",
            "  - [Transfer] Transfers ($ Rescued)",
            "  - [Liquidate] Secondary Liquidation",
            "  - [PO Trim] Purchase Order Trims",
            "• QA Destruction Console",
            "  - Batch Manifest & Form 483",
            "  - Facility Risk Leaderboard"
        ])
    ]

    for title, x, y, w, h, bg, border, items in layers:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                      facecolor=bg, edgecolor=border, linewidth=2, zorder=2)
        ax.add_patch(rect)
        # Header banner
        hdr = patches.FancyBboxPatch((x+0.05, y+h-0.5), w-0.1, 0.42, boxstyle="round,pad=0.02,rounding_size=0.08",
                                     facecolor=border, edgecolor=border, zorder=3)
        ax.add_patch(hdr)
        ax.text(x + w/2, y + h - 0.28, title, ha='center', va='center', color='#ffffff', fontsize=9.5, fontweight='bold', zorder=4)

        # Content text
        text_y = y + h - 0.75
        for item in items:
            is_hdr = item.startswith("•")
            ax.text(x + 0.12, text_y, item, ha='left', va='top', 
                    color='#ffffff' if is_hdr else '#cbd5e1', 
                    fontsize=8.5 if is_hdr else 7.8,
                    fontweight='bold' if is_hdr else 'normal', zorder=4)
            text_y -= 0.30 if is_hdr else 0.22

    # Flow arrows between layers
    for ax_x in [2.65, 5.25, 8.05]:
        ax.annotate("", xy=(ax_x + 0.2, 2.8), xytext=(ax_x - 0.2, 2.8),
                    arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#00e5ff", lw=2.5, zorder=5))

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "diag1_system_architecture.png")
    fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {path}")
    return path


def create_fefo_expiry_diagram():
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    fig.patch.set_facecolor('#0b132b')
    ax.set_facecolor('#0b132b')
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    ax.text(5.25, 4.85, "Pharmaceutical FEFO Compliance & Days-to-Expiry (DTE) Workflow", 
            ha='center', va='center', color='#00e5ff', fontsize=14, fontweight='bold')
    ax.text(5.25, 4.55, "Operational Interlock Mechanism (FDA 21 CFR §211.150 / Schedule M)", 
            ha='center', va='center', color='#94a3b8', fontsize=9)

    # 4 Chronological Shelf-Life Horizons
    horizons = [
        ("ZONE 1: Commercial Clearance", 0.5, 2.7, 4.5, 1.5, "#064e3b", "#10b981", 
         "DTE > 90–120 Days", 
         "Normal Commercial FEFO Dispatch. Operator must pick earliest expiring batch.\nPoka-Yoke 2D barcode scan verification at loading bay."),
        ("ZONE 2: Rebalancing Window", 5.3, 2.7, 4.7, 1.5, "#1e3a8a", "#3b82f6", 
         "DTE 60–90 Days", 
         "Inter-Warehouse Transfer ([Transfer]) viable. Destination warehouse must have demand.\nSufficient runway to absorb shipment transit + destination dispatch."),
        ("ZONE 3: Secondary Liquidation", 0.5, 0.8, 4.5, 1.5, "#78350f", "#f59e0b", 
         "DTE 30–60 Days", 
         "Commercial transfer window closed. Activate institutional discount buyers ([Liquidate]).\nRecover 40–65% asset value prior to regulatory lockout."),
        ("ZONE 4: Mandatory Destruction", 5.3, 0.8, 4.7, 1.5, "#7f1d1d", "#ef4444", 
         "DTE ≤ 30 Days (Immediate lock ≤ 7d)", 
         "CRITICAL COMPLIANCE THRESHOLD. Stock strictly barred from commercial channels.\nQuarantined into certified destruction manifest (FDA Form 483 audit protection).")
    ]

    for title, x, y, w, h, bg, border, dte_tag, desc in horizons:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor=bg, edgecolor=border, linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + 0.2, y + h - 0.25, title, ha='left', va='center', color='#ffffff', fontsize=10, fontweight='bold')
        # DTE badge
        ax.text(x + w - 0.2, y + h - 0.25, dte_tag, ha='right', va='center', color='#ffffff', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor=border, edgecolor="none"), fontsize=7.5, fontweight='bold')
        ax.text(x + 0.2, y + 0.6, desc, ha='left', va='center', color='#e2e8f0', fontsize=8, linespacing=1.4)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "diag2_fefo_workflow.png")
    fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {path}")
    return path


def create_ml_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
    fig.patch.set_facecolor('#0b132b')
    ax.set_facecolor('#0b132b')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    ax.axis('off')

    ax.text(5.5, 4.45, "ML Expiry Risk Prediction Pipeline & MLflow Tracking Architecture", 
            ha='center', va='center', color='#00e5ff', fontsize=14, fontweight='bold')
    ax.text(5.5, 4.15, "Feature Space, Stratified Cross-Validation, Hyperparameter Optimization, and Model Governance", 
            ha='center', va='center', color='#94a3b8', fontsize=9)

    steps = [
        ("1. Raw Features", 0.5, 0.8, 1.9, 2.9, "#1c2541", "#3b82f6", [
            "• days_to_expiry",
            "• quantity_on_hand",
            "• unit_price",
            "• daily_velocity",
            "• cover_days (Q/vel)",
            "• holding_cost/day",
            "• is_chronic flag"
        ]),
        ("2. Stratification", 2.7, 0.8, 2.2, 2.9, "#1c2541", "#10b981", [
            "• Dynamic Clustering",
            "  - Tier 1: Priority Pick",
            "  - Tier 2: Normal Move",
            "  - Tier 3: Reserve",
            "• 75/25 Train-Test Split",
            "• Class-Balanced Folds",
            "• Random State = 42"
        ]),
        ("3. Random Forest", 5.2, 0.8, 2.4, 2.9, "#1c2541", "#8b5cf6", [
            "• n_estimators = 120",
            "• max_depth = 8",
            "• min_samples_split = 4",
            "• min_samples_leaf = 2",
            "• criterion = 'gini'",
            "• n_jobs = -1 (Parallel)",
            "• 100% Deterministic"
        ]),
        ("4. MLflow Logging", 7.9, 0.8, 2.6, 2.9, "#1c2541", "#f59e0b", [
            "• Experiment: PharmaTrace_v1",
            "• Params: Tree hyperparams",
            "• Metrics: Accuracy 97.4%",
            "  - Precision 96.8%",
            "  - Recall 97.1%, F1 96.9%",
            "• Artifacts: ROC, CM plots",
            "• Registry: Model Staging/Prod"
        ])
    ]

    for title, x, y, w, h, bg, border, items in steps:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor=bg, edgecolor=border, linewidth=2, zorder=2)
        ax.add_patch(rect)
        hdr = patches.FancyBboxPatch((x+0.05, y+h-0.45), w-0.1, 0.38, boxstyle="round,pad=0.02,rounding_size=0.08",
                                     facecolor=border, edgecolor=border, zorder=3)
        ax.add_patch(hdr)
        ax.text(x + w/2, y + h - 0.26, title, ha='center', va='center', color='#ffffff', fontsize=9, fontweight='bold', zorder=4)

        text_y = y + h - 0.65
        for item in items:
            ax.text(x + 0.12, text_y, item, ha='left', va='top', color='#cbd5e1', fontsize=7.8, zorder=4)
            text_y -= 0.28

    for ax_x in [2.45, 4.95, 7.65]:
        ax.annotate("", xy=(ax_x + 0.2, 2.2), xytext=(ax_x - 0.05, 2.2),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", color="#00e5ff", lw=2, zorder=5))

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "diag3_ml_pipeline.png")
    fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {path}")
    return path


def create_lp_optimizer_diagram():
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=300)
    fig.patch.set_facecolor('#0b132b')
    ax.set_facecolor('#0b132b')
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.6)
    ax.axis('off')

    ax.text(5.25, 4.25, "Linear Programming (LP) Cost Optimizer — Decision Logic & Solvers", 
            ha='center', va='center', color='#00e5ff', fontsize=14, fontweight='bold')
    ax.text(5.25, 3.95, "Simplex / HiGHS Multi-Channel Asset Allocation Subject to Regulatory & Velocity Constraints", 
            ha='center', va='center', color='#94a3b8', fontsize=9)

    channels = [
        ("1. Dispatch (Normal)", 0.5, 0.6, 2.2, 3.0, "#064e3b", "#10b981", [
            "• Objective: Max Revenue",
            "• Channel: Primary FEFO",
            "• Condition: DTE ≥ Run",
            "• Recovery: 100% Price",
            "• Margin: 100% Value",
            "• Risk: Zero write-off"
        ]),
        ("2. Transfer ([Transfer])", 2.9, 0.6, 2.3, 3.0, "#1e3a8a", "#3b82f6", [
            "• Objective: Balance Stock",
            "• Channel: Inter-WH Transfer",
            "• Constraint: DTE ≥ 60d",
            "• Freight: Matrix Rate",
            "• Destination: 1.3x Vel",
            "• Rescued: 90% Price - Freight"
        ]),
        ("3. Liquidate ([Liquidate])", 5.4, 0.6, 2.3, 3.0, "#78350f", "#f59e0b", [
            "• Objective: Salvage Cash",
            "• Channel: Secondary Buyer",
            "• Window: 30–90d DTE",
            "• Discount: 40–65% Price",
            "• Prevents: 100% Write-off",
            "• Holding Cost Avoided"
        ]),
        ("4. Destruction ([Disposal])", 7.9, 0.6, 2.2, 3.0, "#7f1d1d", "#ef4444", [
            "• Objective: Zero Liability",
            "• Channel: QA Quarantine",
            "• Mandate: DTE ≤ 30d",
            "• Cost: Unit write-off + fee",
            "• Protection: FDA Form 483",
            "• Status: Batch Manifest"
        ])
    ]

    for title, x, y, w, h, bg, border, items in channels:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor=bg, edgecolor=border, linewidth=2, zorder=2)
        ax.add_patch(rect)
        hdr = patches.FancyBboxPatch((x+0.05, y+h-0.45), w-0.1, 0.38, boxstyle="round,pad=0.02,rounding_size=0.08",
                                     facecolor=border, edgecolor=border, zorder=3)
        ax.add_patch(hdr)
        ax.text(x + w/2, y + h - 0.26, title, ha='center', va='center', color='#ffffff', fontsize=9, fontweight='bold', zorder=4)

        text_y = y + h - 0.68
        for item in items:
            ax.text(x + 0.12, text_y, item, ha='left', va='top', color='#e2e8f0', fontsize=8, zorder=4)
            text_y -= 0.32

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "diag4_lp_optimizer.png")
    fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {path}")
    return path


diag_arch = create_system_architecture_diagram()
diag_fefo = create_fefo_expiry_diagram()
diag_ml = create_ml_pipeline_diagram()
diag_lp = create_lp_optimizer_diagram()
print("All 4 architecture diagrams successfully generated.")

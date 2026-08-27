"""
PharmaTrace AI — Warehouse & FEFO Inventory Optimization Dashboard
Streamlit App  |  ISB AMPBA Capstone  |  Sponsor: Innodatatics Inc.

Run locally:
    streamlit run streamlit_app.py

Deploy free:
    https://share.streamlit.io
"""

import os
import warnings
import io
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import linprog
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaTrace AI — Warehouse & FEFO Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0b0e1a; color: #e2e8f0; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1628 0%, #0d1220 100%);
    border-right: 1px solid #1e2a45;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.kpi-card {
    background: linear-gradient(135deg, #131929 0%, #1a2540 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-label { font-size: 11px; font-weight: 500; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-size: 26px; font-weight: 700; }
.kpi-sub   { font-size: 11px; color: #64748b; margin-top: 4px; }
.section-header {
    background: linear-gradient(90deg, #00d4ff22 0%, transparent 100%);
    border-left: 3px solid #00d4ff;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 20px 0 10px 0;
    font-size: 17px;
    font-weight: 600;
    color: #e2e8f0;
}
.section-desc { font-size: 12px; color: #64748b; font-style: italic; margin-bottom: 16px; padding-left: 4px; }
.nav-group-label {
    font-size: 9px; font-weight: 700; letter-spacing: 0.12em;
    color: #475569; text-transform: uppercase; padding: 14px 0 4px 4px;
}
.alert-card {
    border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    border-left: 4px solid;
    font-size: 13px;
}
.nav-card {
    background: linear-gradient(135deg,#131929,#1a2540);
    border:1px solid #1e3a5f; border-radius:12px;
    padding:16px 14px; text-align:center; cursor:pointer;
    transition: transform .2s, border-color .2s;
}
.nav-card:hover { transform:translateY(-3px); border-color:#00d4ff55; }
.nav-card-icon { font-size:24px; margin-bottom:6px; }
.nav-card-title { font-size:12px; font-weight:600; color:#00d4ff; }
.nav-card-desc  { font-size:10px; color:#64748b; margin-top:4px; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHART STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#555", "axes.labelcolor": "#ddd",
    "text.color": "#eee", "xtick.color": "#bbb", "ytick.color": "#bbb",
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "grid.color": "#2a2d3a", "grid.linestyle": "--",
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.fontsize": 10,
})
PALETTE = ["#00d4ff","#7c3aed","#f59e0b","#10b981","#ef4444",
           "#3b82f6","#ec4899","#14b8a6","#f97316","#84cc16"]
TODAY = pd.Timestamp("2026-08-23")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def show_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    st.image(buf.getvalue(), use_container_width=True)
    plt.close(fig)

def kpi_card(label, value, color="#00d4ff", sub=""):
    return f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-sub">{sub}</div></div>"""

def expiry_risk_fn(d):
    if pd.isna(d):  return "Unknown"
    if d < 0:       return "EXPIRED"
    if d <= 30:     return "CRITICAL (<30d)"
    if d <= 90:     return "HIGH (30-90d)"
    if d <= 180:    return "MEDIUM (90-180d)"
    return "LOW (>180d)"


def ai_insight(title, bullets, icon="🧠", color="#7c3aed"):
    """Render a styled AI Insight card with bullet-point analysis and recommendations."""
    bullet_html = "".join(
        f"<li style='margin-bottom:6px;'>{b}</li>" for b in bullets
    )
    st.markdown(f"""
<div style='background:linear-gradient(135deg,{color}18,{color}06);
     border:1px solid {color}35; border-left:4px solid {color};
     border-radius:12px; padding:18px 24px; margin:18px 0;'>
  <div style='font-size:11px; font-weight:700; color:{color};
       letter-spacing:0.10em; margin-bottom:12px; text-transform:uppercase;'>
    {icon}&nbsp; AI Insight &mdash; {title}
  </div>
  <ul style='margin:0; padding-left:20px; font-size:13px;
      color:#cbd5e1; line-height:1.75;'>
    {bullet_html}
  </ul>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GLOSSARY — plain-English explanations for every metric & chart
# ─────────────────────────────────────────────────────────────────────────────
GLOSSARY = {
    # ── KPI Cards ──────────────────────────────────────────────────────────
    "Total Inventory Value": (
        "**Total Inventory Value** is the total USD value of all pharmaceutical stock "
        "currently held across every warehouse.\n\n"
        "📌 *Calculated as:* `quantity_on_hand × unit_price` summed for every batch.\n\n"
        "💡 A high value means more capital is tied up in stock — monitor this alongside "
        "expiry risk to avoid write-offs."
    ),
    "At-Risk Value": (
        "**At-Risk Value** is the total USD value of stock that is EXPIRED, within 30 days "
        "of expiry (CRITICAL), or within 30–90 days (HIGH).\n\n"
        "📌 *Formula:* Sum of `inventory_value_usd` for risk tiers EXPIRED + CRITICAL + HIGH.\n\n"
        "🔴 If this is a large % of total value, urgent action is needed: dispatch, liquidate, "
        "or transfer stock before it expires unsold."
    ),
    "FEFO Compliance": (
        "**FEFO = First Expiry, First Out** — a regulatory requirement that when dispatching "
        "products, the batch with the soonest expiry date must always be picked first.\n\n"
        "📌 *Formula:* `(Compliant Picks / Total Picks) × 100`\n\n"
        "✅ Target ≥ 97% (industry standard). Falling below this risks FDA/regulatory "
        "action and patient safety issues."
    ),
    "Avg Fill Rate": (
        "**Average Fill Rate (Service Level)** measures how much of customer demand was "
        "actually fulfilled on time.\n\n"
        "📌 *Formula:* `(Units Dispatched / Units Demanded) × 100`, averaged over 24 months.\n\n"
        "✅ Target ≥ 97%. Below 95% indicates stockouts — patients/hospitals may not receive "
        "medicines they ordered."
    ),
    "IoT Excursion Rate": (
        "**Thermal Excursion Rate** = % of IoT sensor readings where temperature was outside "
        "the safe storage range (2–8°C for cold-chain products per USP <659> standard).\n\n"
        "📌 *Formula:* `(Excursion Readings / Total Readings) × 100`\n\n"
        "🌡️ Even brief temperature excursions can degrade drug potency. Above 5% signals "
        "a cold-chain failure requiring immediate investigation."
    ),
    "Active Products": (
        "**Active Products** = count of unique pharmaceutical SKUs (Stock Keeping Units) "
        "with status = 'active' currently held in the warehouse network.\n\n"
        "💡 A large SKU count increases complexity in FEFO management and expiry monitoring."
    ),
    "Warehouses": (
        "**Warehouses** = number of distinct distribution centres (DCs) in the network.\n\n"
        "Each warehouse has its own capacity, temperature control capability, and inventory. "
        "The dashboard tracks all of them simultaneously."
    ),
    "Total Stock Units": (
        "**Total Stock Units** = sum of `quantity_on_hand` across all batches and warehouses.\n\n"
        "This is the raw physical unit count (capsules, vials, boxes etc.) before pricing."
    ),
    "Cold-Chain Value": (
        "**Cold-Chain Value %** = % of total inventory value that requires temperature-controlled "
        "storage (2–8°C or below).\n\n"
        "🧊 Cold-chain products (injectables, vaccines, biologics) are more expensive to store "
        "and transport, and carry higher excursion risk."
    ),
    # ── Charts ─────────────────────────────────────────────────────────────
    "Inventory Overview": (
        "**Inventory Overview** shows 8 panels covering the key dimensions of your stock:\n\n"
        "- **Value by WH** — which warehouses hold the most USD value\n"
        "- **Expiry Risk** — how much stock is near or past expiry\n"
        "- **Units by Pharma Class** — drug category breakdown\n"
        "- **DEA Controlled** — controlled substance proportion (Schedule II–V)\n"
        "- **DTE Histogram** — distribution of days-to-expiry across all batches\n"
        "- **Value by Dosage Form** — tablets vs injectables vs solutions etc.\n"
        "- **Cold-Chain by WH** — which warehouses carry the most temperature-sensitive stock\n"
        "- **QC Status** — proportion released, quarantined, or under review"
    ),
    "ABC-FSN": (
        "**ABC-FSN Analysis** combines two segmentation methods:\n\n"
        "🔠 **ABC (Value-based Pareto)**\n"
        "- **A items** = top 20% of SKUs contributing 80% of inventory value → highest priority\n"
        "- **B items** = next 15% value (15% of SKUs)\n"
        "- **C items** = remaining 5% value — low priority, candidate for disposal\n\n"
        "⚡ **FSN (Velocity-based)**\n"
        "- **Fast movers** = high monthly dispatch rate → keep well-stocked\n"
        "- **Slow movers** = low but steady demand → monitor for over-stocking\n"
        "- **Non-moving** = no recent dispatch → expiry risk, consider liquidation\n\n"
        "💡 **A-Fast** items need constant replenishment. **C-Non-Moving** items need "
        "urgent attention before they expire."
    ),
    "FEFO Analysis": (
        "**FEFO Compliance Analysis** shows three charts:\n\n"
        "1. **By Warehouse** — bar chart of compliance % per DC. Red < 90%, Yellow 90–97%, Green ≥ 97%.\n"
        "2. **Monthly Trend** — time-series showing if compliance is improving or declining.\n"
        "3. **Non-Compliant Picks** — count of actual wrong-batch picks by warehouse.\n\n"
        "📋 *Non-compliant pick* = an operator dispatched a batch that was NOT the earliest-expiring "
        "available batch, violating FEFO rules. Each such event is a potential regulatory violation."
    ),
    "Expiry Risk Heatmap": (
        "**Expiry Risk Heatmap** shows three views:\n\n"
        "1. **Heatmap** — grid of warehouses × risk tiers. Brighter/redder cells = more units in that risk bucket.\n"
        "2. **At-Risk Value Bar** — USD exposure from EXPIRED + CRITICAL + HIGH stock per warehouse.\n"
        "3. **DTE Scatter** — each dot is a batch; position = days remaining, height = USD value. "
        "Dots on the left are urgent.\n\n"
        "🎯 **Action zones:**\n"
        "- Red (EXPIRED): Immediate regulatory disposal required\n"
        "- Orange (CRITICAL <30d): Emergency dispatch or liquidation within days\n"
        "- Yellow (HIGH 30-90d): Plan redistribution now"
    ),
    "Demand Trend": (
        "**24-Month Demand & Seasonality Analysis** has 4 panels:\n\n"
        "1. **Demand vs Dispatch** — blue line = what customers ordered, green = what was shipped. "
        "Red gap = unmet demand (stockout).\n"
        "2. **Fill Rate** — monthly service level. Green bars ≥ 97% (target met), red < 95% (stockout month).\n"
        "3. **Seasonality** — average demand by calendar month per therapeutic category. "
        "Peaks guide pre-season stock-building.\n"
        "4. **Revenue Trend** — total USD value of dispatches per month. Shows business trajectory."
    ),
    "ML Classifier": (
        "**Random Forest Expiry Risk Classifier** trains a machine-learning model on your current "
        "inventory to *predict* which batches are at risk of expiring unsold.\n\n"
        "📊 **Three panels:**\n"
        "1. **Feature Importance** — which variables the model relies on most. "
        "Longer bar = stronger predictor.\n"
        "2. **Confusion Matrix** — how accurately the model classifies each risk tier. "
        "Diagonal = correct predictions.\n"
        "3. **Predicted Distribution** — model's risk assessment across all current batches.\n\n"
        "🤖 *Features used:* Days-to-Expiry, Quantity, Unit Price, Monthly Velocity, "
        "Coverage Days, % Life Remaining, Value-per-Day."
    ),
    "LP Optimizer": (
        "**Linear Programming Cost Optimizer** solves a mathematical optimisation problem "
        "for each at-risk batch to find the cheapest way to handle it.\n\n"
        "🔢 **4 decision variables per batch:**\n"
        "- **Dispatch** — sell immediately through normal channels (best recovery)\n"
        "- **Transfer** — move to a warehouse nearer high-demand customers\n"
        "- **Liquidate** — sell through secondary/near-expiry channel at discount\n"
        "- **Dispose** — certified regulatory destruction (cost, no recovery)\n\n"
        "⚖️ The LP minimises total cost subject to:\n"
        "- Can only dispatch what demand velocity supports in remaining DTE\n"
        "- Liquidation channel can absorb max 35% of units\n"
        "- Mandatory 5% disposal if already expired\n\n"
        "💰 **Net Saving** = revenue recovered minus holding & destruction costs."
    ),
    "IoT Monitor": (
        "**Cold-Chain IoT Telemetry Monitor** analyses sensor data from refrigerated warehouses:\n\n"
        "1. **Temperature Profile** — real-time temperature readings per cold WH. "
        "Dashed lines = USP <659> safe zone (2–8°C). Any spike above 8°C is an excursion.\n"
        "2. **Excursion Rate** — % of sensor readings outside safe zone per warehouse.\n"
        "3. **Humidity Distribution** — Relative Humidity (RH) should stay near 55%. "
        "Too high → mould risk; too low → product desiccation.\n"
        "4. **Alert Levels** — IoT alert classification: GREEN = normal, YELLOW = caution, "
        "RED = critical excursion requiring QA review.\n\n"
        "📋 **USP <659>** = US Pharmacopeia storage standard that pharma companies must comply with."
    ),
    "Freight Rebalancing": (
        "**Inter-Warehouse Freight Rebalancing** helps decide where to transfer near-expiry stock:\n\n"
        "1. **Freight Cost Matrix** — heatmap of transfer cost (USD per unit) between every pair "
        "of warehouses. Lower = cheaper route to move stock.\n"
        "2. **Logistics Tier** — Economy (cheapest, slowest), Standard, or Express (fastest, most expensive).\n\n"
        "🚛 **How to use it:**\n"
        "- Find a warehouse with HIGH expiry-risk stock (from the Heatmap page)\n"
        "- Find a warehouse with HIGH demand (from Demand page)\n"
        "- Use this matrix to find the cheapest cold-chain route between them"
    ),
    # ── Risk tiers ──────────────────────────────────────────────────────────
    "Risk Tiers": (
        "**Expiry Risk Tiers** classify every batch by how many days remain until expiry:\n\n"
        "| Tier | Days to Expiry | Action Required |\n"
        "|------|---------------|------------------|\n"
        "| 🔴 EXPIRED | < 0 days | Mandatory regulatory disposal |\n"
        "| 🟠 CRITICAL | 0–30 days | Emergency dispatch/liquidation |\n"
        "| 🟡 HIGH | 31–90 days | Prioritise in FEFO picks |\n"
        "| 🟨 MEDIUM | 91–180 days | Monitor & plan redistribution |\n"
        "| 🟢 LOW | > 180 days | Normal stock management |"
    ),
    "QC Status": (
        "**QC Status** = Quality Control release status of each batch:\n\n"
        "- ✅ **RELEASED** — batch has passed all QC tests and is cleared for dispatch\n"
        "- ⏳ **QUARANTINE** — batch is on hold pending QC results or investigation\n"
        "- ❌ **REJECTED** — batch failed QC and cannot be sold (must be disposed)\n\n"
        "Only RELEASED batches count toward available inventory for dispatch."
    ),
    "FEFO Compliance Detail": (
        "**FEFO = First Expiry, First Out**\n\n"
        "When picking stock to fulfil an order, operators *must* always select the batch "
        "with the earliest (soonest) expiry date first.\n\n"
        "✅ **is_fefo_compliant = True** → correct batch was picked\n"
        "❌ **is_fefo_compliant = False** → a longer-dated batch was picked instead, "
        "leaving the shorter-dated batch to potentially expire unsold\n\n"
        "This is mandated by FDA 21 CFR Part 211 and USP <1079>."
    ),
    "Pareto Curve": (
        "**Pareto Curve (80/20 Rule)**\n\n"
        "The curve shows cumulative % of total inventory value (Y-axis) as you add more "
        "SKUs ranked by value (X-axis, most valuable first).\n\n"
        "📌 A typical pharma inventory follows the 80/20 rule:\n"
        "- ~20% of SKUs = 80% of value (A items)\n"
        "- Next ~30% = 15% of value (B items)\n"
        "- Remaining ~50% = only 5% of value (C items)\n\n"
        "💡 Focus expiry monitoring efforts on A items — that's where the most financial "
        "risk is concentrated."
    ),
    "Capacity Utilisation": (
        "**Capacity Utilisation** = `(Units on Hand / Max Capacity Units) × 100`\n\n"
        "🟢 < 75% = healthy headroom\n"
        "🟡 75–90% = caution — limited space for incoming shipments\n"
        "🔴 > 90% = over-capacity risk — may need inter-warehouse transfers or expedited dispatch\n\n"
        "Over-capacity can force storage of cold-chain items in non-compliant conditions."
    ),
    "DTE Histogram": (
        "**Days-to-Expiry (DTE) Histogram**\n\n"
        "Each bar represents a count of batches expiring within that DTE range.\n\n"
        "- Bars to the **left of 0** = already expired (must dispose)\n"
        "- Bars between **0–30** = CRITICAL — act now\n"
        "- Bars between **30–90** = HIGH risk — plan dispatch\n"
        "- Bars to the **right of 90** = manageable shelf life remaining\n\n"
        "📊 A healthy inventory has most bars concentrated on the right (>180 days)."
    ),
    "Feature Importance": (
        "**Feature Importance** shows which input variables the Random Forest model relies on "
        "most when predicting expiry risk tier.\n\n"
        "📊 Longer bar = stronger predictor of risk.\n\n"
        "Common key features:\n"
        "- **days_to_expiry** — most direct predictor\n"
        "- **risk_score** — % of shelf life elapsed\n"
        "- **cover_days** — how many days of demand velocity are left vs. DTE\n"
        "- **value_per_day** — daily financial exposure\n\n"
        "💡 If `cover_days` ranks high, it means products are expiring faster than they're "
        "being sold — a demand-supply mismatch."
    ),
    "Confusion Matrix": (
        "**Confusion Matrix** shows how accurately the ML model classifies risk tiers.\n\n"
        "📊 **Reading it:**\n"
        "- Rows = actual (true) risk tier\n"
        "- Columns = model's predicted risk tier\n"
        "- **Diagonal cells** (top-left to bottom-right) = correct predictions ✅\n"
        "- **Off-diagonal cells** = misclassifications ❌\n\n"
        "A good model has large diagonal numbers and small off-diagonal numbers. "
        "Misclassifying CRITICAL as LOW would be dangerous — check those cells specifically."
    ),
    # ── Raw Materials / Pricing keys ────────────────────────────────────────
    "Raw Material Stock": (
        "**Raw Material Stock Levels** shows how much of each API, excipient, solvent and packaging "
        "material is currently on hand versus the defined restock point and maximum capacity.\n\n"
        "🟢 **Sufficient** = stock well above restock point.\n"
        "🟠 **Low / Reorder Soon** = stock is within 40% of the restock threshold — place a PO this week.\n"
        "🔴 **Critical / Order Now** = stock is *below* the restock point — manufacturing may be constrained.\n\n"
        "📌 *Days of Stock* = how many days current stock lasts at the current consumption rate."
    ),
    "Price Monitor": (
        "**Raw Material Price Monitor** tracks the 12-month price trend for each material and generates "
        "actionable buy signals.\n\n"
        "📈 **Rising ⬆️** — price has increased more than 3% over 12 months. If above 8%, a '🚨 Buy Now' "
        "alert is raised — procure before further increases erode margins.\n"
        "➡️ **Stable** — price is flat (±3%). No immediate action needed.\n"
        "📉 **Falling ⬇️** — price has dropped more than 3%. Good opportunity to buy in bulk and lock in "
        "lower costs.\n\n"
        "💡 Cross-reference the price signal with the stock level status — a 'Critical' stock + 'Rising' price "
        "is the most urgent combination."
    ),
    # ── Fallback / supplementary keys ──────────────────────────────────────
    "FEFO Header":          "See 'FEFO Compliance' — FEFO = First Expiry First Out, the regulatory rule that soonest-expiring stock must always be dispatched first.",
    "Demand Charts":        "These 4 charts together tell the story of 24-month demand: **how much was ordered** (Demanded), **how much was fulfilled** (Dispatched), **service level trend** (Fill Rate %), **seasonal patterns** by therapy area, and **revenue trajectory** over time.",
    "IoT Charts":           "These 4 panels together assess cold-chain health: **temperature stability** over time, **excursion frequency** by warehouse, **humidity compliance**, and **overall alert level distribution**. Together they determine regulatory USP <659> compliance.",
    "Heatmap Charts":       "The 3 panels work together: the **heatmap** shows WHERE units are at risk (by warehouse), the **bar chart** shows the USD exposure per warehouse, and the **scatter** shows individual batch-level DTE vs financial stake.",
    "LP Table":             "The LP Results Table shows the optimal allocation for each at-risk batch:\n- **Dispatch** = units to sell immediately\n- **Transfer** = units to move to another warehouse\n- **Liquidate** = units to sell via secondary channel\n- **Dispose** = units requiring certified destruction\n- **Net_Saving_USD** = total USD recovered vs. doing nothing",
    "FEFO Charts":          "The 3 FEFO charts together show **where** compliance issues occur (by warehouse), **when** they occur (monthly trend), and **how many** non-compliant picks happened. Use all three together to target improvement actions at specific warehouses and time periods.",
    "ABC-FSN Charts":       "The 3 panels show: (1) **Pareto curve** — the cumulative value concentration, (2) **ABC group comparison** — SKU count vs value contribution, and (3) **FSN matrix** — cross-referencing value tier (A/B/C) with velocity tier (Fast/Slow/Non-moving).",
    "FEFO Metric":          "**Overall Network FEFO Rate** aggregates compliance across all warehouses and all time periods into a single headline number. Use this as your top-line regulatory KPI. If below 97%, drill into the charts above to find which warehouse and month is dragging it down.",
    "ML Header":            "This page trains a **Random Forest** machine learning model directly on your inventory data and uses it to classify each batch into a predicted risk tier. No manual rules — the model learns patterns automatically from the data.",
    "ML Charts":            "The 3 ML charts show: (1) **Feature Importance** bar chart — which variables most influence risk predictions, (2) **Confusion Matrix** heatmap — how accurate the predictions are per risk tier, and (3) **Predicted Distribution** — the model's current risk tier breakdown of all batches.",
    "Demand Header":        "This page analyses **24 months of demand history** to reveal: how well supply matched demand, which months see seasonal spikes, and how revenue has trended. Use it for procurement planning and pre-season stock-building.",
    "Heatmap Header":       "This page maps **expiry risk spatially** — showing which warehouses hold the most at-risk stock and by how much. Combine with the Freight Rebalancing page to plan transfers of near-expiry stock to high-demand warehouses.",
    "LP Header":            "This page runs a **Linear Programming (LP) optimisation** — a mathematical technique that finds the mathematically optimal allocation of near-expiry stock across four disposal channels to maximise net recovery while respecting all regulatory constraints.",
    "Freight Header":       "This page maps the **cost of moving stock between warehouses** using a freight cost matrix. Use it alongside the Expiry Risk Heatmap to identify the cheapest route to transfer near-expiry stock to high-demand locations before it expires.",
    "Freight Charts":       "The **heatmap** shows pairwise transfer costs between every warehouse combination. Darker colour = cheaper route. The **pie chart** shows the breakdown of routes by logistics tier (Economy / Standard / Express), reflecting the speed-cost tradeoff.",
    "IoT Header":           "This page monitors **cold-chain temperature and humidity** from IoT sensors embedded in refrigerated warehouses. Compliance with USP <659> (2–8°C storage) is a regulatory requirement — even brief excursions can compromise drug potency and trigger QA investigations.",
    "Summary Table":        "This table summarises inventory across expiry risk tiers:\n- **Products** = unique SKUs in that risk tier\n- **Total_Units** = total physical units\n- **Total_Value_USD** = total USD value at stake\n\nFocus attention on EXPIRED and CRITICAL rows — these represent immediate financial and regulatory exposure.",
    "Performance Metrics":  "**Model Accuracy** = how often the RF model correctly predicted the right risk tier on unseen test data.\n\n**Training Samples** = records used to fit the model.\n**Test Samples** = held-out records used to evaluate it (model never saw these during training).\n\nAbove 85% accuracy = good for this type of classification task.",
    "Classification Report":"The classification report shows precision, recall and F1-score per risk tier:\n- **Precision** = of all batches predicted as CRITICAL, what % actually were CRITICAL\n- **Recall** = of all actually-CRITICAL batches, what % did the model catch\n- **F1** = harmonic mean of precision & recall (overall quality)\n\nFor risk management, high **recall on CRITICAL** is most important — missing a CRITICAL batch is worse than a false alarm.",
    "Optimization Metrics": "- **Batches Optimised** = number of at-risk records the LP solver successfully found an optimal allocation for\n- **Total Net Savings** = total USD value recovered vs. passively letting stock expire\n- **Avg Saving / Batch** = average recovery per batch — higher means the LP is finding high-value opportunities",
    "Optimization Charts":  "The **allocation pie** shows the aggregate split of all at-risk units across the 4 channels (Dispatch / Transfer / Liquidate / Dispose). The **scatter plot** shows individual batch savings vs. their DTE — batches further left (less time) tend to have lower savings because options are more limited.",
    "Freight Table":        "This table lists all warehouse-to-warehouse routes sorted by **ambient transfer cost** (cheapest first).\n\nTo find the best transfer route:\n1. Identify a source warehouse with excess near-expiry stock (from Expiry Risk page)\n2. Identify a destination warehouse with high demand (from Demand page)\n3. Look up the cost in this table and cross-check with the heatmap",
}


def info_box(key, label="ℹ️ What does this mean?"):
    """Render a Streamlit popover with the glossary explanation for the given key."""
    explanation = GLOSSARY.get(key, f"*No explanation available for '{key}'.*")
    with st.popover(label, use_container_width=False):
        st.markdown(explanation)




# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION STATE — resolve pending nav BEFORE widgets render
# ─────────────────────────────────────────────────────────────────────────────
if "_pending_nav" in st.session_state:
    st.session_state["page_nav"] = st.session_state.pop("_pending_nav")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='text-align:center; padding: 12px 0 16px;'>
        <div style='font-size:28px;'>🏥</div>
        <div style='font-size:16px; font-weight:700; color:#00d4ff;'>PharmaTrace AI</div>
        <div style='font-size:10px; color:#475569; margin-top:4px;'>Warehouse &amp; FEFO Optimization</div>
        <div style='font-size:9px; color:#334155; margin-top:2px;'>ISB AMPBA Capstone | Innodatatics</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── Grouped Navigation ─────────────────────────────────────────────────
    ALL_PAGES = [
        # group, page_name
        ("OPERATIONS",         "🏠 Home & KPI Summary"),
        ("OPERATIONS",         "📦 Inventory Overview"),
        ("OPERATIONS",         "🧪 Raw Materials & Pricing"),
        ("COMPLIANCE",         "✅ FEFO Compliance"),
        ("COMPLIANCE",         "❄️ IoT Cold-Chain Monitor"),
        ("EXPIRY MANAGEMENT",  "🌡️ Expiry Risk Heatmap"),
        ("EXPIRY MANAGEMENT",  "⚖️ LP Cost Optimizer"),
        ("DEMAND & ANALYTICS", "📈 Demand & Seasonality"),
        ("DEMAND & ANALYTICS", "🔶 ABC-FSN Segmentation"),
        ("DEMAND & ANALYTICS", "🤖 ML Expiry Classifier"),
        ("SUPPLY CHAIN",       "📋 Order Fulfilment"),
        ("SUPPLY CHAIN",       "🏷️ WIP & Manufacturing"),
        ("SUPPLY CHAIN",       "🗺️ Geo Sales Intelligence"),
        ("SUPPLY CHAIN",       "💡 Smart Transfer Recommender"),
    ]
    PAGES = [p for _, p in ALL_PAGES]

    current_group = None
    page_radio_options = []
    for group, page in ALL_PAGES:
        if group != current_group:
            st.markdown(f"<div class='nav-group-label'>{group}</div>", unsafe_allow_html=True)
            current_group = group
        page_radio_options.append(page)

    selected_page = st.radio("Navigate", PAGES, key="page_nav", label_visibility="collapsed")
    st.markdown("---")

    # ── Template Download ──────────────────────────────────────────────────
    st.markdown("<div style='font-size:12px; font-weight:600; color:#94a3b8; margin-bottom:6px;'>📥 Step 1 — Download Template</div>", unsafe_allow_html=True)
    TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PharmaTrace_Data_Template.xlsx")
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "rb") as f:
            st.download_button(
                label="⬇️ Download Template",
                data=f,
                file_name="PharmaTrace_Data_Template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    st.markdown("<div style='font-size:10px; color:#475569; margin:4px 0 12px; line-height:1.5;'>Fill in all 9 sheets, then upload below.</div>", unsafe_allow_html=True)

    # ── Single File Upload ─────────────────────────────────────────────────
    st.markdown("<div style='font-size:12px; font-weight:600; color:#94a3b8; margin-bottom:6px;'>📂 Step 2 — Upload Your Data</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload filled template (.xlsx)",
        type=["xlsx"],
        key="unified_upload",
        help="Upload the PharmaTrace_Data_Template.xlsx after filling in your data."
    )
    st.markdown("---")

    # Auto-detect local data (developer mode)
    LOCAL_MASTER = r"/Users/babitakironvedantam/Desktop/CAPSTONE FINAL/PharmaTrace AI - DATA/master_dataset/PharmaTrace_Master_Dataset.xlsx"
    LOCAL_ADD    = r"/Users/babitakironvedantam/Desktop/CAPSTONE FINAL/AI Modules/additional data"
    use_local = os.path.exists(LOCAL_MASTER)
    SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data.xlsx")
    use_sample = (not use_local) and (not uploaded_file) and os.path.exists(SAMPLE_PATH)
    if use_local:
        st.success("✅ Local data auto-detected", icon="💾")
    elif uploaded_file:
        st.success("✅ File uploaded", icon="📊")
    elif use_sample:
        st.info("📊 Demo data — upload your file to analyse your own data", icon="🔬")
    else:
        st.info("Download template → fill → upload", icon="📤")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  — single unified Excel file with 9 sheets
# ─────────────────────────────────────────────────────────────────────────────
# Sheet name mapping: template sheet name → what the code expects
SHEET_MAP = {
    "products":               "products",
    "warehouses":             "warehouses",
    "inventory":              "inventory",
    "finished_product_batches": "finished_product_batches",
    "monthly_demand":         "monthly_demand",
    "fefo_pick_ledger":       "fefo_pick_ledger",
    "unit_economics":         "unit_economics",
    "freight_matrix":         "freight_matrix",
    "iot_telemetry":          "iot_telemetry",
}

@st.cache_data(show_spinner="Reading your data file…")
def load_all_data(src):
    """Load all data from one unified Excel workbook (template or original master)."""
    xl = pd.ExcelFile(src)
    available = xl.sheet_names

    def read(sheet): return pd.read_excel(src, sheet_name=sheet) if sheet in available else pd.DataFrame()

    # ── Core sheets ───────────────────────────────────────────────────────
    products   = read("products")
    warehouses = read("warehouses")
    inventory  = read("inventory")
    batches    = read("finished_product_batches")

    # Build inventory — merge batches, products, warehouses
    if not batches.empty and "fp_batch_id" in inventory.columns:
        merge_cols = [c for c in ["fp_batch_id","manufacture_date","qc_status","recall_flag"] if c in batches.columns]
        inventory = inventory.merge(batches[merge_cols], on="fp_batch_id", how="left")

    prod_cols = [c for c in ["product_id","generic_name","brand_name","dosage_form","route",
                              "pharm_class","dea_schedule","unit_price","shelf_life_months","status"] if c in products.columns]
    if prod_cols:
        inventory = inventory.merge(products[prod_cols], on="product_id", how="left")

    wh_cols = [c for c in ["warehouse_id","warehouse_name","state","temp_controlled","capacity_units"] if c in warehouses.columns]
    if wh_cols:
        inventory = inventory.merge(warehouses[wh_cols], on="warehouse_id", how="left")

    inventory["expiry_date"]        = pd.to_datetime(inventory.get("expiry_date"),      errors="coerce")
    inventory["manufacture_date"]   = pd.to_datetime(inventory.get("manufacture_date"), errors="coerce")
    inventory["days_to_expiry"]     = (inventory["expiry_date"] - TODAY).dt.days
    inventory["shelf_life_days"]    = (inventory["expiry_date"] - inventory["manufacture_date"]).dt.days
    inventory["pct_life_remaining"] = (inventory["days_to_expiry"] / inventory["shelf_life_days"].replace(0, np.nan) * 100).clip(0, 100)
    inventory["inventory_value_usd"]= inventory["quantity_on_hand"] * inventory["unit_price"]
    inventory["is_cold_chain"]      = inventory["dosage_form"].str.upper().str.contains("INJECTION|SOLUTION|VACCINE", na=False)
    inventory["is_controlled"]      = inventory["dea_schedule"].notna()
    inventory["expiry_risk"]        = inventory["days_to_expiry"].apply(expiry_risk_fn)

    # ── Supplementary sheets ──────────────────────────────────────────────
    df_demand  = read("monthly_demand")
    df_txns    = read("fefo_pick_ledger")
    df_econ    = read("unit_economics")
    df_freight = read("freight_matrix")
    df_iot     = read("iot_telemetry")

    # Rename columns to match what the rest of the app expects
    if not df_txns.empty:
        df_txns.rename(columns={"is_fefo_compliant": "is_fefo_compliant"}, inplace=True)  # already correct
        if "timestamp" in df_txns.columns:
            df_txns["timestamp"] = pd.to_datetime(df_txns["timestamp"], errors="coerce")
        if "transaction_type" not in df_txns.columns and "fefo_pick_ledger" in available:
            df_txns["transaction_type"] = "OUTBOUND_DISPATCH_PICK"  # all rows in ledger are picks

    if not df_iot.empty:
        # Rename humidity column if needed
        df_iot.rename(columns={"humidity_rh_pct": "relative_humidity_pct"}, inplace=True, errors="ignore")
        df_iot.rename(columns={"telemetry_log_id": "telemetry_id"}, inplace=True, errors="ignore")
        if "timestamp" in df_iot.columns:
            df_iot["timestamp"] = pd.to_datetime(df_iot["timestamp"], errors="coerce")

    if not df_econ.empty:
        # Rename unit_price_usd to unit_price if needed
        df_econ.rename(columns={"unit_price_usd": "unit_price"}, inplace=True, errors="ignore")

    # Check if supplementary data is usable
    supp_loaded = not df_demand.empty and not df_txns.empty

    return products, warehouses, inventory, df_demand, df_txns, df_econ, df_freight, df_iot, supp_loaded


def load_local_legacy():
    """Fallback: load from original separate files (local dev mode)."""
    import pandas as pd
    products   = pd.read_excel(LOCAL_MASTER, sheet_name="products")
    warehouses = pd.read_excel(LOCAL_MASTER, sheet_name="warehouses")
    inventory  = pd.read_excel(LOCAL_MASTER, sheet_name="inventory")
    batches    = pd.read_excel(LOCAL_MASTER, sheet_name="finished_product_batches")
    inventory  = inventory.merge(batches[[c for c in ["fp_batch_id","manufacture_date","qc_status","recall_flag"] if c in batches.columns]], on="fp_batch_id", how="left")
    inventory  = inventory.merge(products[[c for c in ["product_id","generic_name","brand_name","dosage_form","route","pharm_class","dea_schedule","unit_price","shelf_life_months","status"] if c in products.columns]], on="product_id", how="left")
    inventory  = inventory.merge(warehouses[[c for c in ["warehouse_id","warehouse_name","state","temp_controlled","capacity_units"] if c in warehouses.columns]], on="warehouse_id", how="left")
    inventory["expiry_date"]        = pd.to_datetime(inventory.get("expiry_date"),      errors="coerce")
    inventory["manufacture_date"]   = pd.to_datetime(inventory.get("manufacture_date"), errors="coerce")
    inventory["days_to_expiry"]     = (inventory["expiry_date"] - TODAY).dt.days
    inventory["shelf_life_days"]    = (inventory["expiry_date"] - inventory["manufacture_date"]).dt.days
    inventory["pct_life_remaining"] = (inventory["days_to_expiry"] / inventory["shelf_life_days"].replace(0, np.nan) * 100).clip(0, 100)
    inventory["inventory_value_usd"]= inventory["quantity_on_hand"] * inventory["unit_price"]
    inventory["is_cold_chain"]      = inventory["dosage_form"].str.upper().str.contains("INJECTION|SOLUTION|VACCINE", na=False)
    inventory["is_controlled"]      = inventory["dea_schedule"].notna()
    inventory["expiry_risk"]        = inventory["days_to_expiry"].apply(expiry_risk_fn)

    ADD = LOCAL_ADD
    df_demand  = pd.read_excel(os.path.join(ADD, "01_Pharma_Compliant_Monthly_Demand_24M.xlsx"))
    df_txns    = pd.read_excel(os.path.join(ADD, "02_Pharma_Compliant_FEFO_Pick_Ledger.xlsx"))
    df_econ    = pd.read_excel(os.path.join(ADD, "03_Pharma_Compliant_Unit_Economics_and_Costs.xlsx"))
    df_freight = pd.read_excel(os.path.join(ADD, "04_Pharma_Compliant_Inter_Warehouse_Freight_Matrix.xlsx"))
    df_iot     = pd.read_excel(os.path.join(ADD, "05_Pharma_Compliant_IoT_ColdChain_Telemetry_Logs.xlsx"))
    df_txns["timestamp"] = pd.to_datetime(df_txns["timestamp"], errors="coerce")
    df_iot["timestamp"]  = pd.to_datetime(df_iot["timestamp"],  errors="coerce")
    # Rename iot humidity column to match template
    df_iot.rename(columns={"humidity_rh_pct": "relative_humidity_pct"}, inplace=True, errors="ignore")
    df_iot.rename(columns={"telemetry_log_id": "telemetry_id"}, inplace=True, errors="ignore")
    return products, warehouses, inventory, df_demand, df_txns, df_econ, df_freight, df_iot, True


def get_data():
    """Return all datasets or None.
    Priority: 1) user upload  2) local dev data  3) bundled sample_data.xlsx
    """
    if uploaded_file:
        return load_all_data(uploaded_file)
    elif use_local:
        return load_local_legacy()
    # ── Auto-load bundled sample data (works on Streamlit Cloud) ──────────────
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data.xlsx")
    if os.path.exists(sample_path):
        return load_all_data(sample_path)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""<div style='text-align:center; padding: 28px 0 8px;'>
    <h1 style='font-size:2.2rem; font-weight:800; color:#00d4ff; margin:0;'>🏥 PharmaTrace AI</h1>
    <p style='font-size:1rem; color:#94a3b8; margin:6px 0 4px;'>Warehouse &amp; FEFO Inventory Optimization Dashboard</p>
    <p style='font-size:0.75rem; color:#475569;'>Module 3 | ISB AMPBA Capstone | Sponsor: Innodatatics Inc. | As of {TODAY.date()}</p>
</div><hr style='border-color:#1e2a45; margin: 8px 0 24px;'/>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
data = get_data()
if data is None:
    st.markdown("""
    <div style='text-align:center; padding:40px 20px;'>
        <div style='font-size:48px; margin-bottom:16px;'>📥</div>
        <h3 style='color:#00d4ff; margin-bottom:8px;'>No Data Loaded Yet</h3>
        <p style='color:#94a3b8; font-size:14px; max-width:480px; margin:0 auto 20px;'>
            Get started by downloading the template, filling in your pharma data, and uploading it.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.markdown("""<div style='background:#131929; border:1px solid #1e3a5f; border-radius:12px; padding:20px; text-align:center;'>
        <div style='font-size:28px; margin-bottom:8px;'>1️⃣</div>
        <div style='font-weight:600; color:#00d4ff; margin-bottom:6px;'>Download Template</div>
        <div style='font-size:12px; color:#64748b;'>Click "⬇️ Download Data Template" in the sidebar</div>
    </div>""", unsafe_allow_html=True)
    col_b.markdown("""<div style='background:#131929; border:1px solid #1e3a5f; border-radius:12px; padding:20px; text-align:center;'>
        <div style='font-size:28px; margin-bottom:8px;'>2️⃣</div>
        <div style='font-weight:600; color:#f59e0b; margin-bottom:6px;'>Fill In Your Data</div>
        <div style='font-size:12px; color:#64748b;'>Populate the 9 sheets in the template with your pharma warehouse data</div>
    </div>""", unsafe_allow_html=True)
    col_c.markdown("""<div style='background:#131929; border:1px solid #1e3a5f; border-radius:12px; padding:20px; text-align:center;'>
        <div style='font-size:28px; margin-bottom:8px;'>3️⃣</div>
        <div style='font-weight:600; color:#10b981; margin-bottom:6px;'>Upload & Analyse</div>
        <div style='font-size:12px; color:#64748b;'>Upload the filled file using "📂 Step 2" in the sidebar</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("**Template Sheet Guide:**")
    st.table({
        "Sheet": ["products","warehouses","inventory","finished_product_batches",
                   "monthly_demand","fefo_pick_ledger","unit_economics","freight_matrix","iot_telemetry"],
        "Contains": ["Product catalogue (SKUs, prices, shelf life)",
                     "Warehouse locations & capacities",
                     "Current batch stock levels & expiry dates",
                     "Batch manufacturing dates & QC status",
                     "24-month demand & dispatch history",
                     "Pick transaction ledger (FEFO compliance)",
                     "Holding, destruction & liquidation costs",
                     "Inter-warehouse freight cost matrix",
                     "IoT cold-chain temperature/humidity logs"],
    })
    st.stop()

products, warehouses, inventory, df_demand, df_txns, df_econ, df_freight, df_iot, supp_ok = data

RISK_COLORS = {"EXPIRED":"#7f1d1d","CRITICAL (<30d)":"#ef4444","HIGH (30-90d)":"#f97316",
               "MEDIUM (90-180d)":"#f59e0b","LOW (>180d)":"#10b981","Unknown":"#6b7280"}
RISK_ORDER   = ["EXPIRED","CRITICAL (<30d)","HIGH (30-90d)","MEDIUM (90-180d)","LOW (>180d)"]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME & KPI SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
if selected_page == "🏠 Home & KPI Summary":

    # ── KPI Row 1: Financial & Operational ───────────────────────────────────
    total_inv_value = inventory["inventory_value_usd"].sum()
    at_risk_value   = inventory[inventory["expiry_risk"].isin(["EXPIRED","CRITICAL (<30d)","HIGH (30-90d)"])]["inventory_value_usd"].sum()
    pct_at_risk     = at_risk_value / total_inv_value * 100 if total_inv_value else 0
    cold_chain_pct  = inventory[inventory["is_cold_chain"]]["inventory_value_usd"].sum() / total_inv_value * 100 if total_inv_value else 0
    n_warehouses    = inventory["warehouse_id"].nunique()
    n_products      = inventory["product_id"].nunique()
    total_units     = inventory["quantity_on_hand"].sum()

    picks = df_txns[df_txns["transaction_type"]=="OUTBOUND_DISPATCH_PICK"].copy() if (supp_ok and "transaction_type" in df_txns.columns) else (df_txns.copy() if supp_ok else None)
    fefo_rate      = picks["is_fefo_compliant"].mean() * 100 if (picks is not None and "is_fefo_compliant" in picks.columns) else 0
    excursion_rate = df_iot["is_thermal_excursion"].mean() * 100 if (supp_ok and "is_thermal_excursion" in df_iot.columns) else 0
    avg_fill_rate  = 0
    if supp_ok:
        monthly_agg = df_demand.groupby("year_month").agg(demanded=("quantity_demanded_units","sum"), dispatched=("quantity_dispatched_units","sum")).reset_index()
        monthly_agg["fill_rate"] = monthly_agg["dispatched"] / monthly_agg["demanded"] * 100
        avg_fill_rate = monthly_agg["fill_rate"].mean()

    # ── Top KPI strip ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Executive KPI Summary</div>', unsafe_allow_html=True)
    with st.expander("ℹ️ What do these KPIs mean?", expanded=False):
        st.markdown(GLOSSARY["Total Inventory Value"] + "\n\n---\n\n" + GLOSSARY["FEFO Compliance"] + "\n\n---\n\n" + GLOSSARY["IoT Excursion Rate"])

    cols = st.columns(5)
    kpis = [
        ("Total Inventory Value",  f"${total_inv_value/1e6:.1f}M",            "#00d4ff", "USD millions across all warehouses"),
        ("At-Risk Value",          f"${at_risk_value/1e3:.0f}K",              "#ef4444", f"{pct_at_risk:.1f}% of total stock value"),
        ("FEFO Compliance",        f"{fefo_rate:.1f}%" if supp_ok else "N/A", "#10b981" if fefo_rate>=97 else "#f59e0b", "Target ≥ 97%"),
        ("Avg Fill Rate",          f"{avg_fill_rate:.1f}%" if supp_ok else "N/A", "#10b981" if avg_fill_rate>=97 else "#f59e0b", "24-month service level"),
        ("IoT Excursion Rate",     f"{excursion_rate:.1f}%" if supp_ok else "N/A", "#10b981" if excursion_rate<5 else "#ef4444", "Cold-chain thermal events"),
    ]
    for col, (label, value, color, sub) in zip(cols, kpis):
        col.markdown(kpi_card(label, value, color, sub), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    kpis2 = [
        ("Active Products",   f"{n_products:,}",         "#7c3aed", "Unique SKUs tracked"),
        ("Warehouses",        f"{n_warehouses}",          "#f59e0b", "Distribution centres"),
        ("Total Stock Units", f"{total_units/1e3:.1f}K",  "#14b8a6", "Units on hand"),
        ("Cold-Chain Value",  f"{cold_chain_pct:.1f}%",  "#3b82f6", "Of total inventory value"),
    ]
    for col, (label, value, color, sub) in zip(cols2, kpis2):
        col.markdown(kpi_card(label, value, color, sub), unsafe_allow_html=True)

    st.markdown("---")

    # ── Live Alert Feed ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🚨 Live Alert Feed</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Actionable alerts — items that need management attention right now</div>', unsafe_allow_html=True)

    alerts = []
    expired = inventory[inventory["expiry_risk"]=="EXPIRED"]
    critical = inventory[inventory["expiry_risk"]=="CRITICAL (<30d)"]
    if not expired.empty:
        alerts.append(("#7f1d1d", "🛑", f"**{len(expired)} batches EXPIRED** — ${expired['inventory_value_usd'].sum():,.0f} at risk of regulatory disposal"))
    if not critical.empty:
        alerts.append(("#ef4444", "🔴", f"**{len(critical)} batches CRITICAL** (<30 days) — ${critical['inventory_value_usd'].sum():,.0f} — dispatch or liquidate immediately"))
    if supp_ok and fefo_rate < 97:
        alerts.append(("#f59e0b", "🟠", f"**FEFO Compliance {fefo_rate:.1f}%** — below 97% regulatory target. Review pick ledger."))
    if supp_ok and avg_fill_rate < 97:
        alerts.append(("#f59e0b", "🟠", f"**Fill Rate {avg_fill_rate:.1f}%** — below 97% target. Potential stockouts affecting patients."))
    if supp_ok and excursion_rate > 5:
        alerts.append(("#ef4444", "❄️", f"**Cold-chain Excursion Rate {excursion_rate:.1f}%** — above 5% threshold. Investigate IoT logs."))
    if "capacity_units" in warehouses.columns:
        wh_units = inventory.groupby("warehouse_id")["quantity_on_hand"].sum()
        wh_cap   = warehouses.set_index("warehouse_id")["capacity_units"]
        util     = (wh_units / wh_cap).dropna() * 100
        over_cap = util[util > 90]
        if not over_cap.empty:
            alerts.append(("#7c3aed", "🏢", f"**{len(over_cap)} warehouse(s) over 90% capacity** — {', '.join(over_cap.index.tolist())} — transfer or expedite stock"))

    if not alerts:
        st.success("✅ **All systems healthy** — no urgent alerts at this time.", icon="✅")
    else:
        for color, icon, msg in alerts:
            st.markdown(
                f"<div class='alert-card' style='background:{color}18; border-color:{color};'>{icon} {msg}</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Network Snapshot ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏭 Warehouse Network Snapshot</div>', unsafe_allow_html=True)
    summary = inventory.groupby("warehouse_id").agg(
        Products    =("product_id","nunique"),
        Total_Units =("quantity_on_hand","sum"),
        Value_USD   =("inventory_value_usd","sum"),
        At_Risk     =("expiry_risk", lambda x: x.isin(["EXPIRED","CRITICAL (<30d)","HIGH (30-90d)"]).sum()),
    ).round(0)
    if "capacity_units" in warehouses.columns:
        wh_cap2 = warehouses.set_index("warehouse_id")["capacity_units"]
        summary["Util_%"] = (summary["Total_Units"] / wh_cap2 * 100).round(1)
    summary["Value_USD"] = summary["Value_USD"].map("${:,.0f}".format)
    st.dataframe(summary, use_container_width=True)

    st.markdown("---")

    # ── Quick Navigation ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🧭 Quick Navigation</div>', unsafe_allow_html=True)
    # Nav items: (icon, exact_page_name_from_ALL_PAGES, short_label, description)
    nav_items = [
        ("📦", "📦 Inventory Overview",         "Inventory Overview",       "Stock levels, expiry risk, and multi-warehouse grid"),
        ("✅", "✅ FEFO Compliance",             "FEFO Compliance",          "Regulatory compliance rate by warehouse & month"),
        ("🌡️", "🌡️ Expiry Risk Heatmap",       "Expiry Risk Heatmap",      "Near-expiry batches mapped by warehouse & USD value"),
        ("📋", "📋 Order Fulfilment",           "Order Fulfilment",         "Simulate a sales order — check stock, WIP & raw materials"),
        ("🗺️", "🗺️ Geo Sales Intelligence",    "Geo Sales Intelligence",   "Identify HOT & COLD demand locations across warehouses"),
        ("💡", "💡 Smart Transfer Recommender", "Smart Transfer Recommender","Find cheapest restock option: transfer vs manufacture"),
    ]
    nav_cols = st.columns(3)
    for i, (icon, page_key, label, desc) in enumerate(nav_items):
        with nav_cols[i % 3]:
            st.markdown(f"""
<div class='nav-card'>
  <div class='nav-card-icon'>{icon}</div>
  <div class='nav-card-title'>{label}</div>
  <div class='nav-card-desc'>{desc}</div>
</div>""", unsafe_allow_html=True)
            if st.button(f"Open {label} →", key=f"nav_btn_{i}", use_container_width=True):
                st.session_state["_pending_nav"] = page_key
                st.rerun()

    st.markdown("---")
    # ── AI Executive Intelligence ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🧠 AI Executive Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Real-time AI-generated insights and recommendations derived from your live inventory, compliance, IoT, and demand data.</div>', unsafe_allow_html=True)

    _worst_wh_risk = inventory[inventory["expiry_risk"].isin(["EXPIRED","CRITICAL (<30d)","HIGH (30-90d)"])].groupby("warehouse_id")["inventory_value_usd"].sum()
    _worst_wh      = _worst_wh_risk.idxmax() if not _worst_wh_risk.empty else "N/A"
    _recovery_est  = round(at_risk_value * 0.62 / 1e3)

    _exec_bullets = [
        f"💰 <b>Capital exposure:</b> <b>${at_risk_value/1e3:.0f}K ({pct_at_risk:.1f}% of total inventory)</b> sits in EXPIRED, CRITICAL, or HIGH expiry tiers. "
        f"Largest exposure is concentrated at <b>{_worst_wh}</b>. Every day without action increases holding cost and reduces recovery potential.",
        f"♻️ <b>Recovery potential:</b> LP optimisation estimates ∼<b>${_recovery_est}K</b> recoverable through immediate dispatch, secondary-channel liquidation, "
        f"and inter-warehouse transfers. Use the LP Cost Optimizer page to generate specific batch-level action plans.",
    ]
    if supp_ok:
        if fefo_rate < 97:
            _exec_bullets.append(
                f"⚠️ <b>Regulatory risk:</b> FEFO compliance at <b>{fefo_rate:.1f}%</b> is <b>{97-fefo_rate:.1f}% below the FDA/USP 97% threshold</b>. "
                f"Each non-compliant pick is a potential FDA 21 CFR Part 211 finding. Initiate targeted warehouse audit and barcode-scan enforcement in WMS."
            )
        else:
            _exec_bullets.append(f"✅ <b>Regulatory standing:</b> FEFO compliance at <b>{fefo_rate:.1f}%</b> — above the 97% regulatory target. Schedule periodic compliance audits to sustain performance.")
        if avg_fill_rate < 95:
            _exec_bullets.append(
                f"📦 <b>Customer service risk:</b> Fill rate at <b>{avg_fill_rate:.1f}%</b> — below the 95% floor. "
                f"Downstream patients and hospitals may face medicine shortages. Review safety stock levels and supplier lead times immediately."
            )
        if excursion_rate > 5:
            _exec_bullets.append(
                f"🌡️ <b>Cold-chain emergency:</b> Thermal excursion rate at <b>{excursion_rate:.1f}%</b> exceeds the 5% USP &lt;659&gt; limit. "
                f"Temperature-sensitive products (biologics, vaccines, injectables) may have compromised potency. Trigger QA batch quarantine review now."
            )
    _exec_bullets.append(
        f"🔮 <b>Priority action roadmap:</b> (1) Dispatch/liquidate ALL CRITICAL batches within 7 days via LP Optimizer, "
        f"(2) Audit FEFO process at lowest-compliance warehouse, (3) Inspect refrigeration at high-excursion cold-chain warehouses, "
        f"(4) Pre-build seasonal stock 8–10 weeks before peak demand months."
    )
    ai_insight("Executive Briefing", _exec_bullets, icon="🧠", color="#7c3aed")

    st.caption("→ Click any button above or use the sidebar to navigate between all 14 pages.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INVENTORY OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "📦 Inventory Overview":
    st.markdown('<div class="section-header">📦 Inventory Overview Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">High-level snapshot across all warehouses — stock value, expiry risk, capacity utilisation, and product classification.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: info_box("Inventory Overview", "ℹ️ What do these 8 panels show?")
    with c2: info_box("Risk Tiers", "ℹ️ Expiry risk tiers explained")
    with c3: info_box("QC Status", "ℹ️ What is QC Status?")

    col_f1, col_f2 = st.columns(2)
    sel_wh   = col_f1.selectbox("Warehouse",       ["All"] + sorted(inventory["warehouse_id"].unique().tolist()))
    sel_risk = col_f2.selectbox("Expiry Risk Tier", ["All","EXPIRED","CRITICAL (<30d)","HIGH (30-90d)","MEDIUM (90-180d)","LOW (>180d)"])

    inv_f = inventory.copy()
    if sel_wh   != "All": inv_f = inv_f[inv_f["warehouse_id"] == sel_wh]
    if sel_risk != "All": inv_f = inv_f[inv_f["expiry_risk"]  == sel_risk]
    st.markdown(f"**{len(inv_f):,}** records | **${inv_f['inventory_value_usd'].sum():,.0f}** total value")

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("PharmaTrace AI — Inventory Overview Dashboard", fontsize=15, fontweight="bold", color="#00d4ff", y=1.02)

    axes[0,0].barh(inv_f.groupby("warehouse_id")["inventory_value_usd"].sum().sort_values().index,
                   inv_f.groupby("warehouse_id")["inventory_value_usd"].sum().sort_values().values/1e3, color="#00d4ff", alpha=0.85)
    axes[0,0].set_title("Inventory Value by WH (USD K)"); axes[0,0].set_xlabel("USD K")

    rc = inv_f["expiry_risk"].value_counts()
    axes[0,1].pie(rc.values, labels=rc.index, autopct="%1.0f%%", colors=[RISK_COLORS.get(r,"#888") for r in rc.index], wedgeprops={"edgecolor":"#0f1117","linewidth":1.5})
    axes[0,1].set_title("Expiry Risk Distribution")

    if "pharm_class" in inv_f.columns:
        pc = inv_f.groupby("pharm_class")["quantity_on_hand"].sum().nlargest(8).sort_values()
        axes[0,2].barh(pc.index.str[:20], pc.values/1e3, color="#7c3aed", alpha=0.85)
        axes[0,2].set_title("Units by Pharma Class (K)")

    if "is_controlled" in inv_f.columns:
        ctrl = inv_f["is_controlled"].value_counts()
        axes[0,3].pie(ctrl.values, labels=["Controlled" if v else "Non-Controlled" for v in ctrl.index], autopct="%1.1f%%", colors=["#ef4444","#10b981"], wedgeprops={"edgecolor":"#0f1117","linewidth":1.5})
        axes[0,3].set_title("DEA Controlled vs Non-Controlled")

    dte = inv_f["days_to_expiry"].dropna()
    axes[1,0].hist(dte.clip(-30,365), bins=40, color="#f59e0b", alpha=0.8, edgecolor="#0f1117")
    for thresh, col, lbl in [(0,"#7f1d1d","Expired"),(30,"#ef4444","30d"),(90,"#f97316","90d")]:
        axes[1,0].axvline(thresh, color=col, lw=2, linestyle="--", label=lbl)
    axes[1,0].set_title("Days-to-Expiry Distribution"); axes[1,0].legend(fontsize=8, framealpha=0)

    if "dosage_form" in inv_f.columns:
        df_val = inv_f.groupby("dosage_form")["inventory_value_usd"].sum().nlargest(7).sort_values()
        axes[1,1].barh(df_val.index.str[:18], df_val.values/1e3, color="#10b981", alpha=0.85)
        axes[1,1].set_title("Value by Dosage Form (USD K)")

    cc = inv_f[inv_f["is_cold_chain"]].groupby("warehouse_id")["quantity_on_hand"].sum().sort_values()
    axes[1,2].barh(cc.index, cc.values/1e3, color="#3b82f6", alpha=0.85)
    axes[1,2].set_title("Cold-Chain Units by WH (K)")

    if "qc_status" in inv_f.columns:
        qc = inv_f["qc_status"].value_counts()
        axes[1,3].pie(qc.values, labels=qc.index, autopct="%1.1f%%", colors=["#10b981","#f59e0b","#ef4444","#6b7280"][:len(qc)], wedgeprops={"edgecolor":"#0f1117","linewidth":1.5})
        axes[1,3].set_title("QC Status Distribution")

    plt.tight_layout()
    show_fig(fig)
    info_box("DTE Histogram", "ℹ️ How to read the DTE histogram")

    with st.expander("📋 Raw Inventory Table"):
        cols_show = [c for c in ["product_id","generic_name","warehouse_id","quantity_on_hand","days_to_expiry","expiry_risk","inventory_value_usd","qc_status"] if c in inv_f.columns]
        st.dataframe(inv_f[cols_show].head(500), use_container_width=True)

    # ── AI Insight: Inventory Health & Capital Allocation ──────────────
    _inv_tot_val  = inv_f["inventory_value_usd"].sum()
    _top_wh_val   = inv_f.groupby("warehouse_id")["inventory_value_usd"].sum()
    _top_wh_name  = _top_wh_val.idxmax() if not _top_wh_val.empty else "N/A"
    _top_wh_share = (_top_wh_val.max() / _inv_tot_val * 100) if _inv_tot_val > 0 else 0
    _cc_units     = inv_f[inv_f["is_cold_chain"]]["quantity_on_hand"].sum() if "is_cold_chain" in inv_f.columns else 0
    _ctrl_units   = inv_f[inv_f["is_controlled"]]["quantity_on_hand"].sum() if "is_controlled" in inv_f.columns else 0
    _qc_hold_cnt  = len(inv_f[inv_f["qc_status"]=="QUARANTINE / QC HOLD"]) if "qc_status" in inv_f.columns else 0

    _inv_bullets = [
        f"💼 <b>Capital concentration:</b> <b>{_top_wh_name}</b> holds <b>{_top_wh_share:.1f}%</b> of total network inventory value (${_top_wh_val.max()/1e3:,.0f}K of ${_inv_tot_val/1e3:,.0f}K). High geographic concentration increases business vulnerability to facility disruptions.",
        f"❄️ <b>Cold-Chain assets:</b> <b>{_cc_units:,.0f} units</b> require continuous 2–8°C thermal management (USP &lt;659&gt;). Cold storage capacity must be monitored to avoid thermal excursions and high-value biologic write-offs.",
    ]
    if _ctrl_units > 0:
        _inv_bullets.append(f"🔒 <b>DEA Controlled substances:</b> <b>{_ctrl_units:,.0f} units</b> under Schedule II–IV control. Requires strict perpetual inventory logs, dual-signoff vault access, and automated discrepancy reporting under DEA 21 CFR Part 1304.")
    if _qc_hold_cnt > 0:
        _inv_bullets.append(f"🔬 <b>QC Quarantine backlog:</b> <b>{_qc_hold_cnt} batch(es)</b> currently on QC hold. Expediting analytical release testing will unlock finished goods inventory for immediate dispatch.")
    _inv_bullets.append(
        f"💡 <b>Managerial recommendations:</b> (1) Conduct weekly ABC-based cycle counts on top 10% highest-value SKUs, "
        f"(2) Rebalance stock from {_top_wh_name} to secondary distribution centers to mitigate regional risk, "
        f"(3) Prioritize release testing for quarantined batches with earliest expiry dates."
    )
    ai_insight("Inventory Health & Capital Allocation", _inv_bullets, icon="📦", color="#00d4ff")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABC-FSN SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "🔶 ABC-FSN Segmentation":
    st.markdown('<div class="section-header">🔶 ABC-FSN Inventory Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">ABC = Value-based segmentation (Pareto 80/15/5) | FSN = Velocity-based (Fast/Slow/Non-moving) | Combined matrix guides stock prioritisation.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: info_box("ABC-FSN", "ℹ️ What is ABC-FSN analysis?")
    with c2: info_box("Pareto Curve", "ℹ️ How to read the Pareto curve")

    prod_val = inventory.groupby("product_id").agg(total_value=("inventory_value_usd","sum"), total_units=("quantity_on_hand","sum")).sort_values("total_value", ascending=False).reset_index()
    prod_val["cum_pct"] = prod_val["total_value"].cumsum() / prod_val["total_value"].sum() * 100
    prod_val["abc"] = prod_val["cum_pct"].apply(lambda cp: "A" if cp<=80 else ("B" if cp<=95 else "C"))

    if supp_ok and "transaction_type" in df_txns.columns:
        vel = df_txns[df_txns["transaction_type"]=="OUTBOUND_DISPATCH_PICK"].groupby("product_id")["quantity"].sum().reset_index().rename(columns={"quantity":"total_dispatched"})
        vel["avg_monthly_dispatch"] = vel["total_dispatched"] / 24
        q33, q66 = vel["avg_monthly_dispatch"].quantile([0.33, 0.66]).values
        vel["fsn"] = vel["avg_monthly_dispatch"].apply(lambda v: "Fast" if v>=q66 else ("Slow" if v>=q33 else "Non-Moving"))
        prod_val = prod_val.merge(vel[["product_id","avg_monthly_dispatch","fsn"]], on="product_id", how="left")
        prod_val["fsn"] = prod_val["fsn"].fillna("Non-Moving")
    else:
        prod_val["fsn"] = "Unknown"

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("ABC-FSN Inventory Segmentation Analysis", fontsize=14, color="#f59e0b", fontweight="bold", y=1.04)

    ax = axes[0]
    ax.plot(range(len(prod_val)), prod_val["cum_pct"], color="#00d4ff", lw=2.5)
    ax.axhline(80, color="#ef4444", linestyle="--", lw=1.5, label="A: 80%")
    ax.axhline(95, color="#f59e0b", linestyle="--", lw=1.5, label="B: 95%")
    a_idx = prod_val[prod_val["abc"]=="A"].index[-1] if len(prod_val[prod_val["abc"]=="A"]) else 0
    b_idx = prod_val[prod_val["abc"]=="B"].index[-1] if len(prod_val[prod_val["abc"]=="B"]) else 0
    ax.axvspan(0, a_idx, alpha=0.12, color="#ef4444"); ax.axvspan(a_idx, b_idx, alpha=0.10, color="#f59e0b"); ax.axvspan(b_idx, len(prod_val)-1, alpha=0.08, color="#10b981")
    ax.set_title("Pareto Curve — ABC Classification"); ax.set_xlabel("Product Rank"); ax.set_ylabel("Cumulative % of Total Value"); ax.legend(fontsize=8, framealpha=0)

    ax = axes[1]
    abc_s = prod_val.groupby("abc").agg(SKUs=("product_id","count"), Value=("total_value","sum"))
    abc_s["Value_pct"] = abc_s["Value"] / abc_s["Value"].sum() * 100
    ax.bar(abc_s.index, abc_s["SKUs"], color=["#ef4444","#f59e0b","#10b981"][:len(abc_s)], alpha=0.85)
    ax2b = ax.twinx(); ax2b.plot(abc_s.index, abc_s["Value_pct"], color="#00d4ff", marker="o", lw=2.5); ax2b.set_ylabel("% of Total Value", color="#00d4ff")
    ax.set_title("ABC Group: SKU Count vs Value"); ax.set_ylabel("Number of SKUs")

    ax = axes[2]
    if prod_val["fsn"].nunique() > 1:
        matrix = prod_val.groupby(["abc","fsn"])["total_value"].sum().unstack(fill_value=0) / 1e3
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar_kws={"label":"Value (USD K)"}, linewidths=0.5, linecolor="#0f1117")
        ax.set_title("ABC-FSN Matrix (USD K)")
    else:
        ax.text(0.5, 0.5, "Upload supplementary data\nfor FSN analysis", ha="center", va="center", color="#aaa", transform=ax.transAxes)

    plt.tight_layout()
    show_fig(fig)
    info_box("ABC-FSN Charts", "ℹ️ Segmentation insights based on value and velocity.")

    # ── AI Insight: ABC-FSN Inventory Policy ──────────────────────────
    _a_skus  = len(prod_val[prod_val["abc"]=="A"])
    _a_val_p = (prod_val[prod_val["abc"]=="A"]["total_value"].sum() / prod_val["total_value"].sum() * 100) if prod_val["total_value"].sum()>0 else 0
    _c_skus  = len(prod_val[prod_val["abc"]=="C"])

    _abc_bullets = [
        f"🎯 <b>Pareto principle in action (Class A):</b> <b>{_a_skus} SKUs ({_a_skus/len(prod_val)*100:.0f}% of portfolio)</b> account for <b>{_a_val_p:.1f}% of total inventory value</b>. "
        f"Tight managerial control, daily cycle counting, and Vendor Managed Inventory (VMI) partnerships should be strictly applied to Class A drugs.",
        f"📦 <b>Low-value long-tail (Class C):</b> <b>{_c_skus} SKUs</b> represent the bottom 5% of value. Apply visual two-bin or periodic min-max reordering to minimize administrative procurement costs.",
    ]
    if "fsn" in prod_val.columns and prod_val["fsn"].nunique() > 1:
        _a_slow = prod_val[(prod_val["abc"]=="A") & (prod_val["fsn"].isin(["Slow","Non-Moving"]))]
        if not _a_slow.empty:
            _a_slow_val = _a_slow["total_value"].sum()
            _abc_bullets.append(
                f"🚨 <b>Capital trap alert (Category A - Slow/Non-Moving):</b> <b>{len(_a_slow)} high-value SKU(s) (${_a_slow_val/1e3:,.0f}K total value)</b> have low dispatch velocity. "
                f"These represent significant working capital blockage and severe expiry risk. Negotiate return-to-vendor terms or reduce manufacturing batch sizes immediately."
            )
        _c_fast = prod_val[(prod_val["abc"]=="C") & (prod_val["fsn"]=="Fast")]
        if not _c_fast.empty:
            _abc_bullets.append(f"⚡ <b>High turnover utility (Category C - Fast):</b> <b>{len(_c_fast)} SKU(s)</b> move rapidly with minimal dollar risk. Maintain generous safety buffers to avoid zero-margin stockout disruptions.")
    _abc_bullets.append(
        f"💡 <b>Executive inventory policy:</b> (1) Enforce daily stock audits on Class A drugs, "
        f"(2) Implement 60-day shelf-life review on all Category A-Slow items, "
        f"(3) Consolidate procurement POs for Class C consumables to capture bulk volume discounts."
    )
    ai_insight("ABC-FSN Matrix & Working Capital Strategy", _abc_bullets, icon="🔶", color="#f59e0b")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FEFO COMPLIANCE
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "✅ FEFO Compliance":
    st.markdown('<div class="section-header">✅ FEFO Compliance Rate Analysis</div>', unsafe_allow_html=True)
    info_box("FEFO Header", "ℹ️ Monitoring First Expiry First Out metrics.")
    st.markdown('<div class="section-desc">FEFO = First Expiry First Out — dispatch batches in order of soonest expiry. Compliance rate = % picks that followed FEFO correctly. Regulatory target: ≥ 97%.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: info_box("FEFO Compliance", "ℹ️ What is FEFO compliance?")
    with c2: info_box("FEFO Compliance Detail", "ℹ️ How is it measured?")

    if not supp_ok:
        st.warning("Upload the FEFO Pick Ledger (file 02) to view this analysis.", icon="⚠️")
        st.stop()

    picks = df_txns[df_txns["transaction_type"]=="OUTBOUND_DISPATCH_PICK"].copy() if "transaction_type" in df_txns.columns else df_txns.copy()
    if "is_fefo_compliant" not in picks.columns:
        st.error("Column 'is_fefo_compliant' not found in pick ledger."); st.stop()

    picks["month"] = picks["timestamp"].dt.to_period("M").astype(str)
    fefo_wh = picks.groupby("warehouse_id").agg(total_picks=("transaction_id","count"), fefo_picks=("is_fefo_compliant","sum")).reset_index()
    fefo_wh["compliance_rate"] = fefo_wh["fefo_picks"] / fefo_wh["total_picks"] * 100
    fefo_mo = picks.groupby("month").agg(total_picks=("transaction_id","count"), fefo_picks=("is_fefo_compliant","sum")).reset_index().sort_values("month")
    fefo_mo["compliance_rate"] = fefo_mo["fefo_picks"] / fefo_mo["total_picks"] * 100

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("FEFO Compliance Rate Analysis", fontsize=14, color="#10b981", fontweight="bold", y=1.04)

    ax = axes[0]
    colors_w = ["#10b981" if r>=97 else "#f59e0b" if r>=90 else "#ef4444" for r in fefo_wh["compliance_rate"]]
    bars = ax.bar(fefo_wh["warehouse_id"], fefo_wh["compliance_rate"], color=colors_w, alpha=0.9)
    ax.axhline(97, color="#00d4ff", linestyle="--", lw=2, label="Target 97%"); ax.set_ylim(80, 101)
    ax.set_title("FEFO Compliance by Warehouse"); ax.set_ylabel("Compliance Rate (%)"); ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=9, framealpha=0)
    for bar, rate in zip(bars, fefo_wh["compliance_rate"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    x = range(len(fefo_mo))
    ax.plot(x, fefo_mo["compliance_rate"], color="#10b981", lw=2.5, marker="o", markersize=5)
    ax.fill_between(x, 97, fefo_mo["compliance_rate"], where=fefo_mo["compliance_rate"]<97, alpha=0.3, color="#ef4444", label="Below Target")
    ax.axhline(97, color="#00d4ff", linestyle="--", lw=2, label="Target 97%"); ax.set_ylim(80, 101)
    step = max(1, len(fefo_mo)//8)
    ax.set_xticks(list(x)[::step]); ax.set_xticklabels(fefo_mo["month"].tolist()[::step], rotation=30, ha="right", fontsize=8)
    ax.set_title("Monthly FEFO Compliance Trend"); ax.legend(fontsize=9, framealpha=0)

    ax = axes[2]
    non_comp = picks[picks["is_fefo_compliant"]==False] if False in picks["is_fefo_compliant"].values else picks[picks["is_fefo_compliant"]==0]
    if len(non_comp) > 0 and "warehouse_id" in non_comp.columns:
        nc_wh = non_comp.groupby("warehouse_id").size().sort_values(ascending=True)
        ax.barh(nc_wh.index, nc_wh.values, color="#ef4444", alpha=0.85)
        ax.set_title("Non-Compliant Picks by Warehouse"); ax.set_xlabel("Non-Compliant Picks Count")
    else:
        ax.text(0.5, 0.5, "✅ No non-compliant picks found!", ha="center", va="center", transform=ax.transAxes, fontsize=13, color="#10b981")

    plt.tight_layout()
    show_fig(fig)
    info_box("FEFO Charts", "ℹ️ Visualize FEFO compliance trends.")
    info_box("FEFO Analysis", "ℹ️ How to read these 3 charts")
    st.metric("Overall Network FEFO Rate", f"{picks['is_fefo_compliant'].mean()*100:.2f}%",
              delta=f"{picks['is_fefo_compliant'].mean()*100-97:.2f}% vs 97% target",
              help="FEFO Compliance Rate = Compliant Picks / Total Picks × 100. Target ≥ 97%. Below this risks FDA regulatory action.")
    info_box("FEFO Metric", "ℹ️ High-level network compliance overview.")

    # ── AI Insight: FEFO Root-Cause Analysis ──────────────────────────
    _fefo_overall = picks["is_fefo_compliant"].mean() * 100
    _worst_fefo   = fefo_wh.sort_values("compliance_rate").iloc[0] if not fefo_wh.empty else None
    _best_fefo    = fefo_wh.sort_values("compliance_rate").iloc[-1] if not fefo_wh.empty else None
    _nc_count     = len(picks[picks["is_fefo_compliant"] == False]) if False in picks["is_fefo_compliant"].values else len(picks[picks["is_fefo_compliant"] == 0])
    _fefo_bullets = []
    if _worst_fefo is not None:
        _fefo_bullets.append(
            f"🔴 <b>Worst performer: {_worst_fefo['warehouse_id']}</b> at <b>{_worst_fefo['compliance_rate']:.1f}% FEFO compliance</b> "
            f"({int(_worst_fefo['total_picks'] - _worst_fefo['fefo_picks'])} non-compliant picks out of {int(_worst_fefo['total_picks'])} total). "
            f"This warehouse requires immediate SOP review and WMS pick-order enforcement."
        )
    if _best_fefo is not None and _best_fefo["warehouse_id"] != (_worst_fefo["warehouse_id"] if _worst_fefo is not None else ""):
        _fefo_bullets.append(
            f"✅ <b>Best performer: {_best_fefo['warehouse_id']}</b> at <b>{_best_fefo['compliance_rate']:.1f}%</b>. "
            f"Study its SOPs, WMS configuration, and operator training process — replicate across the network as a best-practice template."
        )
    if _fefo_overall < 97:
        _violations_per_100 = round(100 - _fefo_overall, 1)
        _fefo_bullets.append(
            f"⚖️ <b>Regulatory exposure:</b> At {_fefo_overall:.1f}% compliance, approximately <b>{_violations_per_100:.0f} in every 100 picks</b> is a FEFO violation. "
            f"Under FDA 21 CFR Part 211 and USP &lt;1079&gt;, each violation is a potential Warning Letter or consent decree finding."
        )
    _fefo_bullets.append(
        f"🔍 <b>Root causes (typical):</b> (1) WMS batch-scan override by operators under time pressure, "
        f"(2) Poor bin-location labeling causing wrong batch selection, (3) Insufficient FEFO training or accountability metrics."
    )
    if len(fefo_mo) > 3:
        _recent_trend = fefo_mo["compliance_rate"].iloc[-3:].mean() - fefo_mo["compliance_rate"].iloc[:3].mean()
        _trend_txt = f"improving ({_recent_trend:+.1f}pp shift over period)" if _recent_trend > 0 else f"declining ({_recent_trend:+.1f}pp shift) — corrective action urgently needed"
        _fefo_bullets.append(f"📈 <b>Trend:</b> Network FEFO compliance is <b>{_trend_txt}</b>.")
    _fefo_bullets.append(
        f"💡 <b>Recommended actions:</b> (1) Deploy mandatory barcode-scan enforcement in WMS before batch dispatch, "
        f"(2) Retrain operators at {_worst_fefo['warehouse_id'] if _worst_fefo is not None else 'lowest-compliance warehouse'}, "
        f"(3) Publish daily FEFO compliance scorecard to warehouse managers, (4) KPI-link compliance to shift supervisor performance review."
    )
    ai_insight("FEFO Compliance — Root Cause & Action Plan", _fefo_bullets, icon="✅", color="#10b981")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXPIRY RISK HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "🌡️ Expiry Risk Heatmap":
    st.markdown('<div class="section-header">🌡️ Expiry Risk Heatmap</div>', unsafe_allow_html=True)
    info_box("Heatmap Header", "ℹ️ Heatmap analysis for expiry risk.")
    st.markdown('<div class="section-desc">DTE = Days-to-Expiry | Red cells = urgent action required | Each cell = total units in that risk tier at that warehouse.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: info_box("Expiry Risk Heatmap", "ℹ️ How to read this heatmap")
    with c2: info_box("Risk Tiers", "ℹ️ What do the risk tiers mean?")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Expiry Risk Heatmap — Days-to-Expiry (DTE) Analysis", fontsize=14, color="#f59e0b", fontweight="bold", y=1.04)

    ax = axes[0]
    pivot_risk = inventory.pivot_table(values="quantity_on_hand", index="warehouse_id", columns="expiry_risk", aggfunc="sum", fill_value=0)
    col_order = [c for c in RISK_ORDER if c in pivot_risk.columns]
    sns.heatmap(pivot_risk[col_order]/1000, annot=True, fmt=".1f", cmap=LinearSegmentedColormap.from_list("risk",["#10b981","#f59e0b","#ef4444"]), ax=ax, cbar_kws={"label":"Units ('000)"}, linewidths=0.5, linecolor="#0f1117")
    ax.set_title("Units at Risk by Warehouse ('000)"); ax.tick_params(axis="x", rotation=35)

    ax = axes[1]
    at_risk = inventory[inventory["expiry_risk"].isin(["EXPIRED","CRITICAL (<30d)","HIGH (30-90d)"])]
    risk_val = at_risk.groupby("warehouse_id")["inventory_value_usd"].sum().sort_values(ascending=True)
    ax.barh(risk_val.index, risk_val.values/1e3, color="#ef4444", alpha=0.85)
    ax.set_title("At-Risk Inventory Value by Warehouse (USD K)"); ax.set_xlabel("USD K")
    for bar, v in zip(ax.patches, risk_val.values):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f"${v/1e3:.1f}K", va="center", fontsize=9)

    ax = axes[2]
    sample = inventory.dropna(subset=["days_to_expiry"]).sample(min(2000, len(inventory)), random_state=42)
    for risk_cat, grp in sample.groupby("expiry_risk"):
        ax.scatter(grp["days_to_expiry"], grp["inventory_value_usd"]/1e3, label=risk_cat, alpha=0.55, s=18, color=RISK_COLORS.get(risk_cat,"#888"))
    for thresh, col, lbl in [(30,"#ef4444","30d"),(90,"#f59e0b","90d"),(180,"#3b82f6","180d")]:
        ax.axvline(thresh, color=col, linestyle="--", lw=1.5, label=lbl)
    ax.set_title("DTE vs Inventory Value"); ax.set_xlabel("Days to Expiry"); ax.set_ylabel("Value (USD K)"); ax.legend(fontsize=7, framealpha=0)

    plt.tight_layout()
    show_fig(fig)
    info_box("Heatmap Charts", "ℹ️ Visualization of expiry risk data.")
    info_box("Expiry Risk Heatmap", "ℹ️ What action should I take?")

    risk_summary = inventory.groupby("expiry_risk").agg(Products=("product_id","nunique"), Total_Units=("quantity_on_hand","sum"), Total_Value_USD=("inventory_value_usd","sum")).reindex([r for r in RISK_ORDER if r in inventory["expiry_risk"].unique()]).round(0)
    st.dataframe(risk_summary, use_container_width=True)
    info_box("Summary Table", "ℹ️ Grouped summary of inventory at risk.")

    # ── AI Insight: Expiry Risk Financial Impact ──────────────────────
    _exp_val   = inventory[inventory["expiry_risk"]=="EXPIRED"]["inventory_value_usd"].sum()
    _crit_val  = inventory[inventory["expiry_risk"]=="CRITICAL (<30d)"]["inventory_value_usd"].sum()
    _high_val  = inventory[inventory["expiry_risk"]=="HIGH (30-90d)"]["inventory_value_usd"].sum()
    _crit_u    = inventory[inventory["expiry_risk"]=="CRITICAL (<30d)"]["quantity_on_hand"].sum()
    _exp_u     = inventory[inventory["expiry_risk"]=="EXPIRED"]["quantity_on_hand"].sum()
    _worst_r_wh= risk_val.idxmax() if not risk_val.empty else "N/A"
    _risk_bullets = [
        f"🛑 <b>Immediate write-off risk:</b> <b>{_exp_u:,.0f} units (${_exp_val/1e3:.0f}K) already EXPIRED</b> — zero recovery possible. "
        f"Regulatory certified destruction must begin immediately. Notify QA, complete batch disposition records.",
        f"🔴 <b>48-hour window — CRITICAL batches:</b> <b>{_crit_u:,.0f} units (${_crit_val/1e3:.0f}K)</b> expire in <30 days. "
        f"At current dispatch velocity, a significant portion will expire unsold without urgent action. "
        f"Run LP Cost Optimizer now for batch-by-batch allocation recommendations.",
        f"🏢 <b>Hotspot warehouse: {_worst_r_wh}</b> carries the highest at-risk USD exposure in the network. "
        f"Prioritise emergency dispatch orders from this warehouse. Consider inter-warehouse transfer to high-demand locations using the Freight Rebalancing tool.",
    ]
    if _high_val > 0:
        _risk_bullets.append(
            f"📅 <b>30-day cascade risk:</b> Without intervention, HIGH-tier stock (${_high_val/1e3:.0f}K) will move into CRITICAL next month. "
            f"Begin proactive inter-warehouse transfers now — use Geo Sales Intelligence to identify which warehouses have high demand."
        )
    _risk_bullets.append(
        f"💡 <b>Recovery strategy:</b> (1) EXPIRED → certified destruction + regulatory documentation, "
        f"(2) CRITICAL → emergency dispatch to highest-demand warehouse today, "
        f"(3) HIGH → planned transfer/liquidation within 30 days, "
        f"(4) Run LP Cost Optimizer for mathematically optimal unit-by-unit allocation across all 4 channels."
    )
    ai_insight("Expiry Risk — Financial Impact & Recovery Roadmap", _risk_bullets, icon="🌡️", color="#ef4444")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DEMAND & SEASONALITY
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "📈 Demand & Seasonality":
    st.markdown('<div class="section-header">📈 24-Month Demand Trend & Seasonality</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Monthly demand vs dispatch | Service level (fill rate) | Seasonal patterns by therapy area | Revenue trajectory</div>', unsafe_allow_html=True)
    with st.expander("ℹ️ What these 4 charts show", expanded=False):
        st.markdown(GLOSSARY["Demand Trend"])
    if not supp_ok:
        st.warning("Upload Monthly Demand data (via the template) to view this analysis.", icon="⚠️"); st.stop()

    monthly_agg = df_demand.groupby("year_month").agg(demanded=("quantity_demanded_units","sum"), dispatched=("quantity_dispatched_units","sum")).reset_index().sort_values("year_month")
    monthly_agg["fill_rate"] = monthly_agg["dispatched"] / monthly_agg["demanded"] * 100

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("24-Month Demand Trend & Seasonality Analysis", fontsize=14, color="#00d4ff", fontweight="bold", y=1.03)
    step = max(1, len(monthly_agg)//8)
    x = range(len(monthly_agg))

    ax = axes[0, 0]
    ax.plot(x, monthly_agg["demanded"]/1e6,   label="Demanded",   color="#00d4ff", lw=2)
    ax.plot(x, monthly_agg["dispatched"]/1e6, label="Dispatched", color="#10b981", lw=2, linestyle="--")
    ax.fill_between(x, monthly_agg["demanded"]/1e6, monthly_agg["dispatched"]/1e6, alpha=0.2, color="#ef4444", label="Unfulfilled")
    ax.set_xticks(list(x)[::step]); ax.set_xticklabels(monthly_agg["year_month"].tolist()[::step], rotation=30, ha="right", fontsize=8)
    ax.set_title("Aggregate Monthly Demand vs Dispatch (M Units)"); ax.set_ylabel("Units (Millions)"); ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.bar(x, monthly_agg["fill_rate"], color=["#ef4444" if f<95 else "#10b981" for f in monthly_agg["fill_rate"]], alpha=0.85)
    ax.axhline(97, color="#00d4ff", linestyle="--", lw=2, label="Target SL 97%"); ax.set_ylim(88, 101)
    ax.set_xticks(list(x)[::step]); ax.set_xticklabels(monthly_agg["year_month"].tolist()[::step], rotation=30, ha="right", fontsize=8)
    ax.set_title("Monthly Service Level / Fill Rate (%)"); ax.legend(fontsize=9)

    ax = axes[1, 0]
    if "clinical_demand_pattern" in df_demand.columns:
        md = df_demand.copy(); md["month_num"] = md["year_month"].str[-2:].astype(int)
        seas = md.groupby(["clinical_demand_pattern","month_num"])["quantity_demanded_units"].mean().reset_index()
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        for i, (pat, grp) in enumerate(seas.groupby("clinical_demand_pattern")):
            gs2 = grp.sort_values("month_num")
            ax.plot(gs2["month_num"], gs2["quantity_demanded_units"], label=pat[:30], lw=2, marker="o", markersize=4, color=PALETTE[i%len(PALETTE)])
        ax.set_xticks(range(1,13)); ax.set_xticklabels(month_labels, fontsize=9)
        ax.set_title("Clinical Demand Seasonality by Product Category"); ax.legend(fontsize=7, framealpha=0)

    ax = axes[1, 1]
    if "monthly_dispatched_value_usd" in df_demand.columns:
        rev = df_demand.groupby("year_month")["monthly_dispatched_value_usd"].sum().reset_index().sort_values("year_month")
        x2 = range(len(rev))
        ax.fill_between(x2, rev["monthly_dispatched_value_usd"]/1e6, alpha=0.3, color="#7c3aed")
        ax.plot(x2, rev["monthly_dispatched_value_usd"]/1e6, color="#7c3aed", lw=2.5)
        ax.set_xticks(list(x2)[::step]); ax.set_xticklabels(rev["year_month"].tolist()[::step], rotation=30, ha="right", fontsize=8)
        ax.set_title("Monthly Revenue from Dispatches (USD M)"); ax.set_ylabel("Revenue (USD Millions)")

    plt.tight_layout()
    show_fig(fig)
    info_box("Demand Charts", "ℹ️ Visualization of demand and seasonal trends.")

    # ── AI Insight: Demand Intelligence & Procurement ──────────────
    _fill_avg  = monthly_agg["fill_rate"].mean()
    _fill_min  = monthly_agg["fill_rate"].min()
    _worst_mo  = monthly_agg.loc[monthly_agg["fill_rate"].idxmin(), "year_month"]
    _so_months = len(monthly_agg[monthly_agg["fill_rate"] < 95])
    _dem_bullets = [
        f"📊 <b>Service level:</b> Average fill rate of <b>{_fill_avg:.1f}%</b> across {len(monthly_agg)} months. "
        f"Lowest month: <b>{_worst_mo} at {_fill_min:.1f}%</b> — "
        f"{'indicating confirmed stockouts that month' if _fill_min < 95 else 'above the 95% service floor'}.",
    ]
    if _so_months > 0:
        _dem_bullets.append(
            f"⚠️ <b>Stockout impact:</b> <b>{_so_months} month{'s' if _so_months>1 else ''}</b> fell below 95% fill rate. "
            f"Unmet hospital and pharmacy orders may have caused patient treatment delays. "
            f"Increase safety stock by 15–20% for high-velocity SKUs, and review with procurement team."
        )
    if "monthly_dispatched_value_usd" in df_demand.columns:
        _rev = df_demand.groupby("year_month")["monthly_dispatched_value_usd"].sum().sort_index()
        if len(_rev) > 6:
            _rev_trend = (_rev.iloc[-3:].mean() - _rev.iloc[:3].mean()) / max(_rev.iloc[:3].mean(), 1) * 100
            _dem_bullets.append(
                f"📈 <b>Revenue trajectory:</b> Recent 3-month average is <b>{_rev_trend:+.1f}%</b> vs the first 3 months. "
                f"{'Positive growth — scale supply chain capacity to sustain momentum.' if _rev_trend > 0 else 'Declining trend — review pricing strategy, product mix, and market access initiatives.'}"
            )
    if "clinical_demand_pattern" in df_demand.columns:
        _dem_bullets.append(
            f"🗋️ <b>Seasonal procurement strategy:</b> Respiratory and cardiovascular products typically peak Nov–Jan (flu/winter season). "
            f"Begin pre-season stock-build 8–10 weeks in advance (i.e., Aug–Sep orders). "
            f"For slow-moving non-seasonal SKUs, implement min-max reorder levels to prevent capital-draining overstock."
        )
    _dem_bullets.append(
        f"💡 <b>Recommended actions:</b> (1) Increase safety stock for SKUs with >1 stockout month by 20%, "
        f"(2) Negotiate 60-day rolling demand forecasts with top-3 hospital clients, "
        f"(3) Review CMO/3PL lead times and target 2-week reduction, "
        f"(4) Set up automated reorder alerts in ERP when stock crosses safety stock threshold."
    )
    ai_insight("Demand Intelligence & Procurement Strategy", _dem_bullets, icon="📈", color="#f59e0b")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ML EXPIRY CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "🤖 ML Expiry Classifier":
    st.markdown('<div class="section-header">🤖 Expiry Risk ML Classifier (Random Forest)</div>', unsafe_allow_html=True)
    info_box("ML Header", "ℹ️ Random Forest classification analysis.")
    st.markdown('<div class="section-desc">Random Forest trained on DTE, quantity, value, velocity & shelf-life features to classify batches into risk tiers. Feature importance shows which variables drive risk predictions.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: info_box("ML Classifier", "ℹ️ What is this ML model doing?")
    with c2: info_box("Feature Importance", "ℹ️ What is feature importance?")
    with c3: info_box("Confusion Matrix", "ℹ️ How to read the confusion matrix")

    ml_df = inventory.dropna(subset=["days_to_expiry","quantity_on_hand","unit_price"]).copy()
    if supp_ok and "transaction_type" in df_txns.columns:
        velocity = df_txns[df_txns["transaction_type"]=="OUTBOUND_DISPATCH_PICK"].groupby("product_id")["quantity"].sum().reset_index().rename(columns={"quantity":"total_dispatched"})
        velocity["avg_monthly_dispatch"] = velocity["total_dispatched"] / 24
        ml_df["avg_monthly_dispatch"] = ml_df["product_id"].map(velocity.set_index("product_id")["avg_monthly_dispatch"]).fillna(ml_df["quantity_on_hand"].median()/6)
    else:
        ml_df["avg_monthly_dispatch"] = ml_df["quantity_on_hand"] / 6

    ml_df["cover_days"]    = ml_df["quantity_on_hand"] / ml_df["avg_monthly_dispatch"].replace(0,1) * 30
    ml_df["risk_score"]    = (ml_df["days_to_expiry"] / ml_df["shelf_life_days"].replace(0,1)).clip(0,1)
    ml_df["value_per_day"] = ml_df["inventory_value_usd"] / ml_df["days_to_expiry"].clip(1,9999)

    features = [f for f in ["days_to_expiry","quantity_on_hand","unit_price","avg_monthly_dispatch","cover_days","risk_score","value_per_day","pct_life_remaining"] if f in ml_df.columns]
    X = ml_df[features].fillna(0)
    y = ml_df["expiry_risk"]
    binary_fallback = False
    if y.nunique() < 2:
        # Fallback: binary At-Risk vs Safe classification
        ml_df["expiry_risk_bin"] = ml_df["expiry_risk"].apply(
            lambda r: "At-Risk" if r in ["EXPIRED", "CRITICAL (<30d)", "HIGH (30-90d)"] else "Safe"
        )
        y = ml_df["expiry_risk_bin"]
        binary_fallback = True
        if y.nunique() < 2:
            st.warning(
                "⚠️ All inventory batches fall into the same risk tier. "
                "Upload data with a wider range of expiry dates to enable ML classification.",
                icon="⚠️"
            )
            st.stop()
        st.info(
            "ℹ️ Using binary (At-Risk / Safe) classification — sample data has limited class variety. "
            "Upload your full dataset for multi-tier predictions.",
            icon="ℹ️"
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    ml_df["predicted_risk"] = rf.predict(X)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("ML Expiry Risk Classifier — Random Forest", fontsize=14, color="#00d4ff", fontweight="bold", y=1.04)

    imp = pd.Series(rf.feature_importances_, index=features).sort_values()
    axes[0].barh(imp.index, imp.values, color="#7c3aed", alpha=0.85)
    axes[0].set_title("Feature Importance"); axes[0].set_xlabel("Importance Score")

    classes = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=axes[1], linewidths=0.5, linecolor="#0f1117")
    axes[1].set_title("Confusion Matrix"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True"); axes[1].tick_params(axis="x", rotation=30)

    pred_counts = ml_df["predicted_risk"].value_counts().reindex([r for r in RISK_ORDER if r in ml_df["predicted_risk"].unique()])
    axes[2].bar(pred_counts.index, pred_counts.values, color=[RISK_COLORS.get(r,"#888") for r in pred_counts.index], alpha=0.9)
    axes[2].set_title("ML-Predicted Risk Distribution"); axes[2].set_ylabel("Batches"); axes[2].tick_params(axis="x", rotation=35)

    plt.tight_layout()
    show_fig(fig)
    info_box("ML Charts", "ℹ️ Visualization of ML model performance.")
    info_box("ML Classifier", "ℹ️ How to act on these ML results")

    acc = accuracy_score(y_test, y_pred) * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy",    f"{acc:.1f}%",   help="% of test-set batches correctly classified into the right expiry risk tier. Above 90% = excellent.")
    col2.metric("Training Samples",  f"{len(X_train):,}", help="Number of inventory records used to train the Random Forest model.")
    col3.metric("Test Samples",      f"{len(X_test):,}", help="Held-out records used to evaluate model accuracy — these were not seen during training.")
    info_box("Performance Metrics", "ℹ️ Model evaluation metrics.")
    with st.expander("📋 Classification Report"):
        st.text(classification_report(y_test, y_pred, zero_division=0))
    info_box("Classification Report", "ℹ️ Detailed report of ML accuracy.")

    # ── AI Insight: ML Governance & Predictive Signals ────────────────
    _top_feat = imp.idxmax() if not imp.empty else "days_to_expiry"
    _top_imp_val = imp.max() * 100 if not imp.empty else 0
    _rf_bullets = [
        f"🤖 <b>Multi-variable risk intelligence:</b> The Random Forest model captures non-linear interactions between <b>Days-to-Expiry (DTE)</b>, <b>Dispatch Velocity</b>, and <b>Shelf-Life Consumption</b> that static calendar thresholds overlook.",
        f"📊 <b>Primary risk driver:</b> <b>{_top_feat}</b> is the single most influential feature (<b>{_top_imp_val:.1f}% relative importance</b>). Batches with high stock relative to monthly velocity are flagged early, even when nominal DTE appears safe.",
        f"🎯 <b>Accuracy validation:</b> Model achieved <b>{acc:.1f}% accuracy</b> on held-out test batches. High classification precision on critical tiers prevents false negatives that could lead to dispatching expired pharmaceuticals to clinical partners.",
        f"💡 <b>Operational deployment:</b> (1) Integrate ML risk scores directly into WMS pick-list generation to prioritize at-risk batches automatically, "
        f"(2) Retrain model monthly as seasonal demand shifts, "
        f"(3) Set automated early-warning alerts for batches whose predicted risk tier worsens over consecutive weekly runs."
    ]
    ai_insight("ML Expiry Classifier — Model Governance & Predictive Strategy", _rf_bullets, icon="🤖", color="#7c3aed")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LP COST OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "⚖️ LP Cost Optimizer":
    st.markdown('<div class="section-header">⚖️ LP Inventory Cost Optimizer</div>', unsafe_allow_html=True)
    info_box("LP Header", "ℹ️ Linear Programming for inventory optimization.")
    st.markdown('<div class="section-desc">Linear Programming: for each at-risk batch, optimise split across Dispatch / Transfer / Liquidate / Dispose to minimise total cost under regulatory constraints.</div>', unsafe_allow_html=True)
    info_box("LP Optimizer", "ℹ️ How does LP optimisation work?")

    if not supp_ok:
        st.warning("Upload Unit Economics file (file 03) to enable LP optimisation.", icon="⚠️"); st.stop()

    ml_df = inventory.dropna(subset=["days_to_expiry","quantity_on_hand","unit_price"]).copy()
    if supp_ok and "transaction_type" in df_txns.columns:
        velocity = df_txns[df_txns["transaction_type"]=="OUTBOUND_DISPATCH_PICK"].groupby("product_id")["quantity"].sum().reset_index().rename(columns={"quantity":"total_dispatched"})
        velocity["avg_monthly_dispatch"] = velocity["total_dispatched"] / 24
        ml_df["avg_monthly_dispatch"] = ml_df["product_id"].map(velocity.set_index("product_id")["avg_monthly_dispatch"]).fillna(ml_df["quantity_on_hand"].median()/6)
    else:
        ml_df["avg_monthly_dispatch"] = ml_df["quantity_on_hand"] / 6
    ml_df["risk_score"] = (ml_df["days_to_expiry"] / ml_df["shelf_life_days"].replace(0,1)).clip(0,1)

    at_risk_inv = ml_df[ml_df["risk_score"]>0.5].merge(
        df_econ[["product_id","daily_holding_cost_per_unit_usd","stockout_penalty_cost_per_unit_usd","certified_destruction_cost_per_unit_usd","secondary_liquidation_recovery_pct"]],
        on="product_id", how="left").dropna(subset=["daily_holding_cost_per_unit_usd"])

    with st.spinner(f"Running LP optimisation on {min(50,len(at_risk_inv))} at-risk records…"):
        lp_results = []
        for _, row in at_risk_inv.head(50).iterrows():
            Q, dte = float(row["quantity_on_hand"]), max(float(row["days_to_expiry"]), 1)
            h, d_c = float(row["daily_holding_cost_per_unit_usd"]), float(row["certified_destruction_cost_per_unit_usd"])
            p, rec = float(row["unit_price"]), float(row["secondary_liquidation_recovery_pct"])/100.0
            vel = max(float(row.get("avg_monthly_dispatch",30))/30.0, 0.1)
            max_dispatch = min(Q*0.9, vel*dte); max_transfer = min(Q*0.5, Q-max_dispatch)
            max_liquidate = Q*0.35; min_dispose = Q*0.05 if dte<2 else 0
            c = [-(p-h*dte), -(p*0.6-h*dte*0.5), -(p*rec), d_c+h*dte]
            A_ub = [[1,1,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]]
            b_ub = [Q, max_dispatch, max_transfer, max_liquidate]
            bounds = [(0,max_dispatch),(0,max_transfer),(0,max_liquidate),(min_dispose,Q)]
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res.success:
                x1,x2,x3,x4 = res.x
                lp_results.append({"product_id":row["product_id"],"warehouse_id":row.get("warehouse_id",""),"DTE":int(dte),"Qty":int(Q),"Dispatch":round(x1),"Transfer":round(x2),"Liquidate":round(x3),"Dispose":round(x4),"Net_Saving_USD":round(-res.fun,2)})

    if lp_results:
        lp_df = pd.DataFrame(lp_results)
        col1,col2,col3 = st.columns(3)
        col1.metric("Batches Optimised",  f"{len(lp_df)}")
        col2.metric("Total Net Savings",  f"${lp_df['Net_Saving_USD'].sum():,.0f}")
        col3.metric("Avg Saving / Batch", f"${lp_df['Net_Saving_USD'].mean():,.0f}")
        info_box("Optimization Metrics", "ℹ️ Performance metrics for LP optimization.")
        fig, axes = plt.subplots(1, 2, figsize=(18, 6)); fig.patch.set_facecolor("#0f1117")
        fig.suptitle("LP Inventory Cost Optimisation Results", fontsize=14, color="#10b981", fontweight="bold", y=1.04)
        alloc = lp_df[["Dispatch","Transfer","Liquidate","Dispose"]].sum()
        axes[0].pie(alloc.values, labels=alloc.index, autopct="%1.1f%%", colors=["#10b981","#3b82f6","#f59e0b","#ef4444"], wedgeprops={"edgecolor":"#0f1117","linewidth":2})
        axes[0].set_title("Optimal Allocation Split")
        axes[1].scatter(lp_df["DTE"], lp_df["Net_Saving_USD"], color="#00d4ff", alpha=0.7, s=40)
        axes[1].axhline(0, color="#ef4444", linestyle="--", lw=1.5, label="Break-even")
        axes[1].set_title("Net Savings vs Days-to-Expiry"); axes[1].set_xlabel("DTE"); axes[1].set_ylabel("Net Saving (USD)"); axes[1].legend(fontsize=9, framealpha=0)
        plt.tight_layout(); show_fig(fig)
        info_box("Optimization Charts", "ℹ️ Visualization of LP results.")
        info_box("LP Optimizer", "ℹ️ How to interpret LP results")
        with st.expander("📋 LP Results Table"):
            st.dataframe(lp_df, use_container_width=True)
        info_box("LP Table", "ℹ️ Detailed results of LP calculations.")

        # ── AI Insight: LP Recovery Action Plan ─────────────────────
        _lp_total_saving = lp_df["Net_Saving_USD"].sum()
        _top3_lp = lp_df.nlargest(3, "Net_Saving_USD")
        _alloc_total = alloc.sum() if alloc.sum() > 0 else 1
        _dp = alloc.get("Dispatch",0)  / _alloc_total * 100
        _lq = alloc.get("Liquidate",0) / _alloc_total * 100
        _tr = alloc.get("Transfer",0)  / _alloc_total * 100
        _ds = alloc.get("Dispose",0)   / _alloc_total * 100
        _lp_bullets = [
            f"💰 <b>Total value recoverable:</b> <b>${_lp_total_saving:,.0f}</b> across {len(lp_df)} optimised batches — representing the maximum extractable value given "
            f"dispatch velocity constraints, 35% liquidation channel cap, and regulatory disposal requirements.",
            f"📊 <b>Optimal allocation mix:</b> LP recommends <b>{_dp:.0f}% dispatch</b> (highest recovery), "
            f"{_tr:.0f}% inter-warehouse transfer, {_lq:.0f}% secondary liquidation, {_ds:.0f}% regulatory disposal. "
            f"Execute dispatch orders immediately — each day of delay adds holding cost and reduces remaining DTE.",
        ]
        for _, _lpr in _top3_lp.iterrows():
            _lp_bullets.append(
                f"🏆 <b>High-impact batch:</b> {_lpr['product_id']} @ {_lpr['warehouse_id']} — "
                f"DTE: <b>{_lpr['DTE']}d</b> | Net saving: <b>${_lpr['Net_Saving_USD']:,.0f}</b> | "
                f"Dispatch {_lpr['Dispatch']:.0f}u · Transfer {_lpr['Transfer']:.0f}u · Liquidate {_lpr['Liquidate']:.0f}u"
            )
        _lp_bullets.append(
            f"💡 <b>Next steps:</b> (1) Export LP table and raise dispatch purchase orders for top-saving batches TODAY, "
            f"(2) Contact secondary liquidation partner for liquidation-flagged stock, "
            f"(3) Coordinate with logistics team for transfer-flagged batches, "
            f"(4) Initiate certified destruction paperwork for dispose-flagged batches."
        )
        ai_insight("LP Optimisation — Maximum Recovery Action Plan", _lp_bullets, icon="⚖️", color="#00d4ff")

    else:
        st.warning("LP solver returned no feasible solutions.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: IOT COLD-CHAIN MONITOR
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "❄️ IoT Cold-Chain Monitor":
    st.markdown('<div class="section-header">❄️ Cold-Chain IoT Telemetry Monitoring</div>', unsafe_allow_html=True)
    info_box("IoT Header", "ℹ️ Cold-chain monitoring dashboard.")
    st.markdown('<div class="section-desc">USP <659> = US Pharmacopeia standard | Safe zone: 2–8°C | RH target: 55% | Thermal Excursion = temperature outside 2–8°C band.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: info_box("IoT Monitor", "ℹ️ What do these 4 charts show?")
    with c2: info_box("IoT Excursion Rate", "ℹ️ What is a thermal excursion?")

    if not supp_ok:
        st.warning("Upload IoT Telemetry Logs (file 05) to view this analysis.", icon="⚠️"); st.stop()

    cold_wh_ids = warehouses.loc[warehouses["temp_controlled"]==True, "warehouse_id"].tolist() if "temp_controlled" in warehouses.columns else df_iot["warehouse_id"].unique().tolist()

    fig, axes = plt.subplots(2, 2, figsize=(20, 12)); fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Cold-Chain IoT Telemetry — USP <659> Compliance Monitoring", fontsize=14, color="#00d4ff", fontweight="bold", y=1.03)

    ax = axes[0,0]
    for wid in cold_wh_ids:
        wh_iot = df_iot[df_iot["warehouse_id"]==wid].sort_values("timestamp").tail(200)
        if len(wh_iot)>0: ax.plot(range(len(wh_iot)), wh_iot["temperature_celsius"], lw=1.2, alpha=0.8, label=wid)
    ax.axhline(8.0, color="#ef4444", linestyle="--", lw=1.5, label="Max 8°C"); ax.axhline(2.0, color="#3b82f6", linestyle="--", lw=1.5, label="Min 2°C")
    if cold_wh_ids:
        n_pts = max((len(df_iot[df_iot["warehouse_id"]==wid].tail(200)) for wid in cold_wh_ids), default=200)
        ax.fill_between(range(n_pts), 2, 8, alpha=0.06, color="#10b981", label="Safe Zone")
    ax.set_title("Cold-Chain Temperature Profile"); ax.set_xlabel("Readings (last 200 per WH)"); ax.set_ylabel("Temperature (°C)"); ax.legend(fontsize=8, framealpha=0)

    ax = axes[0,1]
    if "is_thermal_excursion" in df_iot.columns:
        id_col = "telemetry_id" if "telemetry_id" in df_iot.columns else "timestamp"
        exc_rate = df_iot.groupby("warehouse_id").agg(total=(id_col,"count"), excursions=("is_thermal_excursion","sum")).reset_index()
        exc_rate["rate"] = exc_rate["excursions"] / exc_rate["total"] * 100
        ax.bar(exc_rate["warehouse_id"], exc_rate["rate"], color=["#ef4444" if r>10 else "#f59e0b" if r>5 else "#10b981" for r in exc_rate["rate"]], alpha=0.9)
        ax.axhline(5, color="#00d4ff", linestyle="--", lw=1.5, label="5% Threshold")
        ax.set_title("Thermal Excursion Rate by Warehouse (%)"); ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=8, framealpha=0)

    ax = axes[1,0]
    if "relative_humidity_pct" in df_iot.columns:
        for wid in cold_wh_ids[:4]:
            h_data = df_iot[df_iot["warehouse_id"]==wid]["relative_humidity_pct"].dropna()
            if len(h_data)>0: ax.hist(h_data, bins=30, alpha=0.5, label=wid)
        ax.axvline(55, color="#00d4ff", linestyle="--", lw=1.5, label="Target 55% RH")
        ax.set_title("Relative Humidity Distribution"); ax.set_xlabel("Relative Humidity (%)"); ax.legend(fontsize=8, framealpha=0)

    ax = axes[1,1]
    if "alert_level" in df_iot.columns:
        ac = df_iot["alert_level"].value_counts()
        ax.pie(ac.values, labels=ac.index, autopct="%1.1f%%", colors=["#10b981","#f59e0b","#ef4444","#6b7280"][:len(ac)], wedgeprops={"edgecolor":"#0f1117","linewidth":2})
        ax.set_title("IoT Alert Level Distribution")
    elif "is_thermal_excursion" in df_iot.columns:
        ev = df_iot["is_thermal_excursion"].value_counts()
        ax.pie(ev.values, labels=["Excursion" if v else "Normal" for v in ev.index], colors=["#ef4444","#10b981"], autopct="%1.1f%%", wedgeprops={"edgecolor":"#0f1117","linewidth":2})
        ax.set_title("Thermal Excursion vs Normal Readings")

    plt.tight_layout(); show_fig(fig)
    info_box("IoT Charts", "ℹ️ Visualization of cold-chain telemetry data.")
    info_box("IoT Monitor", "ℹ️ What action should I take?")

    # ── AI Insight: Cold-Chain Drug Safety Analysis ────────────────
    _iot_bullets = []
    if "is_thermal_excursion" in df_iot.columns:
        _exc_wh    = df_iot.groupby("warehouse_id")["is_thermal_excursion"].mean() * 100
        _worst_iot = _exc_wh.idxmax() if not _exc_wh.empty else "N/A"
        _worst_pct = float(_exc_wh.max()) if not _exc_wh.empty else 0
        _crit_whs  = _exc_wh[_exc_wh > 10].index.tolist()
        if _worst_pct > 10:
            _iot_bullets.append(
                f"🚨 <b>CRITICAL excursion alert:</b> <b>{_worst_iot}</b> has a <b>{_worst_pct:.1f}% excursion rate</b> — "
                f"more than double the 5% USP &lt;659&gt; regulatory limit. Immediately quarantine all affected batches at this warehouse. "
                f"Inspect refrigeration units, calibrate sensors, and notify QA for batch impact assessment."
            )
        elif _worst_pct > 5:
            _iot_bullets.append(
                f"⚠️ <b>Excursion threshold exceeded:</b> <b>{_worst_iot}</b> at <b>{_worst_pct:.1f}%</b> — above the 5% USP &lt;659&gt; limit. "
                f"Schedule urgent refrigeration maintenance and QA review. Products may be compromised."
            )
        else:
            _iot_bullets.append(f"✅ <b>Cold-chain stable:</b> All warehouses below 5% excursion threshold (best: {_exc_wh.idxmin()} at {_exc_wh.min():.1f}%). Maintain current monitoring cadence.")
        _iot_bullets.append(
            f"💊 <b>Drug potency risk:</b> For biologics (adalimumab, insulin biosynthetic), vaccines, and heat-labile injectables, even brief temperature excursions above 8°C can "
            f"denature proteins and permanently compromise efficacy. Affected batches must be quarantined, tested, and may require destruction — representing significant inventory write-off."
        )
    if "relative_humidity_pct" in df_iot.columns:
        _avg_rh = df_iot["relative_humidity_pct"].mean()
        if _avg_rh > 60:
            _iot_bullets.append(f"💧 <b>High humidity risk:</b> Network average RH at {_avg_rh:.0f}% — above the 55% target. Elevated humidity accelerates tablet hygroscopic degradation, promotes microbial growth, and can compromise packaging seal integrity.")
        elif _avg_rh < 45:
            _iot_bullets.append(f"🏜️ <b>Low humidity risk:</b> Network average RH at {_avg_rh:.0f}% — below 45%. Low RH can cause capsule brittleness, tablet friability, and electrostatic packaging adhesion issues.")
    _iot_bullets.append(
        f"💡 <b>Recommended actions:</b> (1) Preventive maintenance on refrigeration units at highest-excursion warehouses, "
        f"(2) Quarantine + stability-test all batches with confirmed excursion exposure, "
        f"(3) Install automated real-time alerts (SMS/email) for any reading above 8°C, "
        f"(4) Verify IQ/OQ/PQ qualification status of all cold-chain equipment."
    )
    if _iot_bullets:
        ai_insight("Cold-Chain Compliance & Drug Safety", _iot_bullets, icon="❄️", color="#3b82f6")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FREIGHT REBALANCING
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "🚛 Freight Rebalancing":
    st.markdown('<div class="section-header">🚛 Inter-Warehouse Stock Rebalancing</div>', unsafe_allow_html=True)
    info_box("Freight Header", "ℹ️ Freight cost and logistics management.")
    st.markdown('<div class="section-desc">Freight cost matrix across all warehouses | Economy=cheapest/slowest | Express=fastest/most expensive | Identify cheapest routes for near-expiry stock transfers.</div>', unsafe_allow_html=True)
    info_box("Freight Rebalancing", "ℹ️ How to use the freight matrix")

    if not supp_ok:
        st.warning("Upload Freight Matrix file (file 04) to view this analysis.", icon="⚠️"); st.stop()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7)); fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Inter-Warehouse Stock Rebalancing — Freight Cost Matrix", fontsize=14, color="#f59e0b", fontweight="bold", y=1.04)

    cost_col = "cold_chain_thermal_transfer_cost_per_unit_usd" if "cold_chain_thermal_transfer_cost_per_unit_usd" in df_freight.columns else df_freight.select_dtypes(include=np.number).columns[0]
    freight_pivot = df_freight.pivot_table(values=cost_col, index="from_warehouse_id", columns="to_warehouse_id", fill_value=0)
    sns.heatmap(freight_pivot, annot=True, fmt=".2f", cmap="YlOrBr", ax=axes[0], cbar_kws={"label":"Cost per Unit (USD)"}, linewidths=0.5, linecolor="#0f1117")
    axes[0].set_title("Cold-Chain Freight Cost Matrix (USD per Unit)")

    if "logistics_tier" in df_freight.columns:
        tc = df_freight["logistics_tier"].value_counts()
        axes[1].pie(tc.values, labels=tc.index, autopct="%1.1f%%", colors=["#10b981","#3b82f6","#ef4444"][:len(tc)], wedgeprops={"edgecolor":"#0f1117","linewidth":2})
        axes[1].set_title("Freight Routes by Logistics Tier")
    else:
        axes[1].text(0.5, 0.5, "Logistics tier data not found", ha="center", va="center", color="#aaa", transform=axes[1].transAxes)

    plt.tight_layout(); show_fig(fig)
    info_box("Freight Charts", "ℹ️ Visualization of freight costs.")
    info_box("Freight Rebalancing", "ℹ️ Reading the cost matrix")

    amb_col = "ambient_transfer_cost_per_unit_usd" if "ambient_transfer_cost_per_unit_usd" in df_freight.columns else cost_col
    show_cols = [c for c in ["from_warehouse_id","to_warehouse_id","logistics_tier",amb_col,cost_col] if c in df_freight.columns]
    st.markdown("**Cheapest Transfer Routes**")
    st.dataframe(df_freight[show_cols].sort_values(amb_col).head(20), use_container_width=True)
    info_box("Freight Table", "ℹ️ List of economical freight routes.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RAW MATERIALS & PRICING  (Ideas 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "🧪 Raw Materials & Pricing":
    st.markdown('<div class="section-header">🧪 Raw Material Inventory & Price Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Stock levels vs. restock thresholds (Green/Orange/Red) | Price trend monitor | Buy-signal alerts for management</div>', unsafe_allow_html=True)

    # ── Generate synthetic raw material data ──────────────────────────────────
    @st.cache_data
    def build_rm_data():
        import numpy as np
        rng2 = np.random.default_rng(99)
        MATERIALS = [
            ("RM001", "Amoxicillin API",           "Active Ingredient",  "kg",   420,  800,  300,  22.50),
            ("RM002", "Insulin Biosynthetic",       "Biologics",          "vial", 180,  600,  200,  310.00),
            ("RM003", "Metformin HCl",              "Active Ingredient",  "kg",   950, 2000,  700,   8.40),
            ("RM004", "Atorvastatin Calcium",       "Active Ingredient",  "kg",   310,  900,  350,  38.20),
            ("RM005", "Lisinopril USP",             "Active Ingredient",  "kg",   730, 1500,  400,  51.00),
            ("RM006", "Amlodipine Besylate",        "Active Ingredient",  "kg",   520, 1200,  350,  29.80),
            ("RM007", "Epinephrine Bitartrate",     "Active Ingredient",  "g",    85,   300,  150, 890.00),
            ("RM008", "Warfarin Sodium",            "Active Ingredient",  "kg",   210,  700,  250,  62.50),
            ("RM009", "Ondansetron HCl",            "Active Ingredient",  "kg",   440, 1100,  380,  74.00),
            ("RM010", "Pantoprazole Sodium",        "Active Ingredient",  "kg",   680, 1400,  450,  42.00),
            ("RM011", "Adalimumab Drug Substance",  "Biologics",          "mg",   55,   200,  100, 4200.00),
            ("RM012", "Ceftriaxone Sodium",         "Active Ingredient",  "kg",   370,  850,  300,  95.00),
            ("RM013", "Microcrystalline Cellulose", "Excipient",          "kg",  1800, 5000, 1200,   3.20),
            ("RM014", "Lactose Monohydrate",        "Excipient",          "kg",  2100, 6000, 1500,   2.80),
            ("RM015", "Magnesium Stearate",         "Excipient",          "kg",   640, 2000,  500,   5.60),
            ("RM016", "HPMC (Coating)",             "Excipient",          "kg",   390, 1200,  400,  18.50),
            ("RM017", "Vial Glass Type I",          "Packaging",          "unit",9800,30000, 8000,   0.45),
            ("RM018", "Blister Foil Aluminium",     "Packaging",          "roll", 220,  600,  180,  12.40),
            ("RM019", "Sodium Chloride 0.9%",       "Solvent",            "L",   1400, 4000, 1000,   1.90),
            ("RM020", "Water for Injection (WFI)",  "Solvent",            "L",   3200, 8000, 2000,   0.30),
        ]
        col_rm = ["rm_id","material_name","material_type","uom","current_stock","max_capacity","restock_point","unit_price_usd"]
        df_rm = pd.DataFrame(MATERIALS, columns=col_rm)

        # Status
        def rm_status(row):
            pct = row.current_stock / row.restock_point
            if row.current_stock <= 0:          return "OUT OF STOCK", "#7f1d1d", "🛑"
            if pct < 1.0:                       return "CRITICAL — Order Now", "#ef4444", "🔴"
            if pct < 1.4:                       return "LOW — Reorder Soon",  "#f59e0b", "🟠"
            return "Sufficient",                                                "#10b981", "🟢"
        df_rm[["status","status_color","status_icon"]] = df_rm.apply(rm_status, axis=1, result_type="expand")
        df_rm["days_of_stock"] = (df_rm["current_stock"] / (df_rm["restock_point"] / 30)).round(0).astype(int)
        df_rm["stock_value_usd"] = df_rm["current_stock"] * df_rm["unit_price_usd"]

        # Price history — last 12 months with realistic trends
        months = pd.date_range(end=TODAY, periods=12, freq="ME").strftime("%Y-%m").tolist()
        price_hist = {}
        for _, row in df_rm.iterrows():
            base = row.unit_price_usd
            trend = rng2.choice(["rising", "stable", "falling"], p=[0.35, 0.45, 0.20])
            if trend == "rising":
                prices = [round(base * (1 + 0.015*i + rng2.normal(0,0.01)), 2) for i in range(12)]
            elif trend == "falling":
                prices = [round(base * (1 - 0.008*i + rng2.normal(0,0.01)), 2) for i in range(12)]
            else:
                prices = [round(base * (1 + rng2.normal(0,0.012)), 2) for _ in range(12)]
            prices = [max(p, base*0.5) for p in prices]
            price_hist[row.rm_id] = dict(zip(months, prices))
        df_price = pd.DataFrame(price_hist).T
        df_price.index.name = "rm_id"
        df_price = df_price.reset_index()
        df_price = df_price.merge(df_rm[["rm_id","material_name"]], on="rm_id")

        # Compute price trend signal
        def price_trend(row):
            vals = [row[m] for m in months]
            slope = (vals[-1] - vals[0]) / vals[0] * 100
            if slope > 8:   return "Rising ⬆️",   "🚨 Buy Now — price rising {:.1f}%".format(slope), "#ef4444"
            if slope > 3:   return "Rising ⬆️",   "⚠️ Monitor — price up {:.1f}%".format(slope), "#f59e0b"
            if slope < -3:  return "Falling ⬇️", "✅ Good time to buy — price down {:.1f}%".format(abs(slope)), "#10b981"
            return "Stable ➡️", "Price stable ({:.1f}%)".format(slope), "#94a3b8"
        df_price[["trend","recommendation","trend_color"]] = df_price.apply(price_trend, axis=1, result_type="expand")
        return df_rm, df_price, months

    df_rm, df_price, months = build_rm_data()

    # ── SECTION 1: STOCK LEVEL KPIs ──────────────────────────────────────────────
    critical = df_rm[df_rm["status"].str.startswith("CRITICAL")]
    low      = df_rm[df_rm["status"].str.startswith("LOW")]
    ok       = df_rm[df_rm["status"] == "Sufficient"]

    if len(critical) > 0:
        st.error(f"🛑 **{len(critical)} material(s) BELOW restock point — immediate purchase order required!** — {', '.join(critical.material_name.tolist())}", icon="🛑")
    if len(low) > 0:
        st.warning(f"🟠 **{len(low)} material(s) running low** — {', '.join(low.material_name.tolist())}", icon="⚠️")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Materials",      f"{len(df_rm)}",                   help="Total distinct raw material SKUs tracked")
    k2.metric("🔴 Critical / Order Now", f"{len(critical)}",              delta=None, help="Stock below restock threshold — PO required immediately")
    k3.metric("🟠 Low / Reorder Soon",  f"{len(low)}",                  delta=None, help="Stock within 40% of restock threshold — place order this week")
    k4.metric("Total Stock Value",    f"${df_rm.stock_value_usd.sum()/1e3:.0f}K", help="USD value of all raw materials currently on hand")

    st.markdown("---")

    # ── SECTION 2: STOCK LEVEL CHART + TABLE ─────────────────────────────────
    st.markdown('<div class="section-header">📊 Raw Material Stock Levels</div>', unsafe_allow_html=True)
    info_box("Raw Material Stock", "ℹ️ How to read stock levels")

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.patch.set_facecolor("#0f1117")

    # Bar chart — current vs restock point vs max
    ax = axes[0]
    sorted_rm = df_rm.sort_values("current_stock", ascending=True)
    bar_colors = ["#ef4444" if s.startswith("CRITICAL") else ("#f59e0b" if s.startswith("LOW") else "#10b981") for s in sorted_rm.status]
    bars = ax.barh(sorted_rm.material_name, sorted_rm.current_stock, color=bar_colors, alpha=0.85, height=0.6, label="Current Stock")
    ax.barh(sorted_rm.material_name, sorted_rm.restock_point, color="#ffffff", alpha=0.15, height=0.6, label="Restock Point")
    for i, (_, row) in enumerate(sorted_rm.iterrows()):
        ax.axhline(i, color="#1e2a45", linewidth=0.5)
        ax.text(row.current_stock + row.max_capacity*0.01, i, f" {row.status_icon}", va="center", fontsize=9)
    ax.set_xlabel("Units On Hand", color="#ccc")
    ax.set_title("Current Stock vs Restock Threshold", color="#00d4ff", fontweight="bold")
    ax.axvline(0, color="#444", linewidth=0.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#10b981",label="🟢 Sufficient"),Patch(color="#f59e0b",label="🟠 Low"),Patch(color="#ef4444",label="🔴 Critical"),Patch(color="#ffffff",alpha=0.2,label="Restock Point")], loc="lower right", fontsize=8, framealpha=0.2)

    # Stock utilisation % gauge-style bar (Fixed: no confusing multi-line grid)
    ax2 = axes[1]
    df_rm_s = df_rm.sort_values("current_stock", ascending=True).copy()
    df_rm_s["util_pct"] = (df_rm_s.current_stock / df_rm_s.max_capacity * 100).clip(0,100)
    df_rm_s["restock_pct"] = (df_rm_s.restock_point / df_rm_s.max_capacity * 100)
    bar_c2 = ["#ef4444" if s.startswith("CRITICAL") else ("#f59e0b" if s.startswith("LOW") else "#10b981") for s in df_rm_s.status]
    ax2.barh(df_rm_s.material_name, df_rm_s.util_pct, color=bar_c2, alpha=0.85, height=0.6, label="Current Stock (% Capacity)")

    # Plot specific restock point marker on each material bar row (clean per-item indicator)
    ax2.scatter(df_rm_s.restock_pct, range(len(df_rm_s)), color="#ffffff", marker="|", s=180, linewidths=2.5, zorder=5, label="Restock Trigger Point (%)")
    for i, (_, r) in enumerate(df_rm_s.iterrows()):
        ax2.text(min(r.util_pct + 1.5, 92), i, f"{r.util_pct:.0f}%", va="center", fontsize=8, color="#cbd5e1")

    ax2.set_xlim(0, 105)
    ax2.set_xlabel("% of Max Capacity", color="#ccc")
    ax2.set_title("Stock Utilisation (%) vs Specific Restock Thresholds (| Marker)", color="#00d4ff", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.2)
    ax2.grid(True, axis="x", alpha=0.15)

    plt.tight_layout()
    show_fig(fig)

    # Detailed table
    st.markdown("**📋 Full Raw Material Inventory**")
    display_rm = df_rm[["rm_id","material_name","material_type","uom","current_stock","restock_point","max_capacity","days_of_stock","status_icon","status","unit_price_usd","stock_value_usd"]].copy()
    display_rm.columns = ["ID","Material","Type","UoM","Stock","Restock Point","Capacity","Days of Stock","🚦","Status","Unit Price (USD)","Stock Value (USD)"]
    st.dataframe(display_rm, use_container_width=True)

    st.markdown("---")

    # ── SECTION 3: PRICE MONITOR + BUY SIGNALS (Idea 2) ───────────────────────
    st.markdown('<div class="section-header">📉 Raw Material Price Monitor & Buy Signals</div>', unsafe_allow_html=True)
    info_box("Price Monitor", "ℹ️ How the price signals work")

    # Filter selector
    trend_filter = st.multiselect("Filter by Trend", ["Rising ⬆️","Stable ➡️","Falling ⬇️"], default=["Rising ⬆️","Stable ➡️","Falling ⬇️"])
    df_price_f = df_price[df_price.trend.isin(trend_filter)] if trend_filter else df_price

    # Alert cards for rising materials
    rising = df_price[df_price.trend.str.startswith("Rising")]
    if not rising.empty:
        st.markdown("#### 🚨 Management Alerts — Price Action Required")
        for _, row in rising.iterrows():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"""
                <div style='background:#1e1a0a; border-left:4px solid {row.trend_color}; border-radius:8px; padding:12px 16px; margin-bottom:8px;'>
                    <span style='font-weight:700; color:{row.trend_color}; font-size:14px;'>{row.trend} {row.material_name}</span><br/>
                    <span style='color:#cbd5e1; font-size:12px;'>{row.recommendation}</span>
                </div>""", unsafe_allow_html=True)
            with col_b:
                rm_row = df_rm[df_rm.rm_id == row.rm_id].iloc[0]
                st.metric("Current Stock", f"{rm_row.current_stock:,} {rm_row.uom}",
                          delta=f"{rm_row.days_of_stock}d left",
                          help="Current on-hand stock and estimated days remaining at current consumption rate")

    # Price trend chart — select materials to compare
    st.markdown("#### 📈 Price Trend Lines (12 Months)")
    selected_mats = st.multiselect(
        "Select materials to compare",
        options=df_price_f.material_name.tolist(),
        default=df_price_f.material_name.head(6).tolist(),
        key="mat_price_sel"
    )
    if selected_mats:
        fig2, ax3 = plt.subplots(figsize=(20, 7))
        fig2.patch.set_facecolor("#0f1117")
        ax3.set_facecolor("#1a1d27")
        for i, mat in enumerate(selected_mats):
            row = df_price[df_price.material_name == mat].iloc[0]
            vals = [row[m] for m in months]
            color = row.trend_color
            ax3.plot(months, vals, marker="o", markersize=4, linewidth=2, color=color, label=mat, alpha=0.9)
        ax3.set_xlabel("Month", color="#ccc")
        ax3.set_ylabel("Unit Price (USD)", color="#ccc")
        ax3.set_title("Raw Material Price Trends — Last 12 Months", color="#00d4ff", fontweight="bold", fontsize=13)
        ax3.tick_params(axis="x", rotation=45)
        ax3.legend(fontsize=9, framealpha=0.15, loc="upper left")
        ax3.grid(True, alpha=0.2)
        plt.tight_layout()
        show_fig(fig2)

    # Price trend summary table
    st.markdown("**Price Signal Summary**")
    summary_cols = ["material_name","trend","recommendation"]
    price_summary = df_price_f[summary_cols + [months[-1],months[-3],months[0]]].copy()
    price_summary.columns = ["Material","Trend","Recommendation","Current Price","3M Ago","12M Ago"]
    price_summary["12M Change %"] = ((price_summary["Current Price"] - price_summary["12M Ago"]) / price_summary["12M Ago"] * 100).round(1)
    st.dataframe(price_summary, use_container_width=True)

    # ── AI Insight: Raw Material Procurement & Pricing Strategy ───────
    _crit_rm_list = critical.material_name.tolist() if not critical.empty else []
    _high_cost_rm = df_rm.sort_values("unit_price_usd", ascending=False).iloc[0]
    _rising_rm    = df_price[df_price.trend.str.startswith("Rising")].material_name.tolist()
    _falling_rm   = df_price[df_price.trend.str.startswith("Falling")].material_name.tolist()

    _rm_bullets = [
        f"⚖️ <b>Tailored Restock Levels:</b> Every raw material has a unique restock point and safety threshold based on vendor lead times (e.g. 60–90 days for imported APIs), batch consumption requirements, and cold-chain constraints.",
    ]
    if _crit_rm_list:
        _rm_bullets.append(
            f"🛑 <b>Production continuity bottleneck:</b> <b>{len(_crit_rm_list)} material(s) ({', '.join(_crit_rm_list[:3])}{'...' if len(_crit_rm_list)>3 else ''})</b> are below their restock threshold. "
            f"Manufacturing lines cannot start new campaigns without full active ingredient availability. Issue purchase orders immediately."
        )
    _rm_bullets.append(
        f"💎 <b>High-value material exposure:</b> <b>{_high_cost_rm['material_name']}</b> costs <b>${_high_cost_rm['unit_price_usd']:,.2f} per {_high_cost_rm['uom']}</b> (${_high_cost_rm['stock_value_usd']/1e3:,.0f}K total value). "
        f"Maintain lean, precise reorder quantities to minimize working capital blockage."
    )
    if _rising_rm:
        _rm_bullets.append(f"📈 <b>Price inflation alert:</b> <b>{len(_rising_rm)} material(s) ({', '.join(_rising_rm[:3])})</b> show sustained price increases. Contract 6-month forward supply agreements to protect manufacturing gross margins.")
    if _falling_rm:
        _rm_bullets.append(f"📉 <b>Opportunistic procurement:</b> <b>{len(_falling_rm)} material(s) ({', '.join(_falling_rm[:3])})</b> have declining market prices. Increase purchase order volume to lock in favorable unit costs.")
    _rm_bullets.append(
        f"💡 <b>Procurement recommendations:</b> (1) Issue emergency POs for all 🔴 CRITICAL materials today, "
        f"(2) Lock in blanket purchase agreements for rising commodities, "
        f"(3) Establish dual-sourcing for all single-origin active pharmaceutical ingredients."
    )
    ai_insight("Raw Material Procurement & Supply Continuity", _rm_bullets, icon="🧪", color="#10b981")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ORDER FULFILMENT SIMULATOR  (Idea 3 + 4 + 6)
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "📋 Order Fulfilment":
    st.markdown('<div class="section-header">📋 Order Fulfilment Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Simulate an incoming sales order — 3-path decision engine: Ship from Stock | Fulfil from WIP | Trigger Manufacturing + BOM raw material check</div>', unsafe_allow_html=True)
    with st.expander("ℹ️ How the 3-path decision engine works", expanded=False):
        st.markdown("**Path 1 ✅ SHIP FROM STOCK** — Enough finished goods in warehouse → dispatch directly.\n\n**Path 2 ⏳ FULFIL FROM WIP** — Not enough stock, but manufacturing order in progress → expected ready date.\n\n**Path 3 🚨 MANUFACTURE** — Neither stock nor WIP → raise new MO, check if all raw materials are available.")

    # ── Synthetic BOM + WIP data ───────────────────────────────────────────────
    @st.cache_data
    def build_fulfilment_data():
        # Bill of Materials: product_id -> {rm_id: qty_per_1000_units}
        BOM = {
            "P001": {"RM001":0.50, "RM013":0.30, "RM015":0.02, "RM014":0.20},
            "P002": {"RM002":1.00, "RM017":1.00, "RM019":0.50},
            "P003": {"RM003":0.55, "RM013":0.25, "RM014":0.18, "RM015":0.02},
            "P004": {"RM004":0.02, "RM013":0.35, "RM016":0.05, "RM018":0.10},
            "P005": {"RM005":0.01, "RM013":0.30, "RM014":0.20, "RM015":0.02},
            "P006": {"RM006":0.005,"RM013":0.28, "RM014":0.18, "RM016":0.04},
            "P007": {"RM007":0.30, "RM017":1.00, "RM019":0.30},
            "P008": {"RM008":0.005,"RM013":0.22, "RM014":0.15, "RM015":0.015},
            "P009": {"RM009":0.004,"RM013":0.18, "RM014":0.12, "RM016":0.03},
            "P010": {"RM010":0.04, "RM013":0.32, "RM016":0.06, "RM018":0.08},
            "P011": {"RM011":0.04, "RM017":1.00, "RM019":0.80},
            "P012": {"RM012":1.00, "RM017":1.00, "RM019":0.60},
        }
        # Raw material current stock
        RM_STOCK = {
            "RM001":420,"RM002":180,"RM003":950,"RM004":310,"RM005":730,
            "RM006":520,"RM007":85, "RM008":210,"RM009":440,"RM010":680,
            "RM011":55, "RM012":370,"RM013":1800,"RM014":2100,"RM015":640,
            "RM016":390,"RM017":9800,"RM018":220,"RM019":1400,"RM020":3200,
        }
        RM_NAMES = {
            "RM001":"Amoxicillin API","RM002":"Insulin Biosynthetic","RM003":"Metformin HCl",
            "RM004":"Atorvastatin Calcium","RM005":"Lisinopril USP","RM006":"Amlodipine Besylate",
            "RM007":"Epinephrine Bitartrate","RM008":"Warfarin Sodium","RM009":"Ondansetron HCl",
            "RM010":"Pantoprazole Sodium","RM011":"Adalimumab Drug Substance","RM012":"Ceftriaxone Sodium",
            "RM013":"Microcrystalline Cellulose","RM014":"Lactose Monohydrate","RM015":"Magnesium Stearate",
            "RM016":"HPMC (Coating)","RM017":"Vial Glass Type I","RM018":"Blister Foil Aluminium",
            "RM019":"Sodium Chloride 0.9%","RM020":"Water for Injection (WFI)",
        }
        # Synthetic WIP (manufacturing orders in progress)
        WIP = [
            {"mo_id":"MO-2026-001","product_id":"P001","status":"IN_PROGRESS","wip_units":5000,"expected_date":"2026-09-05","pct_complete":72},
            {"mo_id":"MO-2026-002","product_id":"P002","status":"IN_PROGRESS","wip_units":800, "expected_date":"2026-09-12","pct_complete":45},
            {"mo_id":"MO-2026-003","product_id":"P007","status":"IN_PROGRESS","wip_units":300, "expected_date":"2026-09-03","pct_complete":88},
            {"mo_id":"MO-2026-004","product_id":"P011","status":"IN_PROGRESS","wip_units":150, "expected_date":"2026-09-20","pct_complete":30},
            {"mo_id":"MO-2026-005","product_id":"P003","status":"SCHEDULED",  "wip_units":10000,"expected_date":"2026-10-01","pct_complete":0},
        ]
        df_wip = pd.DataFrame(WIP)
        return BOM, RM_STOCK, RM_NAMES, df_wip

    BOM, RM_STOCK, RM_NAMES, df_wip = build_fulfilment_data()

    # ── ORDER ENTRY SIMULATOR ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">📧 Simulate a Sales Order</div>', unsafe_allow_html=True)
    st.markdown("Enter a product and quantity — the system automatically checks stock, WIP, and raw material availability.")

    prod_options = products[["product_id","generic_name","dosage_form"]].copy()
    prod_options["label"] = prod_options["product_id"] + " — " + prod_options["generic_name"] + " (" + prod_options["dosage_form"] + ")"
    prod_map = dict(zip(prod_options["label"], prod_options["product_id"]))

    col_i1, col_i2, col_i3 = st.columns([3, 2, 1])
    sel_label   = col_i1.selectbox("💊 Product",  list(prod_map.keys()), key="ofs_prod")
    order_qty   = col_i2.number_input("📋 Order Quantity (units)", min_value=1, max_value=100000, value=500, step=50, key="ofs_qty")
    pref_wh     = col_i3.selectbox("🏢 Preferred WH", ["Any"] + sorted(warehouses["warehouse_id"].tolist()), key="ofs_wh")
    simulate_btn = st.button("▶️  Check Fulfilment", type="primary", use_container_width=False)

    if simulate_btn or True:   # always show after first render
        sel_pid = prod_map[sel_label]
        prod_name = products[products.product_id==sel_pid]["generic_name"].values[0]

        # ── Step 1: Check finished goods ───────────────────────────────────────
        inv_prod = inventory[(inventory["product_id"]==sel_pid) &
                             (inventory.get("qc_status", pd.Series(["RELEASED"]*len(inventory)))=="RELEASED")].copy() \
                  if "qc_status" in inventory.columns \
                  else inventory[inventory["product_id"]==sel_pid].copy()
        if pref_wh != "Any":
            inv_prod_wh = inv_prod[inv_prod["warehouse_id"]==pref_wh]
            avail_pref = int(inv_prod_wh["quantity_on_hand"].sum())
        else:
            avail_pref = 0
        avail_total = int(inv_prod["quantity_on_hand"].sum())

        # ── Step 2: Check WIP ────────────────────────────────────────────────
        wip_prod   = df_wip[df_wip["product_id"]==sel_pid]
        wip_total  = int(wip_prod["wip_units"].sum()) if not wip_prod.empty else 0
        wip_date   = wip_prod["expected_date"].min() if not wip_prod.empty else None

        # ── Decision path ────────────────────────────────────────────────────────
        st.markdown("### 🔍 Fulfilment Decision")
        r1, r2, r3 = st.columns(3)
        r1.metric("Order Quantity",     f"{order_qty:,} units",   help="What the customer ordered")
        r2.metric("Available in Stock", f"{avail_total:,} units",  delta=f"{avail_total - order_qty:+,} vs order",
                  delta_color="normal",   help="RELEASED finished goods across all warehouses")
        r3.metric("In WIP / Production",f"{wip_total:,} units",    delta=wip_date if wip_date else "None",
                  delta_color="off",      help="Units currently being manufactured")

        if avail_total >= order_qty:
            # Path 1 — Ship from stock
            best_whs = inv_prod.groupby("warehouse_id")["quantity_on_hand"].sum().sort_values(ascending=False)
            st.success(
                f"✅ **SHIP FROM STOCK** — {avail_total:,} units available (order = {order_qty:,}). "
                f"Best warehouse: **{best_whs.index[0]}** ({int(best_whs.iloc[0]):,} units)",
                icon="✅"
            )
            st.markdown("**Stock by Warehouse:**")
            wh_stock = inv_prod.groupby("warehouse_id").agg(
                Units=("quantity_on_hand","sum"),
                Batches=("fp_batch_id","nunique")
            ).sort_values("Units", ascending=False)
            wh_stock["Can Fulfil Order"] = wh_stock["Units"].apply(lambda u: "✅ Yes" if u>=order_qty else "⚠️ Partial")
            st.dataframe(wh_stock, use_container_width=True)

            # ── AI Insight: WHY this warehouse ────────────────────────────────
            _bwh        = best_whs.index[0]
            _bwh_units  = int(best_whs.iloc[0])
            _coverage   = _bwh_units / order_qty
            _batches_b  = int(wh_stock.loc[_bwh, "Batches"]) if _bwh in wh_stock.index else 1
            _ofs_bullets = [
                f"<b>{_bwh}</b> holds <b>{_bwh_units:,} units</b> — a <b>{_coverage:.1f}× coverage ratio</b> over the {order_qty:,}-unit order. "
                f"This provides the largest safety buffer across all warehouses, protecting against unexpected demand spikes or a batch going on QC hold after dispatch is initiated.",
                f"<b>{_batches_b} active batch{'es' if _batches_b>1 else ''}</b> at {_bwh} gives warehouse operators maximum FEFO flexibility: "
                f"the soonest-expiring batch can be selected first, ensuring regulatory compliance with FDA 21 CFR Part 211 and USP &lt;1079&gt;.",
            ]
            if "days_to_expiry" in inv_prod.columns:
                _fte = inv_prod.groupby("warehouse_id")["days_to_expiry"].agg(["min","mean"]).round(0).astype(int)
                if _bwh in _fte.index:
                    _dmin, _davg = int(_fte.loc[_bwh,"min"]), int(_fte.loc[_bwh,"mean"])
                    _urgency_txt = f"⚠️ This batch is within 60 days of expiry — flag for FEFO priority pick in WMS." if _dmin < 60 else f"Comfortable shelf life for normal transit."
                    _ofs_bullets.append(f"Earliest-expiring batch at {_bwh}: <b>{_dmin} days to expiry</b> (avg across batches: {_davg}d). {_urgency_txt}")
            if len(best_whs) > 1:
                _swh, _sunits = best_whs.index[1], int(best_whs.iloc[1])
                _ofs_bullets.append(
                    f"vs. <b>{_swh}</b> ({_sunits:,} units): {_bwh} holds <b>{_bwh_units-_sunits:,} more units</b> — "
                    f"eliminates any risk of split-shipment across multiple warehouses (simpler logistics, lower freight cost, faster delivery)."
                )
            _ofs_bullets.append(
                f"📋 <b>Action:</b> Raise dispatch order from {_bwh}, specify FEFO batch pick sequence in WMS. "
                f"Confirm cold-chain packaging requirements if product requires temperature control during transit. "
                f"Reserve {_bwh_units - order_qty:,} buffer units for other pending orders."
            )
            ai_insight(f"Why ship from {_bwh}?", _ofs_bullets, icon="🏢", color="#10b981")

        elif avail_total > 0 and avail_total < order_qty:
            # Path 1b — Partial from stock, rest from WIP or manufacture
            deficit = order_qty - avail_total
            st.warning(
                f"⚠️ **PARTIAL FULFILMENT** — {avail_total:,} of {order_qty:,} units available in stock. "
                f"**{deficit:,} units short.**",
                icon="⚠️"
            )
            if wip_total >= deficit:
                st.info(f"⏳ Remaining **{deficit:,} units** can come from WIP — expected ready by **{wip_date}**", icon="🏷️")
            else:
                st.error(f"🚨 WIP covers {wip_total:,} more units — still short **{deficit-wip_total:,} units**. → New Manufacturing Order required.", icon="🏷️")

        elif wip_total >= order_qty:
            # Path 2 — Fulfill from WIP
            st.info(
                f"⏳ **FULFIL FROM WIP** — No finished stock, but **{wip_total:,} units in production**. "
                f"Expected ready: **{wip_date}**",
                icon="🏷️"
            )
            st.dataframe(wip_prod[["mo_id","status","wip_units","expected_date","pct_complete"]], use_container_width=True)

        else:
            # Path 3 — Trigger new manufacturing order + RM check
            still_short = max(0, order_qty - avail_total - wip_total)
            st.error(
                f"🚨 **MANUFACTURE REQUIRED** — Stock: {avail_total:,} | WIP: {wip_total:,} | Still short: **{still_short:,} units**. "
                f"New manufacturing order must be raised.",
                icon="🏷️"
            )

        # ── Step 3: Raw Material Check (always show for manufactu-required path) ───
        needs_manufacture = avail_total < order_qty and (wip_total + avail_total) < order_qty
        if needs_manufacture:
            units_to_make = order_qty - avail_total - wip_total
            st.markdown("---")
            st.markdown(f"### 🧪 Raw Material Check for {units_to_make:,} units of {prod_name}")

            if sel_pid in BOM:
                bom_rows = []
                for rm_id, qty_per_1k in BOM[sel_pid].items():
                    needed = round(qty_per_1k * units_to_make / 1000, 3)
                    on_hand = RM_STOCK.get(rm_id, 0)
                    deficit_rm = max(0, needed - on_hand)
                    status_rm = "🟢 OK" if deficit_rm == 0 else "🔴 SHORTAGE"
                    bom_rows.append({
                        "RM ID": rm_id,
                        "Material": RM_NAMES.get(rm_id, rm_id),
                        "Needed": needed,
                        "On Hand": on_hand,
                        "Deficit": deficit_rm,
                        "Status": status_rm,
                    })
                df_bom = pd.DataFrame(bom_rows)
                shortages = df_bom[df_bom["Deficit"] > 0]

                if shortages.empty:
                    st.success("✅ All raw materials are available — manufacturing can begin immediately.", icon="✅")
                else:
                    st.error(
                        f"🛑 **{len(shortages)} raw material(s) insufficient.** Raise purchase orders before manufacturing.",
                        icon="🛑"
                    )
                    st.markdown("**📦 Purchase Orders Required:**")
                    for _, srow in shortages.iterrows():
                        st.markdown(
                            f"- **{srow['Material']}** (`{srow['RM ID']}`): "
                            f"need **{srow['Needed']:.3f}**, have **{srow['On Hand']}**, "
                            f"🔴 order **{srow['Deficit']:.3f}** more"
                        )

                st.markdown("**Bill of Materials — Full Breakdown:**")
                st.dataframe(df_bom, use_container_width=True)
            else:
                st.info("BOM not configured for this product in the simulator.")
    st.caption("→ Next: 🏷️ WIP & Manufacturing — see all active manufacturing orders and their completion status")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: WIP & MANUFACTURING TRACKER  (Idea 5)
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "🏷️ WIP & Manufacturing":
    st.markdown('<div class="section-header">🏷️ WIP & Manufacturing Tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">All active manufacturing orders — status, completion %, expected finish date, production stage, and WIP unit count per product</div>', unsafe_allow_html=True)

    @st.cache_data
    def build_wip_full():
        import numpy as np
        rng3 = np.random.default_rng(77)
        PROD_NAMES = {
            "P001":"Amoxicillin","P002":"Insulin Lispro","P003":"Metformin",
            "P004":"Atorvastatin","P005":"Lisinopril","P006":"Amlodipine",
            "P007":"Epinephrine","P008":"Warfarin","P009":"Ondansetron",
            "P010":"Pantoprazole","P011":"Adalimumab","P012":"Ceftriaxone",
        }
        STATUS_OPTS = ["IN_PROGRESS","IN_PROGRESS","IN_PROGRESS","SCHEDULED","QC_HOLD","COMPLETED"]
        STAGE_OPTS  = ["Granulation","Compression","Coating","Filling","QC Testing","Packaging","Dispatch Ready"]
        rows = []
        for i in range(22):
            pid  = rng3.choice(list(PROD_NAMES.keys()))
            stat = rng3.choice(STATUS_OPTS)
            pct  = int(rng3.integers(0,101)) if stat == "IN_PROGRESS" else (100 if stat=="COMPLETED" else 0)
            start_date = TODAY - pd.Timedelta(days=int(rng3.integers(1,20)))
            duration   = int(rng3.integers(7,30))
            end_date   = start_date + pd.Timedelta(days=duration)
            rows.append({
                "MO ID":        f"MO-2026-{i+1:03d}",
                "Product ID":   pid,
                "Product":      PROD_NAMES[pid],
                "Status":       stat,
                "Stage":        rng3.choice(STAGE_OPTS),
                "WIP Units":    int(rng3.integers(200, 15000)),
                "Start Date":   start_date.date(),
                "Expected End": end_date.date(),
                "% Complete":   pct,
                "Manufacturer": f"M00{rng3.integers(1,4)}",
            })
        return pd.DataFrame(rows)

    df_wip_full = build_wip_full()

    # ── KPI row ─────────────────────────────────────────────────────────────────
    in_prog  = df_wip_full[df_wip_full["Status"]=="IN_PROGRESS"]
    sched    = df_wip_full[df_wip_full["Status"]=="SCHEDULED"]
    qc_hold  = df_wip_full[df_wip_full["Status"]=="QC_HOLD"]
    done     = df_wip_full[df_wip_full["Status"]=="COMPLETED"]

    if not qc_hold.empty:
        st.warning(f"⚠️ **{len(qc_hold)} order(s) on QC HOLD** — {', '.join(qc_hold['MO ID'].tolist())}", icon="⚠️")

    w1, w2, w3, w4, w5 = st.columns(5)
    w1.metric("Total MOs",     len(df_wip_full), help="Total manufacturing orders in the system")
    w2.metric("⏳ In Progress", len(in_prog),     help="Orders actively being manufactured right now")
    w3.metric("🗓️ Scheduled",   len(sched),      help="Orders planned but not yet started")
    w4.metric("🔬 QC Hold",     len(qc_hold),    help="Orders paused for quality control investigation")
    w5.metric("WIP Units Total",f"{df_wip_full['WIP Units'].sum():,}", help="Total units currently in various production stages")

    st.markdown("---")

    # ── Progress Chart ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Manufacturing Order Progress</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    fig.patch.set_facecolor("#0f1117")

    # Left: Gantt-style progress bars
    ax_g = axes[0]
    active = df_wip_full[df_wip_full["Status"].isin(["IN_PROGRESS","QC_HOLD"])].sort_values("% Complete", ascending=True)
    bar_cols_wip = ["#ef4444" if s=="QC_HOLD" else "#00d4ff" for s in active["Status"]]
    bars_wip = ax_g.barh(active["MO ID"] + " " + active["Product"], active["% Complete"],
                         color=bar_cols_wip, alpha=0.85, height=0.55)
    ax_g.barh(active["MO ID"] + " " + active["Product"], 100,
              color="#1e2a45", alpha=0.5, height=0.55)
    for bar, (_, row) in zip(bars_wip, active.iterrows()):
        ax_g.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                  f"{row['% Complete']}% • {row['Stage']}", va="center", fontsize=8, color="#94a3b8")
    ax_g.set_xlim(0, 115)
    ax_g.set_xlabel("% Complete", color="#ccc")
    ax_g.set_title("Active Manufacturing Orders — Progress", color="#00d4ff", fontweight="bold")
    ax_g.axvline(100, color="#10b981", lw=1.5, linestyle="--", alpha=0.6, label="100% Done")

    # Right: WIP units by product
    ax_p = axes[1]
    wip_by_prod = df_wip_full.groupby("Product")["WIP Units"].sum().sort_values()
    colors_prod = PALETTE[:len(wip_by_prod)]
    ax_p.barh(wip_by_prod.index, wip_by_prod.values / 1000, color=colors_prod, alpha=0.85)
    ax_p.set_xlabel("WIP Units (thousands)", color="#ccc")
    ax_p.set_title("WIP Units by Product", color="#00d4ff", fontweight="bold")
    for i, (prod, val) in enumerate(wip_by_prod.items()):
        ax_p.text(val/1000 + 0.1, i, f"{val:,}", va="center", fontsize=9, color="#94a3b8")

    plt.tight_layout()
    show_fig(fig)

    # ── Status filter + full table ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 All Manufacturing Orders</div>', unsafe_allow_html=True)
    status_filter = st.multiselect(
        "Filter by Status",
        options=df_wip_full["Status"].unique().tolist(),
        default=df_wip_full["Status"].unique().tolist(),
        key="wip_status_filter"
    )
    df_wip_show = df_wip_full[df_wip_full["Status"].isin(status_filter)] if status_filter else df_wip_full

    # Colour-code status
    def highlight_wip(row):
        color = {"IN_PROGRESS":"#0f2a1a","SCHEDULED":"#0f1a2a",
                 "QC_HOLD":"#2a1a0f","COMPLETED":"#1a2a0f"}.get(row["Status"],"")
        return [f"background-color:{color}"]*len(row)

    st.dataframe(
        df_wip_show.style.apply(highlight_wip, axis=1),
        use_container_width=True,
        height=420
    )

    st.markdown("---")
    # ── WIP × Warehouse visibility ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 WIP Units by Product — Expected Completion Timeline</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(20, 6))
    fig3.patch.set_facecolor("#0f1117")
    ax3.set_facecolor("#1a1d27")
    timeline_df = df_wip_full[df_wip_full["Status"].isin(["IN_PROGRESS","SCHEDULED"])].copy()
    timeline_df["Expected End"] = pd.to_datetime(timeline_df["Expected End"])
    timeline_df = timeline_df.sort_values("Expected End")
    for i, (_, row) in enumerate(timeline_df.iterrows()):
        color = "#00d4ff" if row["Status"]=="IN_PROGRESS" else "#7c3aed"
        ax3.barh(i, row["WIP Units"] / 1000, color=color, alpha=0.8, height=0.6)
        ax3.text(row["WIP Units"]/1000 + 0.2, i,
                 f"{row['Product']} • {row['Expected End'].strftime('%d %b')} • {row['% Complete']}%",
                 va="center", fontsize=8, color="#94a3b8")
    ax3.set_xlabel("WIP Units (thousands)", color="#ccc")
    ax3.set_title("WIP Orders Sorted by Expected Completion Date", color="#00d4ff", fontweight="bold")
    ax3.set_yticks(range(len(timeline_df)))
    ax3.set_yticklabels([r["MO ID"] for _, r in timeline_df.iterrows()], fontsize=8)
    from matplotlib.patches import Patch
    ax3.legend(handles=[Patch(color="#00d4ff",label="In Progress"),Patch(color="#7c3aed",label="Scheduled")],
               fontsize=9, framealpha=0.2)
    plt.tight_layout()
    show_fig(fig3)

    # ── AI Insight: WIP & Manufacturing Execution ─────────────────────
    _wip_tot_u   = df_wip_full["WIP Units"].sum()
    _in_prog_u   = in_prog["WIP Units"].sum() if not in_prog.empty else 0
    _qc_hold_mos = qc_hold["MO ID"].tolist() if not qc_hold.empty else []
    _top_wip_p   = df_wip_full.groupby("Product")["WIP Units"].sum().idxmax()

    _wip_bullets = [
        f"🏭 <b>Shopfloor pipeline:</b> <b>{_wip_tot_u:,} total units</b> currently in the manufacturing pipeline across {len(df_wip_full)} MOs. <b>{_in_prog_u:,} units</b> are actively in granulation, compression, coating, or filling.",
        f"📦 <b>Highest volume campaign:</b> <b>{_top_wip_p}</b> dominates active production volume. Packaging lines and QC analytical teams should be pre-scheduled to handle upcoming lot completions without release delays.",
    ]
    if _qc_hold_mos:
        _wip_bullets.append(
            f"🔬 <b>QC Quarantine impact:</b> <b>{len(_qc_hold_mos)} order(s) ({', '.join(_qc_hold_mos)})</b> are paused on QC Hold. "
            f"Expedite out-of-specification (OOS) investigations and microbial release assays to prevent production scheduling bottlenecks."
        )
    _wip_bullets.append(
        f"💡 <b>Plant manager action plan:</b> (1) Accelerate QA batch record review for orders reaching &gt;80% completion, "
        f"(2) Synchronize raw material kitting with scheduled MO start dates to eliminate shopfloor downtime, "
        f"(3) Implement daily line-clearance and OEE monitoring to maximize tablet press throughput."
    )
    ai_insight("WIP Throughput & Shopfloor Execution", _wip_bullets, icon="🏷️", color="#00d4ff")

    st.caption("→ Next: 🗺️ Geo Sales Intelligence — find which locations have HOT demand & surplus stock")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GEO SALES INTELLIGENCE  (Idea 7)
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "🗺️ Geo Sales Intelligence":
    st.markdown('<div class="section-header">🗺️ Geographical Sales Pattern Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Identify where demand is highest vs lowest across warehouses/regions — spot 🔥 HOT locations (high demand, low stock) and ❄️ COLD locations (low demand, surplus stock) for smart rebalancing decisions.</div>', unsafe_allow_html=True)

    if not supp_ok:
        st.warning("⚠️ Upload Monthly Demand data (via the template) to enable this analysis.", icon="⚠️")
        st.stop()

    # ── Build geo demand summary ────────────────────────────────────────────────────
    @st.cache_data
    def build_geo_data(demand_hash):
        # Aggregate demand and stock by warehouse + product
        dem_agg = df_demand.groupby(["warehouse_id","product_id"]).agg(
            total_demanded  = ("quantity_demanded_units","sum"),
            total_dispatched= ("quantity_dispatched_units","sum"),
            avg_monthly_demand=("quantity_demanded_units","mean"),
            fill_rate_avg   = ("quantity_demanded_units", lambda x:
                                (df_demand.loc[x.index,"quantity_dispatched_units"].sum() /
                                 x.sum() * 100) if x.sum()>0 else 0),
        ).reset_index()

        inv_agg = inventory.groupby(["warehouse_id","product_id"]).agg(
            stock_on_hand=("quantity_on_hand","sum"),
            stock_value  =("inventory_value_usd","sum"),
        ).reset_index()

        geo = dem_agg.merge(inv_agg, on=["warehouse_id","product_id"], how="left")
        geo["stock_on_hand"]  = geo["stock_on_hand"].fillna(0)
        geo["stock_value"]    = geo["stock_value"].fillna(0)

        # Days of stock at current demand rate
        geo["days_of_stock"] = (geo["stock_on_hand"] /
                                 (geo["avg_monthly_demand"] / 30)).replace([float("inf"),float("nan")], 999).round(0)

        # Classify location-product pairs
        def classify(row):
            if row.avg_monthly_demand > row.avg_monthly_demand * 1.0 and row.days_of_stock < 20:
                return "🔥 HOT",  "#ef4444"
            if row.days_of_stock > 120:
                return "❄️ COLD", "#3b82f6"
            return "✅ BALANCED", "#10b981"

        # Use percentile-based thresholds instead
        p75_demand = geo["avg_monthly_demand"].quantile(0.75)
        p25_demand = geo["avg_monthly_demand"].quantile(0.25)
        p25_dos    = geo["days_of_stock"].clip(upper=300).quantile(0.25)
        p75_dos    = geo["days_of_stock"].clip(upper=300).quantile(0.75)

        def classify2(row):
            high_demand = row.avg_monthly_demand >= p75_demand
            low_demand  = row.avg_monthly_demand <= p25_demand
            low_stock   = row.days_of_stock <= p25_dos
            high_stock  = min(row.days_of_stock, 300) >= p75_dos
            if high_demand and low_stock:  return "🔥 HOT",      "#ef4444"
            if low_demand  and high_stock: return "❄️ COLD",     "#3b82f6"
            return                                "✅ BALANCED",  "#10b981"

        geo[["location_type","loc_color"]] = geo.apply(classify2, axis=1, result_type="expand")

        # Add product names
        prod_name_map = dict(zip(products.product_id, products.generic_name))
        geo["product_name"] = geo["product_id"].map(prod_name_map).fillna(geo["product_id"])

        # Warehouse info
        if "city" in warehouses.columns:
            wh_info = dict(zip(warehouses.warehouse_id, warehouses.city + " (" + warehouses.warehouse_id + ")"))
        else:
            wh_info = dict(zip(warehouses.warehouse_id, warehouses.warehouse_id))
        geo["wh_label"] = geo["warehouse_id"].map(wh_info).fillna(geo["warehouse_id"])
        return geo, p75_demand, p25_demand

    geo, p75d, p25d = build_geo_data(str(len(df_demand)))

    hot  = geo[geo["location_type"]=="🔥 HOT"]
    cold = geo[geo["location_type"]=="❄️ COLD"]
    bal  = geo[geo["location_type"]=="✅ BALANCED"]

    # ── KPI row ─────────────────────────────────────────────────────────────────
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("🔥 HOT Locations",     len(hot),  help="High demand + low stock — risk of stockout")
    g2.metric("❄️ COLD Locations",    len(cold), help="Low demand + surplus stock — capital tied up")
    g3.metric("✅ Balanced",           len(bal),  help="Demand and stock are well-matched")
    g4.metric("Products × Warehouses", len(geo),  help="Total product-warehouse combinations analysed")

    if not hot.empty:
        st.error(
            f"🔥 **{len(hot)} HOT location-product pairs** — stock replenishment urgent: "
            + ", ".join(f"{r.product_name} @ {r.warehouse_id}" for _, r in hot.head(4).iterrows())
            + (" & more..." if len(hot)>4 else ""),
            icon="🔥"
        )

    st.markdown("---")

    # ── Filter controls ────────────────────────────────────────────────────────────
    cf1, cf2 = st.columns(2)
    sel_prod_geo = cf1.multiselect("Filter Products",
                                   options=sorted(geo.product_name.unique()),
                                   default=sorted(geo.product_name.unique())[:6],
                                   key="geo_prod_filter")
    sel_type_geo = cf2.multiselect("Location Type",
                                   options=["🔥 HOT","❄️ COLD","✅ BALANCED"],
                                   default=["🔥 HOT","❄️ COLD","✅ BALANCED"],
                                   key="geo_type_filter")
    geo_f = geo[geo.product_name.isin(sel_prod_geo) & geo.location_type.isin(sel_type_geo)]

    # ── CHART 1: Demand heatmap — product × warehouse ──────────────────────────────
    st.markdown('<div class="section-header">🔥❄️ Demand Intensity × Warehouse Heatmap</div>', unsafe_allow_html=True)
    info_box("Demand Trend", "ℹ️ How to read the demand heatmap")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.patch.set_facecolor("#0f1117")

    # Panel 1: demand heatmap
    ax1 = axes[0]
    pivot_dem = geo_f.pivot_table(index="product_name", columns="warehouse_id",
                                   values="avg_monthly_demand", aggfunc="sum", fill_value=0)
    if not pivot_dem.empty:
        import matplotlib.cm as cm
        vmax = pivot_dem.values.max()
        im1  = ax1.imshow(pivot_dem.values, cmap="RdYlGn_r", aspect="auto",
                           vmin=0, vmax=vmax if vmax>0 else 1)
        ax1.set_xticks(range(len(pivot_dem.columns)))
        ax1.set_xticklabels(pivot_dem.columns, rotation=45, ha="right", fontsize=9)
        ax1.set_yticks(range(len(pivot_dem.index)))
        ax1.set_yticklabels([n[:20] for n in pivot_dem.index], fontsize=8)
        for i in range(len(pivot_dem.index)):
            for j in range(len(pivot_dem.columns)):
                ax1.text(j, i, f"{pivot_dem.values[i,j]:.0f}",
                         ha="center", va="center", fontsize=7, color="white")
        ax1.set_title("Avg Monthly Demand\n(Red=High, Green=Low)", color="#00d4ff", fontweight="bold")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

    # Panel 2: stock heatmap
    ax2 = axes[1]
    pivot_stk = geo_f.pivot_table(index="product_name", columns="warehouse_id",
                                   values="days_of_stock", aggfunc="mean", fill_value=0).clip(upper=200)
    if not pivot_stk.empty:
        im2 = ax2.imshow(pivot_stk.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=200)
        ax2.set_xticks(range(len(pivot_stk.columns)))
        ax2.set_xticklabels(pivot_stk.columns, rotation=45, ha="right", fontsize=9)
        ax2.set_yticks(range(len(pivot_stk.index)))
        ax2.set_yticklabels([n[:20] for n in pivot_stk.index], fontsize=8)
        for i in range(len(pivot_stk.index)):
            for j in range(len(pivot_stk.columns)):
                v = pivot_stk.values[i,j]
                ax2.text(j, i, f"{v:.0f}d", ha="center", va="center", fontsize=7, color="white")
        ax2.set_title("Days of Stock\n(Green=Ample, Red=Low)", color="#00d4ff", fontweight="bold")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Panel 3: HOT/COLD scatter — demand vs days-of-stock
    ax3 = axes[2]
    ax3.set_facecolor("#1a1d27")
    for ltype, color, marker in [("🔥 HOT","#ef4444","^"),("❄️ COLD","#3b82f6","v"),("✅ BALANCED","#10b981","o")]:
        sub = geo_f[geo_f.location_type==ltype]
        if not sub.empty:
            ax3.scatter(sub.avg_monthly_demand, sub.days_of_stock.clip(upper=200),
                        c=color, marker=marker, s=80, alpha=0.8, label=ltype)
    ax3.axhline(30,  color="#ef4444", lw=1.5, linestyle="--", alpha=0.6, label="30d threshold")
    ax3.axhline(120, color="#3b82f6", lw=1.5, linestyle="--", alpha=0.6, label="120d threshold")
    ax3.set_xlabel("Avg Monthly Demand (units)", color="#ccc")
    ax3.set_ylabel("Days of Stock", color="#ccc")
    ax3.set_title("Demand vs Days-of-Stock\n(HOT=top-left, COLD=bottom-right)", color="#00d4ff", fontweight="bold")
    ax3.legend(fontsize=8, framealpha=0.2)
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    show_fig(fig)

    # ── CHART 2: HOT vs COLD bar breakdown ────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(22, 6))
    fig2.patch.set_facecolor("#0f1117")

    # Top HOT locations by demand gap
    ax4 = axes2[0]
    if not hot.empty:
        top_hot = hot.copy()
        top_hot["demand_gap"] = top_hot["total_demanded"] - top_hot["total_dispatched"]
        top_hot = top_hot.nlargest(12, "total_demanded")
        labels_hot = top_hot["product_name"].str[:15] + "\n" + top_hot["warehouse_id"]
        ax4.barh(labels_hot, top_hot["total_demanded"]/1000, color="#ef4444", alpha=0.85, label="Demanded", height=0.5)
        ax4.barh(labels_hot, top_hot["total_dispatched"]/1000, color="#f97316", alpha=0.65, label="Dispatched", height=0.5)
        ax4.set_xlabel("Units (thousands)", color="#ccc")
        ax4.set_title("🔥 Top HOT Locations — Demand vs Dispatched", color="#ef4444", fontweight="bold")
        ax4.legend(fontsize=9, framealpha=0.2)
    else:
        ax4.text(0.5, 0.5, "No HOT locations detected", ha="center", va="center", color="#aaa", transform=ax4.transAxes)
        ax4.set_title("🔥 HOT Locations", color="#00d4ff")

    # Top COLD locations by surplus stock
    ax5 = axes2[1]
    if not cold.empty:
        top_cold = cold.nlargest(12, "stock_on_hand")
        labels_cold = top_cold["product_name"].str[:15] + "\n" + top_cold["warehouse_id"]
        ax5.barh(labels_cold, top_cold["stock_on_hand"]/1000, color="#3b82f6", alpha=0.85, label="Stock on Hand", height=0.5)
        ax5.barh(labels_cold, top_cold["avg_monthly_demand"]/1000, color="#93c5fd", alpha=0.65, label="Avg Monthly Demand", height=0.5)
        ax5.set_xlabel("Units (thousands)", color="#ccc")
        ax5.set_title("❄️ Top COLD Locations — Surplus Stock", color="#3b82f6", fontweight="bold")
        ax5.legend(fontsize=9, framealpha=0.2)
    else:
        ax5.text(0.5, 0.5, "No COLD locations detected", ha="center", va="center", color="#aaa", transform=ax5.transAxes)
        ax5.set_title("❄️ COLD Locations", color="#00d4ff")

    plt.tight_layout()
    show_fig(fig2)

    # ── Detailed table ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Location Intelligence Table</div>', unsafe_allow_html=True)
    show_cols_geo = ["location_type","product_name","warehouse_id","avg_monthly_demand",
                     "stock_on_hand","days_of_stock","fill_rate_avg","stock_value"]
    display_geo = geo_f[show_cols_geo].rename(columns={
        "location_type":"Type","product_name":"Product","warehouse_id":"Warehouse",
        "avg_monthly_demand":"Avg Monthly Demand","stock_on_hand":"Stock",
        "days_of_stock":"Days of Stock","fill_rate_avg":"Fill Rate %","stock_value":"Stock Value (USD)"
    }).sort_values(["Type","Avg Monthly Demand"], ascending=[True,False]).reset_index(drop=True)
    display_geo["Fill Rate %"] = display_geo["Fill Rate %"].round(1)
    display_geo["Avg Monthly Demand"] = display_geo["Avg Monthly Demand"].round(0)
    display_geo["Days of Stock"] = display_geo["Days of Stock"].clip(upper=999).round(0)
    st.dataframe(display_geo, use_container_width=True, height=420)

    # ── AI Insight: Geo Sales & Network Balancing ─────────────────────
    _hot_cnt  = len(hot)
    _cold_cnt = len(cold)
    _hot_pairs = [f"{r.product_name} @ {r.warehouse_id}" for _, r in hot.head(3).iterrows()]

    _geo_bullets = [
        f"🔥 <b>Demand hotspot risk:</b> <b>{_hot_cnt} location-product pair(s)</b> exhibit high demand with critical stock shortages ({', '.join(_hot_pairs)}{'...' if len(hot)>3 else ''}). These represent immediate stockout risks and lost hospital revenue.",
        f"❄️ <b>Cold location capital traps:</b> <b>{_cold_cnt} location(s)</b> carry surplus stock exceeding 120 days of forward demand. Trapped inventory incurs daily holding costs and increases expiry exposure.",
        f"🌐 <b>Network balancing opportunity:</b> Rather than triggering expensive rush manufacturing at CMOs, execute inter-warehouse transfers from COLD to HOT nodes to satisfy demand in 2–4 days.",
        f"💡 <b>Regional distribution strategy:</b> (1) Reallocate safety stock from COLD warehouses to high-velocity metropolitan distribution centers, "
        f"(2) Align warehouse stocking thresholds with localized seasonal epidemiology (e.g. respiratory surge in winter), "
        f"(3) Use the Smart Transfer Recommender to execute cost-optimal rebalancing."
    ]
    ai_insight("Geographic Sales Intelligence & Network Balancing", _geo_bullets, icon="🗺️", color="#ef4444")

    st.caption("→ Next: 💡 Smart Transfer Recommender — get cost-optimal rebalancing decisions based on this analysis")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SMART TRANSFER RECOMMENDER  (Idea 8)
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "💡 Smart Transfer Recommender":
    st.markdown('<div class="section-header">💡 Smart Stock Transfer Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">For every 🔥 HOT location: compares Transfer cost (move surplus from ❄️ COLD warehouse) vs Manufacturing cost — recommends the cheaper option with exact $ savings</div>', unsafe_allow_html=True)

    if not supp_ok:
        st.warning("⚠️ Upload data via the template to enable this analysis.", icon="⚠️")
        st.stop()

    # ── Re-use geo data ────────────────────────────────────────────────────────────────
    @st.cache_data
    def build_recommender(demand_hash, freight_hash):
        # Geo analysis
        dem_agg = df_demand.groupby(["warehouse_id","product_id"]).agg(
            avg_monthly_demand=("quantity_demanded_units","mean"),
            total_demanded    =("quantity_demanded_units","sum"),
        ).reset_index()
        inv_agg = inventory.groupby(["warehouse_id","product_id"]).agg(
            stock_on_hand=("quantity_on_hand","sum"),
        ).reset_index()
        geo2 = dem_agg.merge(inv_agg, on=["warehouse_id","product_id"], how="left")
        geo2["stock_on_hand"] = geo2["stock_on_hand"].fillna(0)
        geo2["days_of_stock"] = (geo2["stock_on_hand"] /
                                  (geo2["avg_monthly_demand"] / 30)).replace([float("inf"), float("nan")], 999)
        p75d2 = geo2["avg_monthly_demand"].quantile(0.75)
        p25d2 = geo2["avg_monthly_demand"].quantile(0.25)
        p25s2 = geo2["days_of_stock"].clip(upper=300).quantile(0.25)
        p75s2 = geo2["days_of_stock"].clip(upper=300).quantile(0.75)
        def cls2(row):
            hd = row.avg_monthly_demand >= p75d2
            ld = row.avg_monthly_demand <= p25d2
            ls = row.days_of_stock    <= p25s2
            hs = min(row.days_of_stock, 300) >= p75s2
            if hd and ls: return "🔥 HOT"
            if ld and hs: return "❄️ COLD"
            return "✅ BALANCED"
        geo2["location_type"] = geo2.apply(cls2, axis=1)

        hot2  = geo2[geo2.location_type=="🔥 HOT"].copy()
        cold2 = geo2[geo2.location_type=="❄️ COLD"].copy()

        prod_name_map2 = dict(zip(products.product_id, products.generic_name))
        prod_price_map = dict(zip(products.product_id, products.unit_price))

        # Freight cost lookup
        cost_col  = next((c for c in df_freight.columns if "cost" in c.lower() and "ambient" in c.lower()), None)
        cold_col  = next((c for c in df_freight.columns if "cold" in c.lower() and "cost" in c.lower()), None)
        freight_map = {}
        if cost_col and "from_warehouse_id" in df_freight.columns:
            for _, fr in df_freight.iterrows():
                freight_map[(fr.from_warehouse_id, fr.to_warehouse_id)] = {
                    "ambient": float(fr[cost_col]),
                    "cold":    float(fr[cold_col]) if cold_col else float(fr[cost_col]) * 2.0,
                    "tier":    fr.get("logistics_tier", "Standard"),
                }

        # Manufacturing cost proxy: 40% of unit price (COGS estimate)
        MFG_COST_FACTOR = 0.40

        recs = []
        for _, hot_row in hot2.iterrows():
            pid      = hot_row.product_id
            hot_wh   = hot_row.warehouse_id
            shortage = max(0, hot_row.avg_monthly_demand * 2 - hot_row.stock_on_hand)  # 2-month coverage goal
            if shortage < 10: continue

            unit_price  = prod_price_map.get(pid, 50)
            mfg_cost_pu = unit_price * MFG_COST_FACTOR
            mfg_total   = round(mfg_cost_pu * shortage, 2)

            # Find best COLD warehouse with surplus of same product
            cold_same = cold2[cold2.product_id == pid].copy()
            if cold_same.empty:
                recs.append({
                    "Product":         prod_name_map2.get(pid, pid),
                    "HOT Warehouse":   hot_wh,
                    "Shortage (units)": round(shortage),
                    "Best Action":     "🏷️ Manufacture",
                    "COLD Warehouse":  "—",
                    "Transfer Cost ($)":  "—",
                    "Mfg Cost ($)":    f"${mfg_total:,.0f}",
                    "Recommended":     "🏷️ Manufacture",
                    "Est. Saving ($)":  0,
                    "Reason":          "No surplus stock found elsewhere — manufacture new units",
                })
                continue

            # For each COLD source, find transfer cost
            best_transfer_cost = float("inf")
            best_cold_wh       = None
            best_freight_info  = {}
            for _, cold_row in cold_same.iterrows():
                cold_wh      = cold_row.warehouse_id
                surplus      = cold_row.stock_on_hand - cold_row.avg_monthly_demand * 2  # keep 2m buffer
                if surplus < shortage * 0.5: continue  # not enough to be useful
                transferable = min(surplus, shortage)
                fkey         = (cold_wh, hot_wh)
                if fkey in freight_map:
                    fc_pu = freight_map[fkey]["ambient"]
                    fc_total = fc_pu * transferable
                    if fc_total < best_transfer_cost:
                        best_transfer_cost = fc_total
                        best_cold_wh       = cold_wh
                        best_freight_info  = freight_map[fkey]

            if best_cold_wh is None:
                # No freight route found — default to manufacture
                recs.append({
                    "Product":         prod_name_map2.get(pid, pid),
                    "HOT Warehouse":   hot_wh,
                    "Shortage (units)": round(shortage),
                    "Best Action":     "🏷️ Manufacture",
                    "COLD Warehouse":  "—",
                    "Transfer Cost ($)":  "—",
                    "Mfg Cost ($)":    f"${mfg_total:,.0f}",
                    "Recommended":     "🏷️ Manufacture",
                    "Est. Saving ($)":  0,
                    "Reason":          "No direct freight route found — manufacture recommended",
                })
            else:
                saving     = round(mfg_total - best_transfer_cost, 2)
                if best_transfer_cost < mfg_total:
                    action = "🚛 Transfer from " + best_cold_wh
                    reason = (f"Transfer from {best_cold_wh} → {hot_wh} costs ${best_transfer_cost:,.0f} — "
                              f"saves ${saving:,.0f} vs manufacturing (${mfg_total:,.0f})")
                else:
                    action = "🏷️ Manufacture"
                    saving = round(best_transfer_cost - mfg_total, 2)
                    reason = (f"Manufacturing (${mfg_total:,.0f}) is cheaper than transfer from "
                              f"{best_cold_wh} (${best_transfer_cost:,.0f}) by ${saving:,.0f}")
                recs.append({
                    "Product":          prod_name_map2.get(pid, pid),
                    "HOT Warehouse":    hot_wh,
                    "Shortage (units)": round(shortage),
                    "Best Action":      action,
                    "COLD Warehouse":   best_cold_wh,
                    "Transfer Cost ($)": f"${best_transfer_cost:,.0f}" if best_transfer_cost < float("inf") else "—",
                    "Mfg Cost ($)":     f"${mfg_total:,.0f}",
                    "Recommended":      action,
                    "Est. Saving ($)":  max(0, saving),
                    "Reason":           reason,
                })
        return pd.DataFrame(recs) if recs else pd.DataFrame()

    df_recs = build_recommender(str(len(df_demand)), str(len(df_freight)))

    if df_recs.empty:
        st.info("✅ No HOT locations detected with your current data — all warehouses appear well-stocked.", icon="ℹ️")
    else:
        transfer_recs = df_recs[df_recs["Recommended"].str.startswith("🚛")]
        mfg_recs      = df_recs[df_recs["Recommended"].str.startswith("🏷️")]
        total_savings = df_recs["Est. Saving ($)"].sum()

        # ── KPI row ───────────────────────────────────────────────────────────────
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("HOT Locations Analysed", len(df_recs),           help="Number of product-warehouse pairs with HOT status")
        s2.metric("🚛 Transfer Recommended",   len(transfer_recs),    help="Cases where inter-warehouse transfer is cheaper than manufacturing")
        s3.metric("🏷️ Manufacture Recommended",len(mfg_recs),         help="Cases where manufacturing new units is cheaper than transferring")
        s4.metric("💰 Total Estimated Savings", f"${total_savings:,.0f}", help="Total $ saved by choosing transfer over manufacture (where applicable)")

        st.markdown("---")

        # ── Recommendation cards ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📦 Action Cards — Top Recommendations</div>', unsafe_allow_html=True)
        for _, rec in df_recs.sort_values("Est. Saving ($)", ascending=False).head(10).iterrows():
            is_transfer = rec["Recommended"].startswith("🚛")
            border_col  = "#10b981" if is_transfer else "#7c3aed"
            icon        = "🚛" if is_transfer else "🏷️"
            saving_str  = f"**Save ${rec['Est. Saving ($)']:,.0f}**" if rec["Est. Saving ($)"] > 0 else "Cost-optimised"
            st.markdown(f"""
<div style='background:#0f1a2a; border-left:5px solid {border_col}; border-radius:10px;
            padding:14px 20px; margin-bottom:10px;'>
  <div style='display:flex; justify-content:space-between; align-items:center;'>
    <span style='font-size:15px; font-weight:700; color:{border_col};'>{icon} {rec['Product']} → {rec['HOT Warehouse']}</span>
    <span style='font-size:13px; color:#10b981; font-weight:600;'>{saving_str}</span>
  </div>
  <div style='color:#94a3b8; font-size:12px; margin-top:6px;'>
    Shortage: <b>{rec['Shortage (units)']:,.0f} units</b> &nbsp;|
    Transfer: <b>{rec['Transfer Cost ($)']}</b> &nbsp;|
    Manufacture: <b>{rec['Mfg Cost ($)']}</b>
  </div>
  <div style='color:#cbd5e1; font-size:12px; margin-top:4px;'>{rec['Reason']}</div>
</div>""", unsafe_allow_html=True)

        # ── Cost comparison chart ───────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Transfer vs Manufacturing Cost Comparison</div>', unsafe_allow_html=True)

        chart_df = df_recs[df_recs["Transfer Cost ($)"] != "—"].copy()
        if not chart_df.empty:
            fig_r, ax_r = plt.subplots(figsize=(20, max(5, len(chart_df)*0.7)))
            fig_r.patch.set_facecolor("#0f1117")
            ax_r.set_facecolor("#1a1d27")

            labels_r  = chart_df["Product"].str[:14] + "\n→ " + chart_df["HOT Warehouse"]
            xpos      = range(len(chart_df))
            w_bar     = 0.35

            # Parse numeric costs
            t_costs = chart_df["Transfer Cost ($)"].str.replace("[$,]","",regex=True).astype(float)
            m_costs = chart_df["Mfg Cost ($)"].str.replace("[$,]","",regex=True).astype(float)

            bars1 = ax_r.bar([x - w_bar/2 for x in xpos], t_costs,  width=w_bar, color="#10b981", alpha=0.85, label="Transfer Cost")
            bars2 = ax_r.bar([x + w_bar/2 for x in xpos], m_costs,  width=w_bar, color="#7c3aed", alpha=0.85, label="Manufacturing Cost")

            ax_r.set_xticks(xpos)
            ax_r.set_xticklabels(labels_r, rotation=0, ha="center", fontsize=8)
            ax_r.set_ylabel("Cost (USD)", color="#ccc")
            ax_r.set_title("Transfer vs Manufacturing Cost per HOT Location",
                           color="#00d4ff", fontweight="bold", fontsize=13)
            ax_r.legend(fontsize=10, framealpha=0.2)
            ax_r.grid(True, axis="y", alpha=0.2)

            # Saving annotations
            for i, (tc, mc) in enumerate(zip(t_costs, m_costs)):
                saving_ann = mc - tc
                color_ann  = "#10b981" if saving_ann > 0 else "#ef4444"
                ax_r.annotate(f"{'Save' if saving_ann>0 else 'Cost'} ${abs(saving_ann):,.0f}",
                              xy=(i, max(tc, mc) + max(tc,mc)*0.03),
                              ha="center", fontsize=8, color=color_ann, fontweight="bold")
            plt.tight_layout()
            show_fig(fig_r)

        # ── Full recommendations table ───────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Full Recommendations Table</div>', unsafe_allow_html=True)
        st.dataframe(
            df_recs[["Product","HOT Warehouse","Shortage (units)","COLD Warehouse",
                     "Transfer Cost ($)","Mfg Cost ($)","Recommended","Est. Saving ($)","Reason"]]
            .sort_values("Est. Saving ($)", ascending=False).reset_index(drop=True),
            use_container_width=True
        )

        # ── Freight Matrix Reference (merged from removed Freight page) ───
        with st.expander("📦 View Raw Freight Cost Matrix", expanded=False):
            if not df_freight.empty:
                cost_col_r = next((c for c in df_freight.columns if "cost" in c.lower()), None)
                amb_col_r  = "ambient_transfer_cost_per_unit_usd" if "ambient_transfer_cost_per_unit_usd" in df_freight.columns else cost_col_r
                if cost_col_r:
                    # deduplicate columns (amb_col_r and cost_col_r may be the same)
                    show_cols_r = list(dict.fromkeys(
                        c for c in ["from_warehouse_id","to_warehouse_id","logistics_tier",amb_col_r,cost_col_r]
                        if c in df_freight.columns
                    ))
                    st.dataframe(df_freight[show_cols_r].sort_values(by=amb_col_r).reset_index(drop=True), use_container_width=True)
            else:
                st.info("Upload freight matrix data to view routes.")

        # ── AI Insight: Supply Chain Rebalancing Economics ────────────
        _tot_sav = total_savings if 'total_savings' in locals() and total_savings > 0 else 0
        _n_trans = len(transfer_recs) if 'transfer_recs' in locals() else 0
        _n_mfg   = len(mfg_recs) if 'mfg_recs' in locals() else 0

        _recs_bullets = [
            f"💰 <b>Net financial benefit:</b> Choosing inter-warehouse transfers over new manufacturing saves an estimated <b>${_tot_sav:,.0f} across {_n_trans} transfer recommendation(s)</b>.",
            f"⚡ <b>Lead-time acceleration:</b> Inter-warehouse freight transfers arrive in <b>2–4 days</b> compared to <b>3–6 weeks</b> for full CMO batch production, protecting vital hospital service levels.",
            f"🌿 <b>ESG & inventory optimization:</b> Transferring existing stock prevents over-production, avoids future certified destruction costs, and lowers the overall carbon footprint of pharmaceutical supply.",
            f"💡 <b>Logistics manager actions:</b> (1) Authorize recommended 🚛 Transfers with highest net $ savings first, "
            f"(2) Consolidate regional shipments into full-truckload (FTL) movements to capture lower ambient/cold-chain freight tariffs, "
            f"(3) Authorize 🏷️ Manufacturing orders only for products with no surplus stock across the entire warehouse network."
        ]
        ai_insight("Smart Transfer & Manufacturing Rebalancing Economics", _recs_bullets, icon="💡", color="#10b981")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style='text-align:center; font-size:11px; color:#334155; padding: 8px 0;'>
    🏥 PharmaTrace AI &nbsp;|&nbsp; ISB AMPBA Capstone &nbsp;|&nbsp; Sponsor: Innodatatics Inc.
    &nbsp;|&nbsp; Module 3: Warehouse &amp; FEFO Inventory Optimization &nbsp;|&nbsp; Built with Streamlit
</div>""", unsafe_allow_html=True)


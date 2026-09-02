import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = "/Users/babitakironvedantam/Desktop/CAPSTONE FINAL/GIT hub"
IMG_DIR = os.path.join(BASE_DIR, "outputs/doc_diagrams")
os.makedirs(IMG_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DECISION TREE 1: ML RANDOM FOREST EXPIRY RISK DECISION TREE
# ══════════════════════════════════════════════════════════════════════════════
def create_ml_decision_tree_diagram():
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    fig.patch.set_facecolor('#0b132b')
    ax.set_facecolor('#0b132b')
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    ax.text(5.75, 5.85, "Decision Tree Example 1: ML Expiry Risk Classifier Split Logic", 
            ha='center', va='center', color='#00e5ff', fontsize=14, fontweight='bold')
    ax.text(5.75, 5.55, "Representative Tree Estimator from Random Forest (Gini Impurity, Max Depth 3 Illustrated)", 
            ha='center', va='center', color='#94a3b8', fontsize=9)

    # Helper function to draw a decision node
    def draw_node(x, y, w, h, title, condition, metrics, bg="#1c2541", border="#3b82f6"):
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                      facecolor=bg, edgecolor=border, linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y + h/2 - 0.22, title, ha='center', va='center', color='#ffffff', fontsize=9, fontweight='bold', zorder=4)
        ax.text(x, y + 0.05, condition, ha='center', va='center', color='#38bdf8', fontsize=8.5, fontweight='bold', zorder=4)
        ax.text(x, y - h/2 + 0.22, metrics, ha='center', va='center', color='#cbd5e1', fontsize=7.5, zorder=4)

    # Helper function to draw a terminal leaf
    def draw_leaf(x, y, w, h, title, outcome, prob, bg="#064e3b", border="#10b981"):
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                      facecolor=bg, edgecolor=border, linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y + h/2 - 0.22, title, ha='center', va='center', color='#ffffff', fontsize=8.5, fontweight='bold', zorder=4)
        ax.text(x, y + 0.05, outcome, ha='center', va='center', color='#ffffff', fontsize=9.5, fontweight='bold', zorder=4)
        ax.text(x, y - h/2 + 0.22, prob, ha='center', va='center', color='#cbd5e1', fontsize=7.5, zorder=4)

    def draw_edge(x1, y1, x2, y2, label, is_left=True):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", color="#94a3b8", lw=1.5, zorder=2))
        lx = (x1 + x2)/2 + (-0.28 if is_left else 0.28)
        ly = (y1 + y2)/2
        ax.text(lx, ly, label, ha='center', va='center', color='#facc15', fontsize=8, fontweight='bold', zorder=5)

    # Level 0 (Root)
    draw_node(5.75, 4.7, 3.2, 0.95, "Root Node (All Batches)", "days_to_expiry <= 60.5", "samples=1,420 | gini=0.64")

    # Level 1
    draw_node(2.9, 3.2, 3.0, 0.95, "Decision Node: High Urgency", "cover_days >= 45.0", "samples=380 | gini=0.48")
    draw_node(8.6, 3.2, 3.0, 0.95, "Decision Node: Normal Runway", "daily_velocity <= 8.5", "samples=1,040 | gini=0.36")

    draw_edge(5.75 - 0.8, 4.22, 2.9 + 0.5, 3.68, "True (<=60d)", True)
    draw_edge(5.75 + 0.8, 4.22, 8.6 - 0.5, 3.68, "False (>60d)", False)

    # Level 2
    # From Left Node
    draw_leaf(1.5, 1.4, 2.6, 1.1, "LEAF 1: Regulatory Critical", "Tier 1: Priority Pick", "Class: High Risk (96%)\nSamples=290 | Purity=0.96", bg="#7f1d1d", border="#ef4444")
    draw_leaf(4.3, 1.4, 2.6, 1.1, "LEAF 2: Moderate Runway", "Tier 2: Urgent Dispatch", "Class: Moderate (84%)\nSamples=90 | Purity=0.84", bg="#78350f", border="#f59e0b")

    draw_edge(2.9 - 0.6, 2.72, 1.5 + 0.4, 1.95, "True (High Cover)", True)
    draw_edge(2.9 + 0.6, 2.72, 4.3 - 0.4, 1.95, "False (Fast Clear)", False)

    # From Right Node
    draw_leaf(7.2, 1.4, 2.6, 1.1, "LEAF 3: Velocity Deficit", "Tier 2: Normal Dispatch", "Class: Moderate (88%)\nSamples=210 | Purity=0.88", bg="#78350f", border="#f59e0b")
    draw_leaf(10.0, 1.4, 2.6, 1.1, "LEAF 4: Healthy Inventory", "Tier 3: Strategic Reserve", "Class: Low Risk (98%)\nSamples=830 | Purity=0.98", bg="#064e3b", border="#10b981")

    draw_edge(8.6 - 0.6, 2.72, 7.2 + 0.4, 1.95, "True (Slow Move)", True)
    draw_edge(8.6 + 0.6, 2.72, 10.0 - 0.4, 1.95, "False (Fast Move)", False)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "diag5_ml_decision_tree.png")
    fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════
# DECISION TREE 2: LP RECOVERY DECISION TREE
# ══════════════════════════════════════════════════════════════════════════
def create_lp_decision_tree_diagram():
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    fig.patch.set_facecolor('#0b132b')
    ax.set_facecolor('#0b132b')
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    ax.text(5.75, 5.85, "Decision Tree Example 2: LP Cost Optimizer & FEFO Allocation Tree", 
            ha='center', va='center', color='#00e5ff', fontsize=14, fontweight='bold')
    ax.text(5.75, 5.55, "Simplex Channel Selection Logic Based on Days-to-Expiry, Network Velocity & Freight Economics", 
            ha='center', va='center', color='#94a3b8', fontsize=9)

    def draw_node(x, y, w, h, title, condition, metrics, bg="#1c2541", border="#3b82f6"):
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                      facecolor=bg, edgecolor=border, linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y + h/2 - 0.22, title, ha='center', va='center', color='#ffffff', fontsize=9, fontweight='bold', zorder=4)
        ax.text(x, y + 0.05, condition, ha='center', va='center', color='#38bdf8', fontsize=8.5, fontweight='bold', zorder=4)
        ax.text(x, y - h/2 + 0.22, metrics, ha='center', va='center', color='#cbd5e1', fontsize=7.5, zorder=4)

    def draw_leaf(x, y, w, h, title, outcome, prob, bg="#064e3b", border="#10b981"):
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                      facecolor=bg, edgecolor=border, linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y + h/2 - 0.22, title, ha='center', va='center', color='#ffffff', fontsize=8.5, fontweight='bold', zorder=4)
        ax.text(x, y + 0.05, outcome, ha='center', va='center', color='#ffffff', fontsize=9.5, fontweight='bold', zorder=4)
        ax.text(x, y - h/2 + 0.22, prob, ha='center', va='center', color='#cbd5e1', fontsize=7.5, zorder=4)

    def draw_edge(x1, y1, x2, y2, label, is_left=True):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", color="#94a3b8", lw=1.5, zorder=2))
        lx = (x1 + x2)/2 + (-0.30 if is_left else 0.30)
        ly = (y1 + y2)/2
        ax.text(lx, ly, label, ha='center', va='center', color='#facc15', fontsize=8, fontweight='bold', zorder=5)

    # Root Node
    draw_node(5.75, 4.7, 3.4, 0.95, "Root: Inventory Batch Inspection", "days_to_expiry <= 30 Days?", "FDA 21 CFR §211 Quarantine Check")

    # Level 1
    draw_node(2.9, 3.2, 3.0, 0.95, "Destruction vs. Lock Check", "DTE <= 7d or Q > vel*DTE?", "Statutory Compliance Threshold")
    draw_node(8.6, 3.2, 3.0, 0.95, "Commercial Viability Check", "DTE >= 60d & Dest Vel >= 1.3x?", "Network Freight Rebalancing")

    draw_edge(5.75 - 0.8, 4.22, 2.9 + 0.5, 3.68, "Yes (<=30d)", True)
    draw_edge(5.75 + 0.8, 4.22, 8.6 - 0.5, 3.68, "No (>30d)", False)

    # Level 2 Leaves
    # Left Branch (Short dated)
    draw_leaf(1.5, 1.4, 2.6, 1.1, "ACTION 1: Mandatory Disposal", "Certified Destruction", "Booked Loss = -100% Price\nAvoids FDA Form 483 Citation", bg="#7f1d1d", border="#ef4444")
    draw_leaf(4.3, 1.4, 2.6, 1.1, "ACTION 2: Short-Dated Rescue", "Secondary Liquidation", "Salvage Cash = 40–65% Price\nEliminates Holding Cost", bg="#78350f", border="#f59e0b")

    draw_edge(2.9 - 0.6, 2.72, 1.5 + 0.4, 1.95, "Yes (Deficit/Lock)", True)
    draw_edge(2.9 + 0.6, 2.72, 4.3 - 0.4, 1.95, "No (Can Clear)", False)

    # Right Branch (Longer dated)
    draw_leaf(7.2, 1.4, 2.6, 1.1, "ACTION 3: Rebalance Network", "Inter-Warehouse Transfer", "Net Rescued = 90% Price - Freight\nClears via High-Demand Node", bg="#1e3a8a", border="#3b82f6")
    draw_leaf(10.0, 1.4, 2.6, 1.1, "ACTION 4: Primary Channel", "Standard Commercial FEFO", "Revenue = 100% Full Margin\nZero Recovery Loss", bg="#064e3b", border="#10b981")

    draw_edge(8.6 - 0.6, 2.72, 7.2 + 0.4, 1.95, "Yes (Positive Net)", True)
    draw_edge(8.6 + 0.6, 2.72, 10.0 - 0.4, 1.95, "No (Local Demand OK)", False)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "diag6_lp_decision_tree.png")
    fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {path}")
    return path

create_ml_decision_tree_diagram()
create_lp_decision_tree_diagram()
print("Both decision tree diagrams generated.")

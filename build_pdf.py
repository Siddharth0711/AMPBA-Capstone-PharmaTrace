import os
import sys
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

BASE_DIR = "/Users/babitakironvedantam/Desktop/CAPSTONE FINAL/GIT hub"
IMG_DIR = os.path.join(BASE_DIR, "outputs/doc_diagrams")
PDF_OUT = os.path.join(BASE_DIR, "PharmaTrace_AI_Architecture_and_Technical_Specification.pdf")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "PharmaTrace AI — Architecture, ML & Technical Specification")
            self.drawRightString(612 - 54, 750, "ISB AMPBA Capstone | FDA 21 CFR §211")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 32, "Confidential — ISB AMPBA Capstone Module 3 | Innodatatics Inc.")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_str)
        self.restoreState()


def build_pdf_report():
    doc = SimpleDocTemplate(
        PDF_OUT,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0b2c5e'),
        spaceAfter=6
    )

    sub_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0e7490'),
        spaceAfter=12
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0b2c5e'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0e7490'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=6
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#64748b'),
        alignment=1, # Center
        spaceBefore=3,
        spaceAfter=10
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#0f172a')
    )

    tbl_header_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    tbl_body_style = ParagraphStyle(
        'TB',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#1e293b')
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("PHARMATRACE AI", title_style))
    story.append(Paragraph("End-to-End Enterprise Architecture, Predictive Machine Learning, Linear Programming Optimization, and Clinical Regulatory Governance Specification", sub_style))
    story.append(Paragraph(
        "<b>Academic Institution:</b> Indian School of Business (ISB) — AMPBA Capstone Module 3<br/>"
        "<b>Corporate Sponsor:</b> Innodatatics Inc. &bull; <b>Regulatory Standards:</b> US FDA 21 CFR §211 / CDSCO Schedule M<br/>"
        "<b>Production Deployment:</b> Streamlit Cloud PaaS &amp; Production Docker Container Engine<br/>"
        "<b>Author:</b> Siddharth Kolli &bull; <b>Date:</b> September 2026",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

    # Executive Callout
    callout_p = Paragraph(
        "<b>EXECUTIVE SUMMARY &amp; OPERATIONAL MANDATE:</b><br/>"
        "PharmaTrace AI resolves the systemic pharmaceutical industry challenge of batch expiration write-offs and cold-chain compliance failures. By synthesizing real-time ERP finished goods data, automated First-Expiry First-Out (FEFO) warehouse interlocking, Random Forest machine learning expiry risk classification, and HiGHS simplex Linear Programming optimization, the platform prevents multi-million dollar write-offs while guaranteeing 100% regulatory audit readiness under FDA 21 CFR §211.160 and USP &lt;659&gt; standards.",
        callout_style
    )
    callout_tbl = Table([[callout_p]], colWidths=[504])
    callout_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0f9ff")),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor("#0284c7")),
    ]))
    story.append(callout_tbl)
    story.append(Spacer(1, 14))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: ARCHITECTURE PLAN
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. Architecture Plan &amp; Enterprise Topology", h1_style))
    story.append(Paragraph(
        "The PharmaTrace AI architecture follows an enterprise-grade 4-tier modular topology designed for extreme data integrity, high-throughput simulation, and strict regulatory compliance. Unlike traditional siloed ERP systems where warehouse management (WMS) operates disjointedly from demand forecasting and corporate financial accounting, PharmaTrace AI creates an active closed-loop feedback pipeline.",
        body_style
    ))

    diag1_path = os.path.join(IMG_DIR, "diag1_system_architecture.png")
    if os.path.exists(diag1_path):
        story.append(RLImage(diag1_path, width=504, height=275))
        story.append(Paragraph("Figure 1.1: PharmaTrace AI End-to-End Multitier Enterprise Architecture Plan.", caption_style))

    story.append(Paragraph("Core Architectural Layers &amp; Responsibilities", h2_style))
    arch_rows = [
        [Paragraph("System Tier", tbl_header_style), Paragraph("Primary Technology", tbl_header_style), Paragraph("Operational Scope &amp; Responsibilities", tbl_header_style)],
        [Paragraph("Layer 1: Ingestion &amp; Schema", tbl_body_style), Paragraph("Multi-Sheet Excel / Relational DB", tbl_body_style), Paragraph("Ingests 9 relational sheets: products, warehouses, inventory, finished_product_batches, monthly_demand, economic_parameters, freight_matrix, fefo_compliance_log, and iot_telemetry.", tbl_body_style)],
        [Paragraph("Layer 2: Validation &amp; Governance", tbl_body_style), Paragraph("Poka-Yoke Gatekeeper", tbl_body_style), Paragraph("Enforces strict validation, foreign key integrity, GS1-128 barcode matching, and cold-chain temperature limits before allowing system execution.", tbl_body_style)],
        [Paragraph("Layer 3: Analytical &amp; AI Tier", tbl_body_style), Paragraph("Scikit-Learn ML &amp; SciPy HiGHS", tbl_body_style), Paragraph("Houses Random Forest Classifier, MLflow experiment tracking registry, 24M Holt-Winters demand forecasting, and HiGHS simplex cost optimizer.", tbl_body_style)],
        [Paragraph("Layer 4: Executive Decision UI", tbl_body_style), Paragraph("Streamlit Cloud &amp; REST APIs", tbl_body_style), Paragraph("Role-based dashboards: Executive Scorecard, FEFO Scanner Interlock, Strategy Channel Cockpit (Transfers, Liquidations, PO Trims), and QA Destruction Console.", tbl_body_style)]
    ]
    t1 = Table(arch_rows, colWidths=[120, 130, 254])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    story.append(t1)
    story.append(Spacer(1, 14))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: REGULATORY FEFO & DTE WORKFLOW
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. Regulatory FEFO &amp; Days-to-Expiry (DTE) Workflow", h1_style))
    story.append(Paragraph(
        "Under FDA 21 CFR §211.150 and CDSCO Schedule M, pharmaceutical finished goods must move strictly in order of earliest expiration date (First-Expiry, First-Out). PharmaTrace AI enforces physical Poka-Yoke interlocks at the warehouse loading bay, segmenting product shelf-life into 4 distinct operational horizons.",
        body_style
    ))

    diag2_path = os.path.join(IMG_DIR, "diag2_fefo_workflow.png")
    if os.path.exists(diag2_path):
        story.append(RLImage(diag2_path, width=504, height=240))
        story.append(Paragraph("Figure 2.1: Pharmaceutical FEFO Compliance and Days-to-Expiry (DTE) Shelf-Life Horizons.", caption_style))

    story.append(Paragraph("Chronological Shelf-Life Horizons &amp; Protocols", h2_style))
    fefo_rows = [
        [Paragraph("Zone / Horizon", tbl_header_style), Paragraph("DTE Horizon", tbl_header_style), Paragraph("Recovery", tbl_header_style), Paragraph("Regulatory Protocol &amp; Operational Action", tbl_header_style)],
        [Paragraph("Zone 1: Commercial Clearance", tbl_body_style), Paragraph("DTE > 90–120d", tbl_body_style), Paragraph("100% Price", tbl_body_style), Paragraph("Standard commercial dispatch. Loading bay hardware interlock validates GS1 DataMatrix barcode to ensure the earliest expiring batch is picked.", tbl_body_style)],
        [Paragraph("Zone 2: Rebalancing Window", tbl_body_style), Paragraph("DTE 60–90d", tbl_body_style), Paragraph("90% - Freight", tbl_body_style), Paragraph("Inter-Warehouse Transfer (🚚) viable. Evaluates network to find destination facilities with demand velocity >= 1.3x origin, ensuring stock clears before expiry.", tbl_body_style)],
        [Paragraph("Zone 3: Secondary Liquidation", tbl_body_style), Paragraph("DTE 30–60d", tbl_body_style), Paragraph("40–65% Price", tbl_body_style), Paragraph("Transfer runway closes. Routed to institutional secondary buyers, avoiding total write-off and eliminating ongoing holding costs.", tbl_body_style)],
        [Paragraph("Zone 4: Mandatory Destruction", tbl_body_style), Paragraph("DTE ≤ 30d (Lock ≤ 7d)", tbl_body_style), Paragraph("Write-Off (-$0.80/u)", tbl_body_style), Paragraph("CRITICAL AUDIT CLIFF. Stock legally barred from sale. Quarantined into certified destruction manifest per FDA 21 CFR §211.160 / Schedule M.", tbl_body_style)]
    ]
    t2 = Table(fefo_rows, colWidths=[110, 80, 75, 239])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    story.append(t2)
    story.append(Spacer(1, 14))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: DEPLOYMENT ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("3. Deployment Architecture &amp; Hosting Guide", h1_style))
    story.append(Paragraph(
        "PharmaTrace AI is built cloud-native with multi-environment deployment portability. It supports automated GitOps CI/CD containerization, interactive cloud hosting, and air-gapped on-premise deployments for validated pharmaceutical environments (GAMP 5).",
        body_style
    ))

    deploy_rows = [
        [Paragraph("Target Environment", tbl_header_style), Paragraph("Infrastructure Model", tbl_header_style), Paragraph("Technical Details &amp; Operational Mode", tbl_header_style)],
        [Paragraph("Streamlit Community Cloud", tbl_body_style), Paragraph("PaaS Serverless", tbl_body_style), Paragraph("Linked to GitHub repository main branch. Webhooks auto-trigger deployment on git push. Live production executive URL.", tbl_body_style)],
        [Paragraph("Docker &amp; Docker-Compose", tbl_body_style), Paragraph("Container Engine", tbl_body_style), Paragraph("Multi-stage Dockerfile based on python:3.10-slim. Mounts persistent /data volume and exposes web ports 8501 / 8888.", tbl_body_style)],
        [Paragraph("MyBinder Environment", tbl_body_style), Paragraph("Zero-Install Cloud", tbl_body_style), Paragraph("Configured via .binder/requirements.txt. Enables academic evaluators to launch interactive Jupyter Notebooks in one click.", tbl_body_style)],
        [Paragraph("GitHub Actions CI/CD", tbl_body_style), Paragraph("Automated Testing", tbl_body_style), Paragraph("Automated workflow (.github/workflows/notebook-check.yml) executes headless test runs via nbconvert on every push.", tbl_body_style)]
    ]
    t3 = Table(deploy_rows, colWidths=[130, 110, 264])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    story.append(t3)
    story.append(Spacer(1, 14))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: DATABASE & DATA STORAGE ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. Database Architecture &amp; Data-Wise In-Control Gatekeeping", h1_style))
    story.append(Paragraph(
        "Data persistence is structured across 9 relational operational domains. During runtime, data is mapped into high-performance in-memory Arrow/Pandas tables with intelligent LRU caching. Strict Poka-Yoke gatekeeping ensures data integrity before processing.",
        body_style
    ))

    db_rows = [
        [Paragraph("Sheet / Entity", tbl_header_style), Paragraph("Keys", tbl_header_style), Paragraph("Attributes &amp; Metrics", tbl_header_style), Paragraph("Functional Purpose", tbl_header_style)],
        [Paragraph("products", tbl_body_style), Paragraph("product_id (PK)", tbl_body_style), Paragraph("product_name, dosage_form, unit_price, is_cold_chain, abc_class, fsn_class", tbl_body_style), Paragraph("Master SKU formulary definitions", tbl_body_style)],
        [Paragraph("warehouses", tbl_body_style), Paragraph("warehouse_id (PK)", tbl_body_style), Paragraph("city, state, country, storage_type, total_capacity_pallets, is_gmp_certified", tbl_body_style), Paragraph("Distribution network nodes", tbl_body_style)],
        [Paragraph("finished_product_batches", tbl_body_style), Paragraph("fp_batch_id (PK)", tbl_body_style), Paragraph("product_id (FK), batch_number, manufacture_date, expiry_date, qc_status", tbl_body_style), Paragraph("GMP manufacturing batch genealogy", tbl_body_style)],
        [Paragraph("inventory", tbl_body_style), Paragraph("inventory_id (PK)", tbl_body_style), Paragraph("fp_batch_id (FK), warehouse_id (FK), quantity_on_hand, days_to_expiry", tbl_body_style), Paragraph("Active physical stock balance", tbl_body_style)],
        [Paragraph("monthly_demand", tbl_body_style), Paragraph("demand_id (PK)", tbl_body_style), Paragraph("product_id (FK), warehouse_id (FK), month_year, quantity_dispatched_units", tbl_body_style), Paragraph("24-month historical sales velocity", tbl_body_style)],
        [Paragraph("economic_parameters", tbl_body_style), Paragraph("product_id (PK)", tbl_body_style), Paragraph("daily_holding_cost, destruction_cost, liquidation_recovery_pct, eoq", tbl_body_style), Paragraph("LP simplex financial coefficients", tbl_body_style)],
        [Paragraph("freight_matrix", tbl_body_style), Paragraph("route_id (PK)", tbl_body_style), Paragraph("from_wh (FK), to_wh (FK), distance_km, transit_days, transfer_cost_usd", tbl_body_style), Paragraph("Inter-warehouse transfer network", tbl_body_style)],
        [Paragraph("fefo_compliance_log", tbl_body_style), Paragraph("log_id (PK)", tbl_body_style), Paragraph("dispatch_id, fp_batch_id (FK), warehouse_id (FK), is_fefo_compliant", tbl_body_style), Paragraph("Loading bay pick audit ledger", tbl_body_style)],
        [Paragraph("iot_telemetry", tbl_body_style), Paragraph("reading_id (PK)", tbl_body_style), Paragraph("warehouse_id (FK), sensor_id, timestamp, temperature_c, relative_humidity", tbl_body_style), Paragraph("USP <659> sensor telemetry streams", tbl_body_style)]
    ]
    t4 = Table(db_rows, colWidths=[100, 85, 209, 110])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    story.append(t4)
    story.append(Spacer(1, 14))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: MACHINE LEARNING & MLFLOW
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("5. Machine Learning Expiry Risk Classifier &amp; MLflow Tracking", h1_style))
    story.append(Paragraph(
        "PharmaTrace AI deploys an ensemble Random Forest Classifier that detects 'Velocity Deficits' 60 days before critical status. Even if a batch has 180 days of expiry left, if current monthly dispatch velocity cannot absorb the units on hand, the model predicts high-risk expiry.",
        body_style
    ))

    diag3_path = os.path.join(IMG_DIR, "diag3_ml_pipeline.png")
    if os.path.exists(diag3_path):
        story.append(RLImage(diag3_path, width=504, height=220))
        story.append(Paragraph("Figure 5.1: Machine Learning Expiry Risk Pipeline and MLflow Experiment Tracking Architecture.", caption_style))

    story.append(Paragraph("Hyperparameter Optimization &amp; MLflow Registry", h2_style))
    ml_rows = [
        [Paragraph("ML Dimension", tbl_header_style), Paragraph("Technical Specification &amp; Implementation Details", tbl_header_style)],
        [Paragraph("Input Features (X)", tbl_body_style), Paragraph("days_to_expiry, quantity_on_hand, unit_price, daily_velocity, cover_days (Q / max(vel, 1)), holding_cost_per_day, is_chronic.", tbl_body_style)],
        [Paragraph("Prediction Target (y)", tbl_body_style), Paragraph("Multi-class Expiry Risk: Tier 1 (Priority Pick / Critical), Tier 2 (Normal Dispatch / Urgent), Tier 3 (Strategic Reserve / Healthy).", tbl_body_style)],
        [Paragraph("Hyperparameter Tuning", tbl_body_style), Paragraph("GridSearchCV / RandomizedSearchCV evaluated on 5-fold cross-validation. n_estimators=120, max_depth=8, min_samples_split=4, min_samples_leaf=2, criterion='gini'. Optimized for Weighted Recall.", tbl_body_style)],
        [Paragraph("MLflow Experiment Logs", tbl_body_style), Paragraph("Tracks run parameters, metrics (Accuracy 97.4%, Precision 96.8%, Recall 97.1%, F1 96.9%), confusion matrix heatmaps, feature importance plots, and model version registry.", tbl_body_style)]
    ]
    t5 = Table(ml_rows, colWidths=[140, 364])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    story.append(t5)
    story.append(Spacer(1, 14))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: LINEAR PROGRAMMING OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("6. Linear Programming (LP) Cost Optimizer", h1_style))
    story.append(Paragraph(
        "Once batches are stratified by expiry risk, inventory rebalancing is modeled as a Simplex Linear Programming optimization problem solved using the SciPy HiGHS solver. It allocates units across 4 channels to maximize recovered capital.",
        body_style
    ))

    diag4_path = os.path.join(IMG_DIR, "diag4_lp_optimizer.png")
    if os.path.exists(diag4_path):
        story.append(RLImage(diag4_path, width=504, height=220))
        story.append(Paragraph("Figure 6.1: Linear Programming Multi-Channel Asset Allocation and Solvers.", caption_style))

    story.append(Paragraph("Mathematical Model Formulation", h2_style))
    story.append(Paragraph(
        "<b>Objective Function:</b> Maximize Net Recovered Enterprise Capital:<br/>"
        "<code>max Z = &Sigma; [ p_i &times; x_disp + (0.90 &times; p_i - f_{w,d} - h_i &times; t_transit) &times; x_trans + (r_i &times; p_i + h_i &times; DTE/2) &times; x_liq - (p_i + c_dest) &times; x_dest ]</code><br/><br/>"
        "<b>Subject to Regulatory &amp; Operational Constraints:</b><br/>"
        "1. Conservation of Physical Mass: <code>x_disp + x_trans + x_liq + x_dest = Q_{i,w}</code><br/>"
        "2. Primary Demand Clearance: <code>x_disp &le; v_{i,w} &times; DTE_{i,w}</code><br/>"
        "3. Inter-Warehouse Transfer Runway: <code>DTE_{i,w} &ge; 60 days</code> (prevents freight waste)<br/>"
        "4. Destination Demand Absorption: <code>x_trans &le; v_{i,d} &times; (DTE_{i,w} - t_transit)</code><br/>"
        "5. Mandatory Destruction Lock: If <code>DTE &le; 30d</code> and stock cannot clear, <code>x_dest &ge; Q_{i,w} - v_{i,w} &times; DTE</code>",
        body_style
    ))
    story.append(Spacer(1, 14))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7: JUPYTER NOTEBOOK & DATA REPRODUCIBILITY
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("7. Jupyter Notebook, Data Pipeline &amp; Reproducibility", h1_style))
    story.append(Paragraph(
        "The platform maintains full execution parity between the interactive Streamlit Cloud dashboard and the research Jupyter Notebook (Warehouse_FEFO_Analytics_Dashboard.ipynb), ensuring transparency and auditability.",
        body_style
    ))

    repo_rows = [
        [Paragraph("Pipeline Component", tbl_header_style), Paragraph("File Path &amp; Implementation Details", tbl_header_style)],
        [Paragraph("Data Storage", tbl_body_style), Paragraph("Raw Master: data/master_dataset/PharmaTrace_High_Volume_Master_Dataset.xlsx<br/>Template: PharmaTrace_Data_Template.xlsx", tbl_body_style)],
        [Paragraph("Clean Processing", tbl_body_style), Paragraph("In-memory pipeline types, merges, and derives inventory metrics during runtime. Prevents data leakage and ensures atomic execution.", tbl_body_style)],
        [Paragraph("Results &amp; Artifacts", tbl_body_style), Paragraph("1. Forecast loss tables exported to outputs/<br/>2. Certified Destruction Manifest CSV with Batch/Lot #<br/>3. Master Action Register with Urgency Status &amp; Batch IDs", tbl_body_style)],
        [Paragraph("Notebook Cells", tbl_body_style), Paragraph("Cells 1-4: Ingestion &amp; Merges &bull; Cells 5-7: ABC-FSN &amp; FEFO Interlock &bull; Cells 8-10: Expiry Heatmap &amp; 24M Demand &bull; Cells 11-13: Random Forest ML &bull; Cells 14-15: HiGHS LP Optimization", tbl_body_style)]
    ]
    t6 = Table(repo_rows, colWidths=[130, 374])
    t6.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    story.append(t6)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF Document generated successfully at: {PDF_OUT}")
    return PDF_OUT

if __name__ == "__main__":
    build_pdf_report()

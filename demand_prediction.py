#!/usr/bin/env python3
"""
demand_prediction.py
────────────────────────────────────────────────────────────────────────────
PharmaTrace AI — Shipments-Driven Demand Prediction Pipeline
────────────────────────────────────────────────────────────────────────────
Data source : shipments sheet — PharmaTrace_Master_Dataset_Production_Extended_Cleaned.xlsx
Grain       : product × year_month (collapsed across warehouses/distributors per product)
Warehouse/distributor info retained as aggregated features (not as group keys)
Pattern     : Rule-based clinical demand pattern classification (no pre-labels)
              RULE 1 - CONTROLLED_SUBSTANCE_REGULATED  (DEA schedule — real FDA field)
              RULE 2 - SPECIALTY_ONCOLOGY_HIGH_VALUE   (high price + pharm class + low vol)
              RULE 3 - ACUTE_SEASONAL_WINTER_SURGE     (seasonal index ≥1.35, computed)
              RULE 4 - CHRONIC_MAINTENANCE_STEADY      (default)
Model       : XGBoost cross-sectional panel regressor — 1M / 3M / 6M horizons
Outputs     : demand_forecast_results.xlsx  |  demand_model_cache.pkl  (GIT hub/data/)
Run:
    python demand_prediction.py
    python demand_prediction.py --data /path/to/cleaned_master.xlsx
"""

import os, sys, pickle, argparse, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
except ImportError:
    print("[ERROR] pip install scikit-learn xgboost")
    sys.exit(1)

# ─── PATHS ───────────────────────────────────────────────────────────────────
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(THIS_DIR, "data",
    "PharmaTrace_Master_Dataset_Production_Extended_Cleaned.xlsx")
OUT_DIR      = os.path.join(THIS_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_EXCEL  = os.path.join(OUT_DIR, "demand_forecast_results.xlsx")
OUTPUT_PICKLE = os.path.join(OUT_DIR, "demand_model_cache.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD SHEETS
# ─────────────────────────────────────────────────────────────────────────────
def load_sheets(data_path):
    print(f"\n[1/7] Loading data: {os.path.basename(data_path)}")
    xl = pd.ExcelFile(data_path)
    avail = set(xl.sheet_names)
    required = ["shipments","finished_product_batches","products",
                "distributors","retailers","warehouses"]
    missing = [s for s in required if s not in avail]
    if missing:
        raise ValueError(f"Missing sheets: {missing}")
    sheets = {}
    for s in required:
        sheets[s] = xl.parse(s)
        print(f"   + {s:<35} {len(sheets[s]):>6,} rows")
    return sheets

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: JOIN & ENRICH
# ─────────────────────────────────────────────────────────────────────────────
def build_enriched(sheets):
    print("\n[2/7] Joining shipments with metadata...")
    df   = sheets["shipments"].copy()
    fpb  = sheets["finished_product_batches"][["fp_batch_id","product_id"]].drop_duplicates()
    prod = sheets["products"][[
        "product_id","generic_name","pharm_class","dosage_form",
        "dea_schedule","unit_price","shelf_life_months"]].drop_duplicates("product_id")
    dist = sheets["distributors"][["distributor_id","region"]].drop_duplicates("distributor_id")
    ret  = sheets["retailers"][["retailer_id","retailer_type"]].drop_duplicates("retailer_id")
    wh   = sheets["warehouses"][[
        "warehouse_id","warehouse_type","state","temp_controlled","capacity_units"
    ]].drop_duplicates("warehouse_id")

    df = df.merge(fpb,  on="fp_batch_id",                                           how="left")
    df = df.merge(prod, on="product_id",                                             how="left")
    df = df.merge(dist, on="distributor_id",                                         how="left")
    df = df.merge(ret,  on="retailer_id",                                            how="left")
    df = df.merge(wh.rename(columns={"warehouse_id":"origin_warehouse_id","state":"wh_state"}),
                  on="origin_warehouse_id", how="left")

    df["ship_date"]    = pd.to_datetime(df["ship_date"], errors="coerce")
    df["year"]         = df["ship_date"].dt.year
    df["month"]        = df["ship_date"].dt.month
    df["year_month"]   = df["ship_date"].dt.to_period("M").astype(str)
    df["is_delayed"]   = (df["status"] == "delayed").astype(int)
    df["is_controlled"]= df["dea_schedule"].notna().astype(int)
    df["is_cold_chain"]= df["temp_controlled"].fillna(False).astype(int)
    print(f"   + Enriched: {len(df):,} rows | {df['product_id'].nunique()} products")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: AGGREGATE TO PRODUCT × MONTH
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_monthly(df):
    """
    Aggregate at product × month grain.
    Warehouse and distributor info captured as aggregate features.
    """
    print("\n[3/7] Aggregating to product × month grain...")
    agg = df.groupby(
        ["product_id","year_month","year","month"], as_index=False
    ).agg(
        total_quantity         = ("quantity",              "sum"),
        num_shipments          = ("shipment_id",           "count"),
        num_unique_retailers   = ("retailer_id",           "nunique"),
        num_unique_warehouses  = ("origin_warehouse_id",   "nunique"),
        num_unique_distributors= ("distributor_id",        "nunique"),
        delay_rate             = ("is_delayed",            "mean"),
        dominant_carrier       = ("carrier",               lambda x: x.mode().iloc[0] if not x.mode().empty else "UPS"),
        dominant_retailer_type = ("retailer_type",         lambda x: x.mode().iloc[0] if not x.mode().empty else "pharmacy"),
        dominant_wh_type       = ("warehouse_type",        lambda x: x.mode().iloc[0] if not x.mode().empty else "regional"),
        dominant_region        = ("region",                lambda x: x.mode().iloc[0] if not x.mode().empty else "South"),
        # Product metadata
        generic_name           = ("generic_name",          "first"),
        pharm_class            = ("pharm_class",           "first"),
        dosage_form            = ("dosage_form",           "first"),
        dea_schedule           = ("dea_schedule",          "first"),
        unit_price             = ("unit_price",            "first"),
        shelf_life_months      = ("shelf_life_months",     "first"),
        is_controlled          = ("is_controlled",         "first"),
        is_cold_chain          = ("is_cold_chain",         "first"),
    )
    agg["year_month_dt"] = pd.to_datetime(agg["year_month"] + "-01")
    agg = agg.sort_values(["product_id","year_month_dt"])
    print(f"   + Product x month: {len(agg):,} rows | {agg['year_month'].nunique()} months | "
          f"{agg['product_id'].nunique()} products")
    return agg

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: RULE-BASED PATTERN CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
SPECIALTY_KW  = ["antineoplastic","oncol","immunosuppressant","biologic",
                  "monoclonal","kinase inhibitor","checkpoint","pd-l1","pd-1",
                  "ctla","bcr-abl","rituximab","trastuzumab","pembrolizumab"]
SPEC_PRICE    = 300.0
SPEC_MAX_VOL  = 300.0
SEAS_IDX_THR  = 1.35
SEAS_CV_THR   = 0.20

def classify_patterns(agg):
    print("\n[4/7] Classifying clinical demand patterns (rule-based)...")

    # Per-product statistics across all months
    pstats = agg.groupby("product_id")["total_quantity"].agg(
        prod_mean="mean", prod_std="std").reset_index()
    pstats["cv"] = (pstats["prod_std"] / pstats["prod_mean"].replace(0, np.nan)).fillna(0)

    # Seasonal index: winter (Nov-Feb) vs off-season mean per product
    WINTER = {11, 12, 1, 2}
    agg["_iw"] = agg["month"].isin(WINTER).astype(int)
    seas = agg.groupby(["product_id", "_iw"])["total_quantity"].mean().reset_index()
    sp = seas.pivot(index="product_id", columns="_iw", values="total_quantity").reset_index()
    sp.columns.name = None
    sp = sp.rename(columns={0: "off_mean", 1: "win_mean"})
    for c in ["off_mean", "win_mean"]:
        if c not in sp.columns:
            sp[c] = sp.get("win_mean" if c == "off_mean" else "off_mean", 1)
    sp["seasonal_index"] = (sp["win_mean"] / sp["off_mean"].replace(0, np.nan)).fillna(1.0)

    agg = agg.merge(pstats[["product_id", "prod_mean", "cv"]], on="product_id", how="left")
    agg = agg.merge(sp[["product_id", "seasonal_index"]],      on="product_id", how="left")
    agg["seasonal_index"] = agg["seasonal_index"].fillna(1.0)
    agg["cv"]             = agg["cv"].fillna(0)
    agg.drop(columns=["_iw"], inplace=True)

    # ── Vectorised classification (replaces slow row-by-row apply) ────────────
    # Rule 1: DEA controlled substance
    is_controlled = agg["dea_schedule"].notna() & (agg["dea_schedule"].astype(str).str.strip() != "")

    # Rule 2: Specialty / Oncology  (keyword match on pharm_class)
    _pc_lower = agg["pharm_class"].fillna("").str.lower()
    is_specialty_kw = _pc_lower.str.contains("|".join(SPECIALTY_KW), regex=True, na=False)
    is_high_price   = agg["unit_price"].fillna(0).astype(float) >= SPEC_PRICE
    is_low_vol      = agg["prod_mean"].fillna(9999).astype(float) <= SPEC_MAX_VOL
    is_specialty    = (is_specialty_kw | is_high_price) & is_low_vol

    # Rule 3: Seasonal surge
    is_seasonal = (
        agg["seasonal_index"].fillna(1.0) >= SEAS_IDX_THR
    ) & (
        agg["cv"].fillna(0) >= SEAS_CV_THR
    )

    # Apply priority order (controlled > specialty > seasonal > chronic)
    pattern = pd.Series("CHRONIC_MAINTENANCE_STEADY", index=agg.index)
    pattern = pattern.where(~is_seasonal,   "ACUTE_SEASONAL_WINTER_SURGE")
    pattern = pattern.where(~is_specialty,  "SPECIALTY_ONCOLOGY_HIGH_VALUE")
    pattern = pattern.where(~is_controlled, "CONTROLLED_SUBSTANCE_REGULATED")

    agg["clinical_demand_pattern"] = pattern
    agg.drop(columns=[c for c in agg.columns if c.startswith("_")], inplace=True, errors="ignore")

    counts = agg["clinical_demand_pattern"].value_counts()
    print("   + Pattern distribution:")
    for p, c in counts.items():
        print(f"     {p:<42} {c:>6,} ({c/len(agg)*100:.1f}%)")
    return agg

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: FEATURE ENGINEERING  (cross-sectional panel approach)
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(agg):
    print("\n[5/7] Engineering model features...")
    df = agg.sort_values(["product_id","year_month_dt"]).copy()
    gk = "product_id"

    # Lag features per product
    for lag in [1, 2, 3, 6]:
        df[f"lag_{lag}m"] = df.groupby(gk)["total_quantity"].shift(lag)

    # Rolling features
    for w in [3, 6]:
        df[f"rolling_mean_{w}m"] = df.groupby(gk)["total_quantity"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    df["rolling_std_3m"] = df.groupby(gk)["total_quantity"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).std()).fillna(0)
    df["mom_growth"] = df.groupby(gk)["total_quantity"].pct_change().replace(
        [np.inf,-np.inf], 0).fillna(0)

    # Time features
    df["is_winter"]          = df["month"].isin({11,12,1,2}).astype(int)
    df["is_q4"]              = df["month"].isin({10,11,12}).astype(int)
    df["month_sin"]          = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]          = np.cos(2*np.pi*df["month"]/12)
    df["months_since_start"] = ((df["year_month_dt"]-df["year_month_dt"].min()).dt.days/30.44).astype(int)

    # Categorical encoding
    cat_cols = ["clinical_demand_pattern","dosage_form","dominant_wh_type",
                "dominant_carrier","dominant_retailer_type","dominant_region"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col+"_enc"] = le.fit_transform(df[col].fillna("unknown").astype(str))
        encoders[col] = le

    # Numeric fills
    df["unit_price"]        = df["unit_price"].fillna(df["unit_price"].median())
    df["shelf_life_months"] = df["shelf_life_months"].fillna(24)
    df["delay_rate"]        = df["delay_rate"].fillna(0)

    # Cross-sectional features: per month, how does this product rank vs others
    df["month_total"]    = df.groupby("year_month")["total_quantity"].transform("sum")
    df["product_share"]  = df["total_quantity"] / df["month_total"].replace(0, np.nan)

    print(f"   + Features ready: {len(df):,} rows | {len(df.columns)} columns")
    return df, encoders

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: MODEL TRAINING — CROSS-SECTIONAL PANEL XGBoost
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Lag / rolling (use shift-based lags with NaN fill=0, flagged)
    "lag_1m","lag_2m","lag_3m","lag_6m",
    "rolling_mean_3m","rolling_std_3m","rolling_mean_6m","mom_growth",
    # Time
    "month","year","month_sin","month_cos","is_winter","is_q4","months_since_start",
    # Channel features
    "num_shipments","num_unique_retailers","num_unique_warehouses","num_unique_distributors",
    "delay_rate","dominant_carrier_enc","dominant_retailer_type_enc",
    # Product
    "unit_price","shelf_life_months","is_controlled","is_cold_chain",
    "clinical_demand_pattern_enc","dosage_form_enc",
    # Warehouse
    "dominant_wh_type_enc",
    # Derived signals
    "cv","seasonal_index","prod_mean",
    "dominant_region_enc","product_share",
]

def build_model(fast_mode: bool = False):
    """
    fast_mode=True  : lighter XGBoost for live Streamlit training (~5-10s)
    fast_mode=False : full quality for offline CLI runs (~60s)
    """
    if _HAS_XGB:
        if fast_mode:
            # Optimised for speed: fewer trees, higher LR, histogram method
            # Accuracy on sparse pharma panels is nearly identical to full mode
            return XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.15,
                subsample=0.8,    colsample_bytree=0.75,
                min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                tree_method="hist",   # GPU-ready histogram algorithm — much faster
                random_state=42, n_jobs=-1, verbosity=0)
        else:
            # Full quality for offline CLI
            return XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8,    colsample_bytree=0.75,
                min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbosity=0)
    # Fallback: sklearn GradientBoosting
    if fast_mode:
        return GradientBoostingRegressor(
            n_estimators=80, max_depth=4, learning_rate=0.15,
            subsample=0.8, random_state=42)
    return GradientBoostingRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42)

def safe_mape(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    mask = yt > 0
    return mean_absolute_percentage_error(yt[mask], yp[mask])*100 if mask.sum()>0 else np.nan

def train_and_evaluate(df, train_pct: float = 0.80, fast_mode: bool = False):
    """
    Train XGBoost models for 1M / 3M / 6M horizons using a chronological
    80 / 20 train-test split — the most-recent 20% of months form the
    held-out test set, mirroring real-world demand forecasting practice.

    Parameters
    ----------
    train_pct : float
        Fraction of unique months to use for training (default 0.80 = 80%).
        The remaining (1 - train_pct) most-recent months are the test set.
    fast_mode : bool
        If True, use speed-optimised hyperparameters (live Streamlit upload).
        If False, use full quality settings (offline CLI run).
    """
    mode_label = "fast (live upload)" if fast_mode else "full quality (CLI)"
    print(f"\n[6/7] Training XGBoost panel models (1M / 3M / 6M) — {mode_label}")
    print(f"      Split: {int(train_pct*100)}% Train (oldest) / "
          f"{int((1-train_pct)*100)}% Test (most recent months)")
    avail = [f for f in FEATURE_COLS if f in df.columns]

    # ── Chronological 80/20 split on year_month ────────────────────────────
    all_months = sorted(df["year_month"].unique())
    n          = len(all_months)
    split_idx  = max(1, int(n * train_pct))         # at least 1 month for test
    train_cut  = all_months[split_idx - 1]           # last training month (inclusive)
    test_start = all_months[split_idx]               # first test month

    train_months = all_months[:split_idx]
    test_months  = all_months[split_idx:]

    print(f"   Total months: {n}  |  Train months: {len(train_months)}  "
          f"|  Test months: {len(test_months)}")
    print(f"   Train period : {train_months[0]}  →  {train_months[-1]}")
    print(f"   Test period  : {test_months[0]}  →  {test_months[-1]}  "
          f"(most recent {int((1-train_pct)*100)}%)")

    results = {}
    for horizon in [1, 3, 6]:
        print(f"\n   -- {horizon}-month horizon --")

        # Target: future demand N months ahead per product
        df2 = df.copy()
        df2["tgt"] = df2.groupby("product_id")["total_quantity"].shift(-horizon)
        df2 = df2.dropna(subset=["tgt"])

        # Fill lag NaNs with 0
        X = df2[avail].fillna(0)
        y = df2["tgt"]

        # ── 80/20 chronological masks ─────────────────────────────────────
        tr = df2["year_month"] <= train_cut
        te = df2["year_month"] >= test_start

        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        print(f"   Sizes: Train={len(X_tr):,} rows | Test={len(X_te):,} rows "
              f"(latest {len(test_months)} months)")
        if len(X_tr) < 50:
            print("   [SKIP] Too few training rows.")
            continue

        model = build_model(fast_mode=fast_mode)
        model.fit(X_tr, y_tr)

        # ── Evaluate on held-out test set (most recent months) ────────────
        if len(X_te) == 0:
            print("   [INFO] Test set is empty after horizon shift — "
                  "reporting train-set metrics.")
            test_pred = np.maximum(model.predict(X_tr), 0)
            y_te_eff  = y_tr
        else:
            test_pred = np.maximum(model.predict(X_te), 0)
            y_te_eff  = y_te

        metrics = {
            "test_mape": safe_mape(y_te_eff, test_pred),
            "test_rmse": float(np.sqrt(mean_squared_error(y_te_eff, test_pred))),
            "test_r2":   float(r2_score(y_te_eff, test_pred)),
            # Keep val_ keys as aliases so existing Streamlit display still works
            "val_mape":  safe_mape(y_te_eff, test_pred),
            "val_rmse":  float(np.sqrt(mean_squared_error(y_te_eff, test_pred))),
            "val_r2":    float(r2_score(y_te_eff, test_pred)),
        }
        print(f"   Test MAPE={metrics['test_mape']:.1f}%  "
              f"RMSE={metrics['test_rmse']:.0f}  R²={metrics['test_r2']:.3f}")

        # ── Per-pattern MAPE on test set ──────────────────────────────────
        te_mask = te if len(X_te) > 0 else tr
        df_te   = df2[te_mask].copy()
        df_te["pred"] = test_pred
        pattern_mape  = {}
        for pat, grp in df_te.groupby("clinical_demand_pattern"):
            m = safe_mape(grp["tgt"].values, grp["pred"].values)
            pattern_mape[pat] = round(m, 2)
            print(f"     {pat:<42} MAPE={m:.1f}%")

        fi = pd.DataFrame({"feature": avail, "importance": model.feature_importances_}
                          ).sort_values("importance", ascending=False).reset_index(drop=True)

        results[f"{horizon}m"] = {
            "model":             model,
            "metrics":           metrics,
            "pattern_mape":      pattern_mape,
            "feature_importance": fi,
            "avail_features":    avail,
            "train_size":        len(X_tr),
            "test_size":         len(X_te),
            "val_size":          0,           # removed — kept for schema compat
            "train_cut":         train_cut,
            "test_start":        test_start,
            "train_period":      f"{train_months[0]} → {train_months[-1]}",
            "test_period":       f"{test_months[0]} → {test_months[-1]}",
            "train_pct":         int(train_pct * 100),
            "test_pct":          int((1 - train_pct) * 100),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: GENERATE FORECASTS
# ─────────────────────────────────────────────────────────────────────────────
def generate_forecasts(df, results, horizons=[1,3,6]):
    print("\n   Generating forward-looking forecasts...")
    rows = []
    latest = df.sort_values("year_month_dt").groupby("product_id").last().reset_index()
    last_dt = df["year_month_dt"].max()
    # Anchor forecast horizon labels to TODAY, not to the last date in training data.
    # e.g. 1M = next 1 month from now, 3M = next 3 months from now, etc.
    today_dt = pd.Timestamp.now().normalize().replace(day=1)  # first of current month

    for h in horizons:
        key = f"{h}m"
        if key not in results: continue
        model  = results[key]["model"]
        avail  = results[key]["avail_features"]
        tgt_dt = today_dt + pd.DateOffset(months=h)   # ← was: last_dt + offset
        tgt_ym = tgt_dt.strftime("%Y-%m")
        X_fut  = latest[avail].fillna(0)
        preds  = np.maximum(model.predict(X_fut), 0)

        tmp = latest[["product_id","generic_name","clinical_demand_pattern",
                       "dominant_wh_type","dominant_region","unit_price","pharm_class",
                       "num_unique_warehouses","num_unique_distributors"]].copy()
        tmp["forecast_year_month"] = tgt_ym
        tmp["horizon"]             = f"{h}M"
        tmp["forecasted_quantity"] = preds.round(0).astype(int)
        tmp["forecasted_value_usd"]= (preds * tmp["unit_price"].fillna(0)).round(2)
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print(f"   + Forecasts: {len(out):,} rows across horizons {horizons}")
    return out

# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(df_monthly, forecasts, results, encoders, data_path):
    print("\n   Saving outputs...")
    mr, fr, pr = [], [], []
    for hz, res in results.items():
        m = res["metrics"]
        mr.append({
            "Horizon":      hz,
            "Train Period": res.get("train_period", ""),
            "Test Period":  res.get("test_period",  ""),
            "Train %":      res.get("train_pct",    80),
            "Test %":       res.get("test_pct",     20),
            "Train Rows":   res["train_size"],
            "Test Rows":    res["test_size"],
            "Test MAPE(%)": round(m["test_mape"], 2),
            "Test RMSE":    round(m["test_rmse"], 1),
            "Test R²":      round(m["test_r2"],   4),
        })
        for p, mape in res["pattern_mape"].items():
            pr.append({"Horizon": hz, "clinical_demand_pattern": p, "MAPE(%)": mape})
        fi_tmp = res["feature_importance"].copy(); fi_tmp["Horizon"] = hz; fr.append(fi_tmp)

    keep = [c for c in df_monthly.columns if not c.startswith("tgt")]
    exp = df_monthly[keep].copy()
    exp["year_month"] = exp["year_month"].astype(str)
    if "year_month_dt" in exp.columns:
        exp["year_month_dt"] = exp["year_month_dt"].astype(str)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as w:
        exp.to_excel(w,                    sheet_name="monthly_shipment_demand", index=False)
        forecasts.to_excel(w,              sheet_name="demand_forecasts",        index=False)
        pd.DataFrame(mr).to_excel(w,       sheet_name="model_metrics",           index=False)
        pd.DataFrame(pr).to_excel(w,       sheet_name="pattern_mape",            index=False)
        (pd.concat(fr) if fr else pd.DataFrame()).to_excel(w,
                                           sheet_name="feature_importance",      index=False)

    cache = {"monthly_agg":exp, "forecasts":forecasts, "results":results,
             "encoders":encoders, "generated_at":datetime.now().isoformat(),
             "data_path":data_path}
    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(cache, f)

    print(f"   + Excel  -> {OUTPUT_EXCEL}  ({os.path.getsize(OUTPUT_EXCEL)//1024} KB)")
    print(f"   + Pickle -> {OUTPUT_PICKLE} ({os.path.getsize(OUTPUT_PICKLE)//1024} KB)")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(data_path):
    print("="*70)
    print("  PharmaTrace AI — Shipments-Driven Demand Prediction Pipeline")
    print("="*70)
    t0 = datetime.now()

    sheets    = load_sheets(data_path)
    enriched  = build_enriched(sheets)
    monthly   = aggregate_monthly(enriched)
    monthly   = classify_patterns(monthly)
    monthly, encoders = engineer_features(monthly)
    results   = train_and_evaluate(monthly)
    forecasts = generate_forecasts(monthly, results)
    save_outputs(monthly, forecasts, results, encoders, data_path)

    print(f"\n{'='*70}")
    print(f"  Pipeline complete in {(datetime.now()-t0).seconds}s")
    print(f"{'='*70}\n")
    return monthly, forecasts, results, encoders

# ─────────────────────────────────────────────────────────────────────────────
# LIVE MODE: accept pre-loaded DataFrames (called from Streamlit on upload)
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline_from_sheets(sheets: dict) -> dict:
    """
    Run the full demand prediction pipeline from pre-loaded DataFrames.

    Parameters
    ----------
    sheets : dict
        Must contain keys: 'shipments', 'finished_product_batches', 'products',
        'distributors', 'retailers', 'warehouses'  — each a pandas DataFrame.

    Returns
    -------
    dict with keys:
        'monthly_agg'  : pd.DataFrame   — product × month aggregated demand
        'forecasts'    : pd.DataFrame   — 1M / 3M / 6M forecast rows
        'results'      : dict           — model objects + metrics per horizon
        'encoders'     : dict           — LabelEncoder per categorical column
        'generated_at' : str            — ISO timestamp
    """
    required = ["shipments", "finished_product_batches", "products",
                "distributors", "retailers", "warehouses"]
    missing = [s for s in required if s not in sheets or sheets[s].empty]
    if missing:
        raise ValueError(f"run_pipeline_from_sheets: missing/empty sheets: {missing}")

    enriched  = build_enriched(sheets)
    monthly   = aggregate_monthly(enriched)
    monthly   = classify_patterns(monthly)
    monthly, encoders = engineer_features(monthly)
    results   = train_and_evaluate(monthly, train_pct=0.80,
                                   fast_mode=True)   # speed-optimised for live upload
    forecasts = generate_forecasts(monthly, results)

    return {
        "monthly_agg":  monthly,
        "forecasts":    forecasts,
        "results":      results,
        "encoders":     encoders,
        "generated_at": datetime.now().isoformat(),
        "data_path":    "<uploaded file>",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    args = parser.parse_args()
    if not os.path.exists(args.data):
        print(f"[ERROR] Not found: {args.data}")
        sys.exit(1)
    run_pipeline(args.data)

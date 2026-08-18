#!/usr/bin/env python3
"""
Figure 2 analysis — compute all panels from raw NHANES 2017-2018 data.

This script performs the analysis only (no figure rendering).
All outputs are JSON/CSV data files consumed by render.py.

Outputs:
    data/nhanes_real.csv                           cleaned NHANES cohort
    data/nhanes_synthetic.csv                      SPHERE-rotated version
    code/figure2/nhanes_anonymeter.json             Anonymeter privacy scores
    code/figure2/panel_e_ols.json                  OLS regression coefficients

Panel c/d use pre-computed AI-generated summaries in
code/figure2/panel_c_summaries.json (no API call required).

Usage:
    python3 code/figure2/analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.stats import pearsonr

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # <repo>/code/figure2
CODE = HERE.parent                              # <repo>/code
ROOT = CODE.parent                              # <repo>
RES  = HERE
DATA = ROOT / "data"
NHANES_CACHE = DATA / "nhanes_cache"


# ── NHANES file manifest ─────────────────────────────────────────────────────
_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
_NHANES_FILES = {
    "demo":   ("DEMO_J.XPT",   ["SEQN","RIDAGEYR","RIAGENDR","RIDRETH3","DMDEDUC2","INDFMPIR"]),
    "bmx":    ("BMX_J.XPT",    ["SEQN","BMXBMI","BMXWAIST"]),
    "bpx":    ("BPX_J.XPT",    ["SEQN","BPXSY1","BPXSY2","BPXSY3"]),
    "tchol":  ("TCHOL_J.XPT",  ["SEQN","LBXTC"]),
    "hdl":    ("HDL_J.XPT",    ["SEQN","LBDHDD"]),
    "ghb":    ("GHB_J.XPT",    ["SEQN","LBXGH"]),
    "biopro": ("BIOPRO_J.XPT", ["SEQN","LBXSCR","LBXSGL"]),
    "hscrp":  ("HSCRP_J.XPT",  ["SEQN","LBXHSCRP"]),
    "smq":    ("SMQ_J.XPT",    ["SEQN","SMQ020","SMQ040"]),
    "alq":    ("ALQ_J.XPT",    ["SEQN","ALQ130"]),
    "paq":    ("PAQ_J.XPT",    ["SEQN","PAQ605","PAQ650"]),
    "diet":   ("DR1TOT_J.XPT", ["SEQN","DR1TSODI"]),
}

FEATURE_COLS = [
    "age", "male", "race_black", "race_hispanic", "race_asian",
    "education", "poverty_ratio", "bmi", "waist_circumference",
    "total_cholesterol", "hdl_cholesterol", "hba1c",
    "creatinine", "glucose", "crp",
    "current_smoker", "drinks_per_day", "physically_active", "sodium_intake",
]

CONTINUOUS = [
    "age", "education", "poverty_ratio",
    "bmi", "waist_circumference",
    "total_cholesterol", "hdl_cholesterol", "hba1c",
    "creatinine", "glucose", "crp",
    "drinks_per_day", "sodium_intake",
]
BINARY = ["male", "race_black", "race_hispanic", "race_asian",
          "current_smoker", "physically_active"]

SPHERE_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Download & clean NHANES 2017-2018
# ══════════════════════════════════════════════════════════════════════════════

def _download_xpt(filename: str) -> pd.DataFrame:
    NHANES_CACHE.mkdir(parents=True, exist_ok=True)
    local = NHANES_CACHE / filename
    if local.exists():
        return pd.read_sas(local, format="xport")
    import requests
    url = f"{_BASE_URL}/{filename}"
    print(f"    [download] {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    local.write_bytes(resp.content)
    from io import BytesIO
    return pd.read_sas(BytesIO(resp.content), format="xport")


def load_nhanes() -> pd.DataFrame:
    """Download, merge, clean NHANES 2017-2018. Cached as data/nhanes_real.csv."""
    csv_path = DATA / "nhanes_real.csv"
    if csv_path.exists():
        print(f"  [cache] {csv_path}")
        return pd.read_csv(csv_path)

    print("  Downloading NHANES 2017-2018 from CDC...")
    dfs = {}
    for key, (fname, cols) in _NHANES_FILES.items():
        raw = _download_xpt(fname)
        dfs[key] = raw[[c for c in cols if c in raw.columns]].copy()

    df = dfs["demo"]
    for key in list(_NHANES_FILES.keys())[1:]:
        df = df.merge(dfs[key], on="SEQN", how="left")

    df = df[df["RIDAGEYR"] >= 20].copy()

    sbp_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3"] if c in df.columns]
    df["sbp_mean"] = df[sbp_cols].mean(axis=1, skipna=True)
    df = df.dropna(subset=["sbp_mean"])

    _pm1 = lambda s: 2.0 * s.astype(float) - 1.0
    df["age"]               = df["RIDAGEYR"].astype(float)
    df["male"]              = _pm1(df["RIAGENDR"] == 1)
    df["race_black"]        = _pm1(df["RIDRETH3"] == 4)
    df["race_hispanic"]     = _pm1(df["RIDRETH3"].isin([1, 2]))
    df["race_asian"]        = _pm1(df["RIDRETH3"] == 6)
    df["education"]         = df["DMDEDUC2"].replace({7: np.nan, 9: np.nan}).astype(float)
    df["poverty_ratio"]     = df["INDFMPIR"].astype(float)
    df["bmi"]               = df["BMXBMI"].astype(float)
    df["waist_circumference"] = df["BMXWAIST"].astype(float)
    df["total_cholesterol"]  = df["LBXTC"].astype(float)
    df["hdl_cholesterol"]   = df["LBDHDD"].astype(float)
    df["hba1c"]             = df["LBXGH"].astype(float)
    df["creatinine"]        = df["LBXSCR"].astype(float)
    df["glucose"]           = df["LBXSGL"].astype(float)
    df["crp"]               = df["LBXHSCRP"].astype(float)
    df["current_smoker"]    = _pm1((df["SMQ020"] == 1) & df["SMQ040"].isin([1, 2]))
    df["drinks_per_day"]    = df["ALQ130"].replace({777: np.nan, 999: np.nan}).fillna(0).astype(float)
    df["physically_active"] = _pm1((df["PAQ605"] == 1) | (df["PAQ650"] == 1))
    df["sodium_intake"]     = df["DR1TSODI"].astype(float)

    out = df[FEATURE_COLS + ["sbp_mean"]].copy()
    out = out.dropna(thresh=int(0.7 * len(FEATURE_COLS)), subset=FEATURE_COLS)
    for col in FEATURE_COLS:
        out[col] = out[col].fillna(out[col].median())

    DATA.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}  (n={len(out)}, p={len(FEATURE_COLS)})")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Apply SPHERE rotation
# ══════════════════════════════════════════════════════════════════════════════

def apply_sphere(df_real: pd.DataFrame) -> pd.DataFrame:
    """SPHERE-rotate [features | sbp_mean] jointly -> nhanes_synthetic.csv."""
    csv_path = DATA / "nhanes_synthetic.csv"
    if csv_path.exists():
        print(f"  [cache] {csv_path}")
        return pd.read_csv(csv_path)

    # SPHERE is available as a macOS desktop app (https://github.com/statzihuai/SPHERE)
    # and a command-line tool (https://github.com/statzihuai/sphere-cli).
    from sphere import sphere

    Z = df_real[FEATURE_COLS].values.astype(float)
    y = df_real["sbp_mean"].values.astype(float)

    np.random.seed(SPHERE_SEED)
    Zy_star = sphere(np.column_stack([Z, y]))
    Zstar = Zy_star[:, :-1]
    ystar = Zy_star[:, -1]

    df_star = pd.DataFrame(Zstar, columns=FEATURE_COLS)
    df_star["sbp_mean"] = ystar
    df_star.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")
    return df_star


# ══════════════════════════════════════════════════════════════════════════════
# Panel b — Anonymeter privacy (analysis only; rendering in render.py)
# ══════════════════════════════════════════════════════════════════════════════

def _run_anonymeter(df_ori, df_syn, label, seed=42):
    from anonymeter.evaluators import (
        SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator
    )
    import anonymeter.evaluators.singling_out_evaluator as so_mod
    import anonymeter.evaluators.linkability_evaluator as lk_mod

    _ORIG_RNG = np.random.default_rng

    def reseed(s=seed):
        """Reset anonymeter's three RNG sources for reproducibility."""
        np.random.seed(s)                                    # pandas .sample
        so_mod.rng = np.random.default_rng(s)                # singling-out module
        lk_mod.np.random.default_rng = lambda *a, **k: _ORIG_RNG(s)  # linkability module

    n_atk = min(2000, len(df_ori))
    n_attacks = 500
    out = {}
    rng_ = np.random.RandomState(seed)
    all_cols = list(df_ori.columns)

    try:
        reseed()
        ev = SinglingOutEvaluator(df_ori.head(n_atk), df_syn.head(n_atk),
                                  n_attacks=n_attacks, n_cols=3)
        ev.evaluate()
        out["singling_out"] = ev.risk().value
    except Exception as e:
        print(f"    singling-out failed: {e}"); out["singling_out"] = np.nan

    try:
        reseed()
        link_ = rng_.choice(all_cols, size=min(20, len(all_cols)), replace=False).tolist()
        half_ = len(link_) // 2
        ev = LinkabilityEvaluator(df_ori.head(n_atk), df_syn.head(n_atk),
                                  aux_cols=(link_[:half_], link_[half_:]),
                                  n_attacks=n_attacks)
        ev.evaluate()
        out["linkability"] = ev.risk().value
    except Exception as e:
        print(f"    linkability failed: {e}"); out["linkability"] = np.nan

    try:
        risks = []
        aux_size = min(10, len(all_cols) - 1)
        for si, secret in enumerate(all_cols):
            aux = [c for c in all_cols if c != secret][:aux_size]
            try:
                reseed()
                ev = InferenceEvaluator(df_ori.head(n_atk), df_syn.head(n_atk),
                                        aux_cols=aux, secret=secret, n_attacks=n_attacks)
                ev.evaluate()
                risks.append(ev.risk().value)
            except Exception:
                pass
        out["inference"] = float(np.mean(risks)) if risks else np.nan
    except Exception as e:
        print(f"    inference failed: {e}"); out["inference"] = np.nan

    print(f"  {label}: SO={out['singling_out']:.4f}, "
          f"LK={out['linkability']:.4f}, INF={out['inference']:.4f}")
    return out


def panel_b_privacy(df_real, df_synth):
    """Anonymeter privacy scores. Saves JSON only; rendering is in render.py."""
    rng = np.random.RandomState(42)
    df_colshuf = df_real.copy()
    for col in df_colshuf.columns:
        df_colshuf[col] = rng.permutation(df_colshuf[col].values)

    r_real = _run_anonymeter(df_real, df_real, "Real")
    r_sph = _run_anonymeter(df_real, df_synth, "SPHERE")
    r_cs = _run_anonymeter(df_real, df_colshuf, "ColShuffle")

    raw_vals = {"Real": r_real, "SPHERE": r_sph, "ColShuf.": r_cs}

    anon_path = RES / "nhanes_anonymeter.json"
    anon_path.write_text(json.dumps(raw_vals, indent=2))
    print(f"  Saved -> {anon_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Panel e — OLS regression coefficients (computed directly from raw data)
# ══════════════════════════════════════════════════════════════════════════════

def panel_e_ols(df_real, df_synth):
    """Compute OLS on real and SPHERE data. Saves JSON with coefficients."""
    y_r = df_real["sbp_mean"].values
    y_s = df_synth["sbp_mean"].values
    n = len(y_r)

    # Standardise continuous predictors (same transform as the AI code)
    X_r = df_real[FEATURE_COLS].values.astype(float).copy()
    X_s = df_synth[FEATURE_COLS].values.astype(float).copy()
    for j, col in enumerate(FEATURE_COLS):
        if col in CONTINUOUS:
            mu, sd = X_r[:, j].mean(), X_r[:, j].std()
            if sd > 0:
                X_r[:, j] = (X_r[:, j] - mu) / sd
                X_s[:, j] = (X_s[:, j] - mu) / sd

    # Add intercept
    Xr = np.column_stack([np.ones(n), X_r])
    Xs = np.column_stack([np.ones(n), X_s])

    # OLS
    beta_r = lstsq(Xr, y_r, rcond=None)[0]
    beta_s = lstsq(Xs, y_s, rcond=None)[0]

    # SE and p-values
    def _ols_stats(X, y, beta):
        res = y - X @ beta
        dof = len(y) - X.shape[1]
        s2 = (res @ res) / dof
        from numpy.linalg import pinv
        XtXi = pinv(X.T @ X)
        se = np.sqrt(np.maximum(s2 * np.diag(XtXi), 0))
        t = beta / np.where(se > 0, se, 1)
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(np.abs(t), dof)
        r2 = 1 - np.sum(res**2) / np.sum((y - y.mean())**2)
        return se, p, r2

    se_r, p_r, r2_r = _ols_stats(Xr, y_r, beta_r)
    se_s, p_s, r2_s = _ols_stats(Xs, y_s, beta_s)

    # Pearson r of coefficients (excluding intercept)
    r_coef = pearsonr(beta_r[1:], beta_s[1:])[0]

    # Build results dict
    coefs = {}
    for j, col in enumerate(FEATURE_COLS):
        coefs[col] = {
            "real_coef": float(beta_r[j + 1]),
            "real_se": float(se_r[j + 1]),
            "real_p": float(p_r[j + 1]),
            "sphere_coef": float(beta_s[j + 1]),
            "sphere_se": float(se_s[j + 1]),
            "sphere_p": float(p_s[j + 1]),
        }

    result = {
        "n": n,
        "n_vars": len(FEATURE_COLS),
        "r2_real": float(r2_r),
        "r2_sphere": float(r2_s),
        "coef_pearson_r": float(r_coef),
        "coefficients": coefs,
    }

    out_path = RES / "panel_e_ols.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  R2 real={r2_r:.4f}  SPHERE={r2_s:.4f}")
    print(f"  Coefficient Pearson r = {r_coef:.6f}")
    print(f"  Saved -> {out_path}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    RES.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Figure 2 — NHANES analysis")
    print("=" * 60)

    # Step 1: NHANES data
    print("\n▸ Loading NHANES 2017-2018...")
    df_real = load_nhanes()
    print(f"  n={len(df_real)}, p={len(FEATURE_COLS)}")

    # Step 2: SPHERE rotation
    print("\n▸ Applying SPHERE rotation...")
    df_synth = apply_sphere(df_real)

    # Panel b: Privacy
    print("\n▸ Panel b (privacy)...")
    panel_b_privacy(df_real, df_synth)

    # Panel e: OLS coefficients
    print("\n▸ Panel e (OLS coefficients)...")
    panel_e_ols(df_real, df_synth)

    print("\n" + "=" * 60)
    print("Analysis complete. Run code/figure2/render.py to build the figure.")
    print("=" * 60)


if __name__ == "__main__":
    main()

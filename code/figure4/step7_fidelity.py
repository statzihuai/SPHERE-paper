#!/usr/bin/env python3
"""
Step 7 — distributional fidelity of the released rotation (panel b).

Per modality, over up to 500 shared columns, four scores on a 0-100 scale:

    ks     1 - the two-sample Kolmogorov-Smirnov statistic, per column
    mean   1 - |mean(syn) - mean(real)| / |mean(real)|
    var    1 - |var(syn)  - var(real)|  / |var(real)|
    corr   1 - |r(syn) - r(real)| / |r(real)| over 500 random column pairs,
           skipping pairs whose real correlation is under 0.05 (a near-zero
           denominator would otherwise dominate the average)

and their mean as the composite. The column subsample and the pair draw are
seeded, so the table is reproducible; none of the four is an orthogonal
invariant, so none is expected to come out exact.

    python3 step7_fidelity.py \\
        --real-root  data/adrc/real \\
        --sphere-dir results/figure4/sphere_xk/x2 \\
        --out-csv    results/figure4/panel_b_fidelity.csv
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats

from adrc_common import CROSS_MODS, FIG_MOD_FILE, load_csv

MAX_COLS = 500      # columns kept per modality
N_PAIRS = 500       # column pairs behind the correlation score
CORR_FLOOR = 0.05   # real |r| below this is skipped (unstable ratio)
EPS = 1e-8


def modality_fidelity(real_path: Path, sphere_path: Path) -> dict:
    real_df, sphere_df = load_csv(real_path, max_cols=MAX_COLS), load_csv(sphere_path, max_cols=MAX_COLS)
    common = [c for c in sorted(set(real_df.columns) & set(sphere_df.columns)) if "_orth_" not in c]
    if len(common) > MAX_COLS:
        common = list(np.random.default_rng(1).choice(common, size=MAX_COLS, replace=False))
    R, S = real_df[common].astype(float), sphere_df[common].astype(float)

    ks_s, mn_s, vr_s = [], [], []
    for col in common:
        rv, sv = R[col].dropna().values, S[col].dropna().values
        if len(rv) < 5 or len(sv) < 5:
            continue
        ks_s.append(1.0 - spstats.ks_2samp(rv, sv)[0])
        mn_s.append(np.clip(1.0 - abs(sv.mean() - rv.mean()) / (abs(rv.mean()) + EPS), 0, 1))
        vr_s.append(np.clip(1.0 - abs(sv.var() - rv.var()) / (abs(rv.var()) + EPS), 0, 1))

    rng_p = np.random.default_rng(2)
    n_pairs = min(N_PAIRS, len(common) * (len(common) - 1) // 2)
    corr_diffs, cols_arr = [], np.array(common)
    if len(cols_arr) >= 2:
        for pi, pj in rng_p.choice(len(cols_arr), size=(n_pairs, 2), replace=True):
            if pi == pj:
                continue
            ci, cj = cols_arr[pi], cols_arr[pj]
            jr, js = R[[ci, cj]].dropna(), S[[ci, cj]].dropna()
            if len(jr) < 5 or len(js) < 5:
                continue
            cr = spstats.pearsonr(jr[ci].values, jr[cj].values)[0]
            cs = spstats.pearsonr(js[ci].values, js[cj].values)[0]
            if abs(cr) < CORR_FLOOR:
                continue
            val = np.clip(1.0 - abs(cs - cr) / (abs(cr) + EPS), 0, 1)
            if np.isfinite(val):
                corr_diffs.append(val)

    ks, mn, vr, co = (np.nanmean(ks_s) if ks_s else np.nan,
                      np.nanmean(mn_s) if mn_s else np.nan,
                      np.nanmean(vr_s) if vr_s else np.nan,
                      np.nanmean(corr_diffs) if corr_diffs else np.nan)
    return dict(ks=ks * 100, mean=mn * 100, var=vr * 100, corr=co * 100,
                composite=np.nanmean([ks, mn, vr, co]) * 100)


def main():
    ap = argparse.ArgumentParser(description="Distributional fidelity per modality (panel b)")
    ap.add_argument("--real-root", type=Path, required=True)
    ap.add_argument("--sphere-dir", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for mod in CROSS_MODS:
        rpath, spath = args.real_root / FIG_MOD_FILE[mod], args.sphere_dir / FIG_MOD_FILE[mod]
        if not rpath.exists() or not spath.exists():
            continue
        print(f"  {mod} …", end=" ", flush=True)
        try:
            rows.append(dict(modality=mod, **modality_fidelity(rpath, spath)))
            print(f"comp={rows[-1]['composite']:.1f}")
        except Exception as e:
            print(f"ERROR: {e}")
        gc.collect()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[write] {len(rows)} modalities -> {args.out_csv}")


if __name__ == "__main__":
    main()

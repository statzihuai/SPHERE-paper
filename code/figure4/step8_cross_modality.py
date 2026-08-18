#!/usr/bin/env python3
"""
Step 8 — cross-modality preservation (panels f and g).

For every pair of the eight measured modalities, 50 random features are drawn
from each side, the cross-correlation block between them is computed on the
participants they share, and the real and synthetic blocks are compared:

    panel f   |Pearson r| between the two correlation blocks, x100
    panel g   |Pearson r| between the two -log10 p blocks, x100

This is the test the per-site rotation has to pass and the single-modality
metrics cannot see: a pair of modalities is generally measured on participants
spread over several rotation sites, so nothing forces the between-block
structure to survive.

The feature draw is repeated over --seeds seeds and the scores averaged, since
a single 50-feature draw of a 6,900-column proteomics panel is noisy.

    python3 step8_cross_modality.py \\
        --real-root  data/adrc/real \\
        --sphere-dir results/figure4/sphere_xk/x2 \\
        --seeds 100 \\
        --out-csv    results/figure4/panels_fg_cross_modality.csv
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats
from scipy.stats import t as tdist

from adrc_common import CROSS_MODS, FIG_MOD_FILE, load_csv

MAX_FEAT_CM = 50    # features drawn per modality per seed
MIN_SHARED = 10     # participants a pair needs before it is scored


def _load_modality_cm(real_root: Path, sphere_dir: Path, seed: int):
    rcm, scm = {}, {}
    for mod in CROSS_MODS:
        rp, sp = real_root / FIG_MOD_FILE[mod], sphere_dir / FIG_MOD_FILE[mod]
        try:
            r_df = load_csv(rp, max_cols=MAX_FEAT_CM, seed=seed)
            s_df = load_csv(sp, max_cols=MAX_FEAT_CM, seed=seed)
            r_df = r_df[[c for c in r_df.columns if "_orth_" not in c]].select_dtypes(include="number")
            s_df = s_df[[c for c in s_df.columns if "_orth_" not in c]].select_dtypes(include="number")
            shared = sorted(set(r_df.columns) & set(s_df.columns))
            if len(shared) > MAX_FEAT_CM:
                shared = list(np.random.default_rng(seed).choice(shared, size=MAX_FEAT_CM, replace=False))
            rcm[mod], scm[mod] = r_df[shared], s_df[shared]
        except Exception as e:
            print(f"  ERROR loading {mod} (seed={seed}): {e}")
        gc.collect()
    return rcm, scm


def _cross_scores(R_A, R_B, S_A, S_B):
    def _corr_lp(A, B):
        pids = A.index.intersection(B.index)
        if len(pids) < MIN_SHARED:
            return None, None
        a, b = A.loc[pids].values.astype(float), B.loc[pids].values.astype(float)
        mask = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        n = mask.sum()
        if n < MIN_SHARED:
            return None, None
        a, b = a[mask], b[mask]
        sa, sb = a.std(0), b.std(0)
        sa[sa < 1e-8] = 1.0; sb[sb < 1e-8] = 1.0
        a, b = (a - a.mean(0)) / sa, (b - b.mean(0)) / sb
        C = (a.T @ b) / n
        t = C * np.sqrt(n - 2) / np.sqrt(np.maximum(1 - C ** 2, 1e-10))
        lp = -np.log10(2 * tdist.sf(np.abs(t), df=n - 2) + 1e-300)
        return C, lp

    Cr, lpr = _corr_lp(R_A, R_B)
    Cs, lps = _corr_lp(S_A, S_B)
    if Cr is None or Cs is None or Cr.size < 2:
        return np.nan, np.nan
    return (abs(spstats.pearsonr(Cr.flatten(), Cs.flatten())[0]) * 100,
            abs(spstats.pearsonr(lpr.flatten(), lps.flatten())[0]) * 100)


def cross_modality(real_root: Path, sphere_dir: Path, seeds: int):
    n_cm = len(CROSS_MODS)
    acc_s, acc_p, acc_n = (np.zeros((n_cm, n_cm)), np.zeros((n_cm, n_cm)), np.zeros((n_cm, n_cm)))
    for sd in range(seeds):
        if (sd + 1) % 20 == 0 or sd == 0:
            print(f"  cross-modality seed {sd+1}/{seeds}")
        rcm, scm = _load_modality_cm(real_root, sphere_dir, sd)
        for i, mi in enumerate(CROSS_MODS):
            for j, mj in enumerate(CROSS_MODS):
                if j < i or mi not in rcm or mj not in rcm:
                    continue
                sc, sp = _cross_scores(rcm[mi], rcm[mj], scm[mi], scm[mj])
                if np.isfinite(sc) and np.isfinite(sp):
                    acc_s[i, j] += sc; acc_p[i, j] += sp; acc_n[i, j] += 1
        gc.collect()

    rows = []
    for i in range(n_cm):
        for j in range(i, n_cm):
            rows.append(dict(modality_i=CROSS_MODS[i], modality_j=CROSS_MODS[j],
                             corr_score=acc_s[i, j] / acc_n[i, j] if acc_n[i, j] else np.nan,
                             pval_score=acc_p[i, j] / acc_n[i, j] if acc_n[i, j] else np.nan,
                             n_seeds=int(acc_n[i, j])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Cross-modality correlation/p preservation (panels f, g)")
    ap.add_argument("--real-root", type=Path, required=True)
    ap.add_argument("--sphere-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=100, help="Feature draws to average over")
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    df = cross_modality(args.real_root, args.sphere_dir, args.seeds)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    off = df[df.modality_i != df.modality_j]
    print(f"[write] {len(df)} pairs -> {args.out_csv}")
    print(f"  off-diagonal mean: corr {off.corr_score.mean():.2f}  p {off.pval_score.mean():.2f}")


if __name__ == "__main__":
    main()

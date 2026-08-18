#!/usr/bin/env python3
"""
Figure 1 panel c — distributional fidelity.

Measures how far each synthesiser's output deviates from the real data in
moments, correlation structure, and marginals. Also records (dataset, n, p)
used by the dataset-overview scatter in panel b.

Metrics (per dataset × method):
  pct_delta_mean   % deviation in column means
  pct_delta_var    % deviation in column variances
  pct_delta_cor    % deviation in correlation (Frobenius)
  ks_statistic     mean KS distance across columns

Usage:
    python3 code/figure1/panel_c_fidelity.py [--methods ...] [--skip-deep] [--workers N]
"""
import numpy as np
import pandas as pd

import _common as C


def compute(datasets, methods):
    from scipy.stats import ks_2samp

    # Global scale references, pooled over all 33 datasets (see docstring).
    all_means = np.concatenate([d["Z"].mean(0) for d in datasets])
    all_vars = np.concatenate([d["Z"].var(0) for d in datasets])
    mean_ref = float(np.nanmean(np.abs(all_means))) + 1e-8
    var_ref = float(np.nanmean(all_vars)) + 1e-8

    rows = []
    for d in datasets:
        Z = d["Z"]
        n, pfull = Z.shape
        R = np.corrcoef(Z.T)
        ref_cor = float(np.linalg.norm(R, "fro")) / pfull + 1e-8
        nan_row = [d["name"], n, d["p"], None, np.nan, np.nan, np.nan, np.nan]

        for m in methods:
            if C._skip(m, n, d["p"]):
                rows.append(nan_row[:3] + [m] + nan_row[4:]); continue
            path = C.SYNTH_FULL / f"{d['name']}__{m}.npy"
            if not path.exists():
                rows.append(nan_row[:3] + [m] + nan_row[4:]); continue
            Zs = np.load(path)
            if np.isnan(Zs).all():
                rows.append(nan_row[:3] + [m] + nan_row[4:]); continue

            md = np.max(np.abs(Zs.mean(0) - Z.mean(0))) / mean_ref * 100
            vd = np.max(np.abs(Zs.var(0) - Z.var(0))) / var_ref * 100
            cd = np.linalg.norm(np.corrcoef(Zs.T) - R, "fro") / pfull / ref_cor * 100
            ks = float(np.mean([ks_2samp(Z[:, j], Zs[:, j]).statistic
                                for j in range(pfull)]))
            rows.append([d["name"], n, d["p"], m, md, vd, cd, ks])

    return pd.DataFrame(rows, columns=["dataset", "n", "p", "method",
                                       "pct_delta_mean", "pct_delta_var",
                                       "pct_delta_cor", "ks_statistic"])


def main():
    args = C.build_parser("Figure 1 panel d — distributional fidelity").parse_args()
    datasets, methods = C.setup(args, stages={"full"},
                                description="Figure 1 panel d — distributional fidelity")
    print("\n> Panel d (fidelity)")
    df = compute(datasets, methods)
    C.save_csv(C.RES / "panel_c_fidelity.csv", df, methods)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Figure 1 panel d — utility for statistical inference.

Tests whether OLS analysis on synthetic data preserves the real data's
conclusions: false-positive rate, power, and CI coverage.

For each dataset, 5 simulated response vectors are generated on the fly
(y_null = no effect, y_signal = first n_sig columns carry β=0.27).
OLS is fitted on the synthetic [y|X], and |t|>1.96 is the rejection rule.

Metrics:
  type1_error   fraction of null columns rejected (target 0.05)
  power         fraction of signal columns detected (higher = better)
  ci_coverage   1 − type1_error (target 0.95)

Usage:
    python3 code/figure1/panel_d_inference.py [--methods ...] [--skip-deep] [--workers N]
"""
import numpy as np
import pandas as pd

import _common as C

T_CRIT = 1.96      # two-sided 5% normal critical value
N_SIMS = 5         # simulations shipped per dataset


def compute(datasets, methods):
    from numpy.linalg import lstsq, pinv

    rows = []
    for di, d in enumerate(datasets):
        simY = C.generate_simY(di, d["Z"])
        n_sig = int(simY["n_sig"])

        for m in methods:
            if C._skip(m, d["n"], d["p"]):
                for s in range(N_SIMS):
                    rows.append([d["name"], d["n"], d["p"], m, s,
                                 np.nan, np.nan, np.nan])
                continue

            for s in range(N_SIMS):
                t1 = pw = ci = np.nan
                for k in (0, 1):          # 0 = y_null, 1 = y_signal
                    path = C.SYNTH_D / f"{d['name']}__{m}__sim{s}_y{k}.npy"
                    if not path.exists():
                        continue
                    Zs = np.load(path)
                    if np.isnan(Zs).all():
                        continue

                    ys = Zs[:, 0]
                    Xs = np.column_stack([np.ones(len(Zs)), Zs[:, 1:]])
                    dof = len(Zs) - Xs.shape[1]
                    if dof < 1:
                        continue

                    b = lstsq(Xs, ys, rcond=None)[0]
                    res = ys - Xs @ b
                    s2 = (res @ res) / dof
                    XtXi = pinv(Xs.T @ Xs)          # pinv: tolerate rank deficiency
                    se = np.sqrt(np.maximum(s2 * np.diag(XtXi), 0))
                    t = np.abs(b[1:] / np.where(se[1:] > 0, se[1:], 1))
                    p_cols = len(t)

                    if k == 0:
                        # Null response: y_null ~ N(0,1) independent of X, so
                        # ALL p columns are true nulls — use the full vector.
                        t1 = float(np.mean(t > T_CRIT))
                        ci = float(np.mean(t < T_CRIT))
                    else:
                        # Signal response: the first n_sig columns carry the effect.
                        pw = float(np.mean(t[:n_sig] > T_CRIT))

                rows.append([d["name"], d["n"], d["p"], m, s, t1, pw, ci])

    return pd.DataFrame(rows, columns=["dataset", "n", "p", "method", "sim",
                                       "type1_error", "power", "ci_coverage"])


def main():
    args = C.build_parser("Figure 1 panel e — OLS inference utility").parse_args()
    datasets, methods = C.setup(args, stages={"D"},
                                description="Figure 1 panel e — utility for statistical inference")
    print("\n> Panel e (inference)")
    df = compute(datasets, methods)
    C.save_csv(C.RES / "panel_d_inference.csv", df, methods)


if __name__ == "__main__":
    main()

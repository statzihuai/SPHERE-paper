#!/usr/bin/env python3
"""
Proteome-wide association study (PWAS) — OLS of each protein on age,
adjusted for sex, on both real and SPHERE-rotated data.

Model (per protein j):
    protein_j ~ age + sex

    python3 step9_pwas.py \\
        --proteomics-csv $UKB_DIR/olink.csv \\
        --sphere-npy output/proteomics/prot_sphere.npy \\
        --out-dir output/pwas
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

META_COLS = {"IID", "age", "sex"}


def pwas(Z, age, sex):
    """OLS protein ~ age + sex for each column of Z.

    Returns DataFrame with beta_age, se, t, p, neglog10p per protein.
    """
    n, p = Z.shape
    X = np.column_stack([np.ones(n), age, sex])
    Q, R = np.linalg.qr(X)
    dof = n - X.shape[1]

    betas, ses, ts, ps = [], [], [], []
    for j in range(p):
        y = Z[:, j]
        b = np.linalg.solve(R, Q.T @ y)
        resid = y - X @ b
        s2 = (resid @ resid) / dof
        # SE of the age coefficient (index 1)
        R_inv = np.linalg.inv(R)
        var_b = s2 * (R_inv @ R_inv.T)
        se = np.sqrt(var_b[1, 1])
        t = b[1] / se if se > 0 else 0.0
        p_val = 2.0 * stats.t.sf(abs(t), dof)
        betas.append(b[1])
        ses.append(se)
        ts.append(t)
        ps.append(p_val)

    return pd.DataFrame({
        "beta_age": betas,
        "se": ses,
        "t": ts,
        "p": ps,
        "neglog10p": [-np.log10(max(pv, 1e-300)) for pv in ps],
    })


def main():
    ap = argparse.ArgumentParser(description="PWAS: protein ~ age + sex")
    ap.add_argument("--proteomics-csv", required=True)
    ap.add_argument("--sphere-npy", required=True,
                    help="SPHERE-rotated protein matrix from prot_sphere.py")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.proteomics_csv)
    prot_cols = [c for c in df.columns if c not in META_COLS]
    Z_real = df[prot_cols].to_numpy(np.float64)
    age = df["age"].to_numpy(np.float64)
    sex = df["sex"].to_numpy(np.float64)

    Z_sphere = np.load(args.sphere_npy)
    assert Z_real.shape == Z_sphere.shape, "shape mismatch"

    n, p = Z_real.shape
    print(f"[pwas] n={n:,}  proteins={p:,}")

    results = {}
    for label, Z in [("real", Z_real), ("sphere", Z_sphere)]:
        result = pwas(Z, age, sex)
        result.insert(0, "protein", prot_cols)
        results[label] = result
        out = os.path.join(args.out_dir, f"pwas_{label}.csv")
        result.to_csv(out, index=False)
        n_sig = (result["p"] < 0.05 / p).sum()
        print(f"[pwas] {label}: {n_sig:,} Bonferroni-significant (p < {0.05/p:.2e})")
        print(f"[pwas] -> {out}")

    # Combined npz for figure3_plot.py
    npz_out = os.path.join(args.out_dir, "pwas.npz")
    np.savez(npz_out,
             beta_real=results["real"]["beta_age"].to_numpy(),
             nlp_real=results["real"]["neglog10p"].to_numpy(),
             beta_synth=results["sphere"]["beta_age"].to_numpy(),
             nlp_synth=results["sphere"]["neglog10p"].to_numpy())
    print(f"[pwas] -> {npz_out}")


if __name__ == "__main__":
    main()

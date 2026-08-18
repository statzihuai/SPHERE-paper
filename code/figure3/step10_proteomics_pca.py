#!/usr/bin/env python3
"""
PCA embedding of real and SPHERE-rotated Olink proteomics for visual
comparison (panel e). Concatenates a subsample of real and SPHERE rows,
standardises, and projects to 2 PCs.

    python3 step10_proteomics_pca.py \\
        --proteomics-csv $UKB_DIR/olink.csv \\
        --sphere-npy output/proteomics/prot_sphere.npy \\
        --n-sub 5000 --seed 42 \\
        --out-dir output/proteomics
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

META_COLS = {"IID", "age", "sex"}


def main():
    ap = argparse.ArgumentParser(
        description="Proteomics PCA embedding for panel e")
    ap.add_argument("--proteomics-csv", required=True)
    ap.add_argument("--sphere-npy", required=True,
                    help="SPHERE-rotated protein matrix from prot_sphere.py")
    ap.add_argument("--n-sub", type=int, default=5000,
                    help="subsample size per group (default 5000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.proteomics_csv)
    prot_cols = [c for c in df.columns if c not in META_COLS]
    Z_real = df[prot_cols].to_numpy(np.float64)
    Z_sphere = np.load(args.sphere_npy)
    assert Z_real.shape == Z_sphere.shape, "shape mismatch"

    n = Z_real.shape[0]
    n_sub = min(args.n_sub, n)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)[:n_sub]

    # Concatenate subsample: [real; SPHERE]
    Z_both = np.vstack([Z_real[idx], Z_sphere[idx]])
    sc = StandardScaler().fit(Z_both)
    pca = PCA(n_components=2, random_state=args.seed)
    emb = pca.fit_transform(sc.transform(Z_both))
    var_explained = pca.explained_variance_ratio_ * 100
    labels = np.array([0] * n_sub + [1] * n_sub)  # 0 = Real, 1 = SPHERE

    out = os.path.join(args.out_dir, "prot_pca_sub.npz")
    np.savez(out, real=emb[:n_sub], synth=emb[n_sub:], labels=labels,
             var_explained=var_explained)
    print(f"[prot-pca] n_sub={n_sub:,}  var PC1={var_explained[0]:.1f}%  "
          f"PC2={var_explained[1]:.1f}%")
    print(f"[prot-pca] -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Apply SPHERE rotation to the Olink proteomics matrix (continuous mode).

    python3 prot_sphere.py \\
        --proteomics-csv $UKB_DIR/olink.csv \\
        --seed 42 --out-dir output/proteomics
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd



def _sphere2(Z):
    """Two iterations of SPHERE (k = 2)."""
    return sphere(sphere(Z))


META_COLS = {"IID", "age", "sex"}


def main():
    ap = argparse.ArgumentParser(
        description="SPHERE rotation of Olink proteomics")
    ap.add_argument("--proteomics-csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.proteomics_csv)
    prot_cols = [c for c in df.columns if c not in META_COLS]
    Z = df[prot_cols].to_numpy(np.float64)
    n, p = Z.shape
    print(f"[sphere-prot] n={n:,}  p={p:,}")

    np.random.seed(args.seed)
    Z_sphere = _sphere2(Z)

    out = os.path.join(args.out_dir, "prot_sphere.npy")
    np.save(out, Z_sphere)
    print(f"[sphere-prot] -> {out}")


if __name__ == "__main__":
    main()

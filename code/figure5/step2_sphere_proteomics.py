#!/usr/bin/env python3
"""
SPHERE rotation of the UKB Olink proteomics matrix (continuous mode).

The full protein panel and Age are rotated jointly, and within each
partition (train rows separately, test rows separately), so that each
subset's Gram matrix and column means are preserved exactly:

    train mean/sd preserved   -> identical z-scoring
    train X'X, X'y preserved  -> identical LASSO coefficients
    orthogonal within test    -> identical test Pearson r

L1 quantities such as MAE are not orthogonally invariant and match only
approximately.

    python3 step2_sphere_proteomics.py \\
        --proteomics-csv $DATA_DIR/olink_with_split.csv \\
        --panels-json organ_panels.json \\
        --seed 0 --out-dir output/organaging
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd



def _sphere2(Z):
    """Two iterations of SPHERE (k = 2)."""
    return sphere(sphere(Z))


SEED = 0


def rotate_partition(Z, seed):
    """Continuous (orthogonal) SPHERE on one partition. Preserves Z'Z + means."""
    np.random.seed(seed)
    return np.asarray(_sphere2(np.asarray(Z, dtype=float)), dtype=float)


def main():
    ap = argparse.ArgumentParser(
        description="SPHERE within-partition rotation of Olink proteomics + Age")
    ap.add_argument("--proteomics-csv", required=True)
    ap.add_argument("--panels-json", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    panels = json.loads(open(args.panels_json).read())
    prots = panels.get("Conventional", list(panels.values())[0])
    df = pd.read_csv(args.proteomics_csv)
    df.columns = [str(c).lower() for c in df.columns]
    prots = [p for p in prots if p in df.columns]

    print(f"[data] n={len(df)}  proteins={len(prots)}")

    cols = prots + ["age"]
    synth = df.copy()
    # Age arrives as an integer column and the rotated values are continuous,
    # so cast the rotation targets to float before assigning into them.
    synth[cols] = synth[cols].astype(float)
    for part, seed_off in [("train", 0), ("test", 1)]:
        mask = (df["split"] == part).values
        Z = df.loc[mask, cols].to_numpy(float)
        Zs = rotate_partition(Z, seed=args.seed + seed_off)
        synth.loc[mask, cols] = Zs
        print(f"  {part}: n={int(mask.sum())}  rotated {len(cols)} columns (proteins + Age)")

    out = os.path.join(args.out_dir, "synth_proteomics.parquet")
    synth.to_parquet(out, index=False)
    print(f"[write] -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Define ancestry clusters by hierarchical clustering on principal components,
for within-cluster SPHERE rotation.

Ward linkage is built on a subsample (the full matrix is too large for
in-memory linkage), then every sample is assigned to the nearest sub-cluster
centroid; clusters smaller than --min-size are merged into the nearest
large one. The number of clusters is data-driven (roughly n / min_size).

    python3 step1_ancestry_clusters.py \\
        --pheno-csv $UKB_DIR/pheno.csv \\
        --roster    $UKB_DIR/roster.txt \\
        --out       ancestry_groups.csv \\
        --n-pc 40 --min-size 1000 --seed 0
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster


def hclust_groups(P, min_size, sample_for_tree=40_000, seed=0):
    """Ward linkage on a subsample, then assign all rows to nearest centroid.

    The initial cut produces roughly ``n // min_size`` groups so that the
    number of clusters is data-driven rather than a fixed constant. Clusters
    below `min_size` are then iteratively merged into the nearest large
    cluster until every group meets the threshold (or only one remains).
    """
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    sub = rng.choice(n, size=min(sample_for_tree, n), replace=False)
    Zl = linkage(P[sub], method="ward")
    n_init = max(n // min_size, 2)
    sub_lab = fcluster(Zl, t=n_init, criterion="maxclust")
    labs = np.unique(sub_lab)
    cents = np.vstack([P[sub][sub_lab == L].mean(0) for L in labs])
    grp = labs[(((P[:, None, :] - cents[None, :, :]) ** 2).sum(2)).argmin(1)]
    while True:
        gids, cnts = np.unique(grp, return_counts=True)
        small = gids[cnts < min_size]
        if small.size == 0 or gids.size <= 1:
            break
        big = gids[cnts >= min_size]
        if big.size == 0:
            break
        gc = {g: P[grp == g].mean(0) for g in gids}
        for sg in small:
            dd = {bg: np.sum((gc[sg] - gc[bg]) ** 2) for bg in big}
            grp[grp == sg] = min(dd, key=dd.get)
    _, grp = np.unique(grp, return_inverse=True)
    return grp


def main():
    ap = argparse.ArgumentParser(
        description="Ancestry clustering on PCs for within-cluster SPHERE rotation")
    ap.add_argument("--pheno-csv", required=True,
                    help="CSV with IID + PC1..PCn (official PCs)")
    ap.add_argument("--roster", required=True,
                    help="roster.txt (FID IID) — samples to assign, in order")
    ap.add_argument("--out", required=True, help="output group file (IID,group)")
    ap.add_argument("--n-pc", type=int, default=40)
    ap.add_argument("--min-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rst = pd.read_csv(args.roster, sep=r"\s+", header=None,
                      names=["FID", "IID"], dtype=str)
    rid = rst["IID"].tolist()
    pc_cols = [f"PC{i}" for i in range(1, args.n_pc + 1)]
    ph = (pd.read_csv(args.pheno_csv, usecols=lambda c: c in (["IID"] + pc_cols),
                      dtype={"IID": str}).set_index("IID").reindex(rid))
    P = ph[pc_cols].to_numpy(np.float64)
    miss = np.isnan(P).any(1)
    if miss.any():
        P = np.where(np.isnan(P), np.nanmean(P, 0), P)
        print(f"[groups] {int(miss.sum())} samples had missing PCs -> mean-filled")

    print(f"[groups] roster n={len(rid):,}  cluster on PC1-{args.n_pc}  "
          f"min_size={args.min_size}")
    grp = hclust_groups(P, args.min_size, seed=args.seed)
    gids, cnts = np.unique(grp, return_counts=True)
    print(f"[groups] {gids.size} groups  sizes={sorted(cnts.tolist())}")

    out = pd.DataFrame({"IID": rid, "group": grp.astype(int)})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[groups] -> {args.out}")


if __name__ == "__main__":
    main()

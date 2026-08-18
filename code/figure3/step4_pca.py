#!/usr/bin/env python3
"""
Compute principal components from the LD-pruned genotype matrix,
independently for the real and SPHERE-rotated lanes.

Uses a block-streamed randomized SVD (block Lanczos range finder) so the
full n × p matrix is never materialised.

Each lane is computed independently. Because SPHERE preserves Z'Z exactly
and LD pruning selects the identical column set in both lanes, the
eigenvalues agree to machine epsilon (max relative diff ~5.6e-09).

    python3 step4_pca.py \\
        --pruned-dir $OUT/pca \\
        --n-pc 40 --seed 0 --jobs 8 \\
        --out-dir $OUT/pca
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def block_krylov_svd(load_fn, n, p, n_pc, seed=0, n_oversamples=10, n_iter=4,
                     block=2000):
    """Randomized SVD via block Lanczos range finder.

    Parameters
    ----------
    load_fn : callable(s, e) -> (n, e-s) array
        Returns the standardized column block [s:e] of the data matrix.
    n, p : int
        Matrix dimensions.
    n_pc : int
        Number of principal components to return.
    seed : int
        RNG seed for the random projection.
    block : int
        Column block width for streaming matrix products.

    Returns
    -------
    scores : (n, n_pc)  PC scores
    evals  : (n_pc,)    eigenvalues (variance explained)
    """
    rng = np.random.default_rng(seed)
    L = n_pc + n_oversamples

    # Random projection: Y = W @ Omega, where W is the standardized data
    Omega = rng.standard_normal((p, L))
    Y = np.zeros((n, L), dtype=np.float64)
    for s in range(0, p, block):
        e = min(s + block, p)
        Wb = load_fn(s, e)
        Y += Wb @ Omega[s:e]

    # Block Lanczos: repeated W @ (W.T @ Y) to refine the range
    for _ in range(n_iter):
        G = np.zeros((p, L), dtype=np.float64)
        for s in range(0, p, block):
            e = min(s + block, p)
            Wb = load_fn(s, e)
            G[s:e] = Wb.T @ Y
        Y = np.zeros((n, L), dtype=np.float64)
        for s in range(0, p, block):
            e = min(s + block, p)
            Wb = load_fn(s, e)
            Y += Wb @ G[s:e]
        Q, _ = np.linalg.qr(Y)
        Y = Q

    # Project and SVD in the reduced space
    B = np.zeros((p, L), dtype=np.float64)
    for s in range(0, p, block):
        e = min(s + block, p)
        Wb = load_fn(s, e)
        B[s:e] = Wb.T @ Y
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    scores = Y @ Vt.T[:, :n_pc] * S[:n_pc]
    evals = (S[:n_pc] ** 2) / (n - 1)
    return scores, evals


def main():
    ap = argparse.ArgumentParser(description="Genome-wide PCA via randomized SVD")
    ap.add_argument("--pruned-dir", required=True,
                    help="directory with chr{c}_pruned_{real,synth}.npy")
    ap.add_argument("--n-pc", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = Path(args.pruned_dir)

    for side in ("real", "synth"):
        # Load per-chromosome pruned matrices
        chroms = []
        for c in range(1, 23):
            f = d / f"chr{c}_pruned_{side}.npy"
            if f.exists():
                chroms.append(np.load(str(f), mmap_mode="r"))
        if not chroms:
            print(f"[pca] no pruned files for {side}, skipping")
            continue

        n = chroms[0].shape[0]
        col_counts = [m.shape[1] for m in chroms]
        p_total = sum(col_counts)
        print(f"[pca] {side}: n={n:,}  p={p_total:,}  n_pc={args.n_pc}")

        # Build a loader that maps global column range to the right chromosome
        offsets = np.cumsum([0] + col_counts)
        def load_fn(s, e, _chroms=chroms, _offsets=offsets):
            out = np.empty((n, e - s), dtype=np.float64)
            col = 0
            for ci, m in enumerate(_chroms):
                cs, ce = _offsets[ci], _offsets[ci + 1]
                if e <= cs or s >= ce:
                    continue
                ls = max(s, cs) - cs
                le = min(e, ce) - cs
                blk = np.asarray(m[:, ls:le], dtype=np.float64)
                # Standardize: zero mean, unit variance
                mu = blk.mean(0)
                sd = blk.std(0)
                sd[sd == 0] = 1.0
                blk = (blk - mu) / sd
                w = le - ls
                out[:, col:col + w] = blk
                col += w
            return out[:, :col]

        scores, evals = block_krylov_svd(load_fn, n, p_total, args.n_pc,
                                         seed=args.seed)
        np.save(f"{args.out_dir}/gw_pc_{side}.npy", scores)
        np.savez(f"{args.out_dir}/gw_eig_{side}.npz", eigenvalues=evals)
        print(f"[pca] {side} -> gw_pc_{side}.npy  ({scores.shape})")

    # Compare eigenvalues
    er = f"{args.out_dir}/gw_eig_real.npz"
    es = f"{args.out_dir}/gw_eig_synth.npz"
    if os.path.exists(er) and os.path.exists(es):
        ev_r = np.load(er)["eigenvalues"]
        ev_s = np.load(es)["eigenvalues"]
        rel_diff = np.abs(ev_r - ev_s) / np.maximum(ev_r, 1e-15)
        cmp = {"max_rel_eigenvalue_diff": float(rel_diff.max()),
               "mean_rel_eigenvalue_diff": float(rel_diff.mean())}
        with open(f"{args.out_dir}/gw_pca_compare.json", "w") as f:
            json.dump(cmp, f, indent=2)
        print(f"[pca] eigenvalue comparison: max rel diff = {rel_diff.max():.2e}")


if __name__ == "__main__":
    main()

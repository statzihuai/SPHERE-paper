#!/usr/bin/env python3
"""
LD pruning on the raw genotype matrix, computed independently in each lane
(real and SPHERE).

SPHERE rotation produces continuous dosages outside [0, 2], so PLINK2 cannot
read the rotated matrix. This script computes pairwise r² directly in float64
and applies the same greedy window algorithm as plink2 --indep-pairwise.
Because SPHERE preserves Z'Z exactly, both lanes select the identical
variant set (261,793 in the UKB WGS analysis).

    python3 step3_ld_prune.py \\
        --real-npy output/genotype_real.npy \\
        --sphere-npy output/genotype_sphere.npy \\
        --out-dir output/pruned
"""
import argparse
import os

import numpy as np
from numba import njit


# ── Column standardisation ───────────────────────────────────────────────

def standardize_f64(Z):
    """Center each column and scale to unit L2 norm, all in float64.

    After this, Z[:, i] @ Z[:, j] is exactly the Pearson correlation.
    Zero-variance columns become all-zero (correlation defined as 0).
    """
    W = Z.astype(np.float64, copy=True)
    W -= W.mean(axis=0)
    nrm = np.sqrt(np.einsum("ij,ij->j", W, W))
    safe = np.where(nrm > 0, nrm, 1.0)
    W /= safe
    W[:, nrm == 0] = 0
    return W


# ── Banded r² computation ───────────────────────────────────────────────

def compute_band_r2(Zn, bandwidth, block=2000):
    """Compute r² for every variant pair at lag 1..bandwidth.

    --indep-pairwise only compares variants inside a sliding window, so
    no pair further apart than (bandwidth) positions is ever examined.
    Storing r² for lags 1..bandwidth as a (bandwidth, p) array is
    therefore sufficient.

    Zn must already be column-standardized to unit L2 norm.
    Returns a (bandwidth, p) float64 array where out[d-1, i] = r²(i, i+d).
    """
    n, p = Zn.shape
    out = np.zeros((bandwidth, p), np.float64)
    for s in range(0, p, block):
        e = min(s + block, p)
        hi = min(e + bandwidth, p)
        # G[a, b] = correlation(s+a, s+b)
        G = Zn[:, s:e].T @ Zn[:, s:hi]
        b = e - s
        rows = np.arange(b)
        for d in range(1, bandwidth + 1):
            valid = rows[(rows + d) < (hi - s)]
            if valid.size == 0:
                break
            r = G[valid, valid + d]
            out[d - 1, s + valid] = r * r
    return out


# ── Greedy window pruner ────────────────────────────────────────────────

@njit(cache=True)
def _prune_band(band_r2, maf, ws, step, thr, use_maf):
    """Greedy window prune mirroring plink2 --indep-pairwise.

    Slides a window of ws variants by step positions. Inside each window,
    repeatedly drops one member of any surviving pair with r² > thr until
    no such pair remains.  Tie-breaking: when use_maf is True, the variant
    with lower MAF is dropped (matches plink2 default behaviour).

    Returns a uint8 removal mask (1 = removed).
    """
    bw, p = band_r2.shape
    removed = np.zeros(p, np.uint8)
    start = 0
    while True:
        end = start + ws
        if end > p:
            end = p
        while True:
            found = False
            for i in range(start, end):
                if removed[i] == 1:
                    continue
                for j in range(i + 1, end):
                    if removed[j] == 1:
                        continue
                    d = j - i
                    if d > bw:
                        break
                    if band_r2[d - 1, i] > thr:
                        if use_maf:
                            drop = i if maf[i] < maf[j] else j
                        else:
                            drop = j
                        removed[drop] = 1
                        found = True
                        if drop == i:
                            break
                if found and removed[i] == 1:
                    continue
            if not found:
                break
        if end >= p:
            break
        start += step
    return removed


def ld_prune(Z, maf, ws=50, step=5, thr=0.1, maf_min=0.05):
    """Full LD-pruning pipeline: MAF filter → standardize → band r² → prune.

    Returns a boolean keep-mask over the original p variants.
    """
    maf_keep = maf >= maf_min
    Z_filt = Z[:, maf_keep].astype(np.float64)
    maf_filt = maf[maf_keep]

    Zn = standardize_f64(Z_filt)
    band = compute_band_r2(Zn, ws - 1)
    removed = _prune_band(band, np.ascontiguousarray(maf_filt, np.float64),
                          ws, step, thr, True)

    keep_inner = removed == 0
    keep = np.zeros(len(maf), dtype=bool)
    keep[maf_keep] = keep_inner
    return keep


def main():
    ap = argparse.ArgumentParser(
        description="Float64 LD pruning (replaces plink2 --indep-pairwise)")
    ap.add_argument("--real-npy", required=True,
                    help="genotype matrix, real lane (n x p)")
    ap.add_argument("--sphere-npy", required=True,
                    help="genotype matrix, SPHERE lane (n x p)")
    ap.add_argument("--ws", type=int, default=50)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--maf-min", type=float, default=0.05)
    args = ap.parse_args()

    Z_real = np.load(args.real_npy)
    Z_sphere = np.load(args.sphere_npy)
    n, p = Z_real.shape
    maf = Z_real.mean(axis=0).astype(np.float64) / 2.0

    # Prune each lane independently
    keep_r = ld_prune(Z_real, maf, args.ws, args.step, args.thr, args.maf_min)
    keep_s = ld_prune(Z_sphere, maf, args.ws, args.step, args.thr, args.maf_min)

    # Because SPHERE preserves Z'Z, the keep sets must be identical
    assert np.array_equal(keep_r, keep_s), (
        f"keep sets differ: real={keep_r.sum()}, sphere={keep_s.sum()}, "
        f"real_only={(keep_r & ~keep_s).sum()}, sphere_only={(~keep_r & keep_s).sum()}")

    print(f"[prune] n={n:,}  p={p:,}  kept={keep_r.sum():,}  identical=True")


if __name__ == "__main__":
    main()

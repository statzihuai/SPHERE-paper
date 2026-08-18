#!/usr/bin/env python3
"""
Apply SPHERE rotation within ancestry clusters.

Each cluster is rotated independently so no individual is mixed across
ancestry boundaries. AD status, sex and age are co-rotated with the
genotypes using the same key (seed-based — no key file needed).

The PCs are NOT co-rotated; each lane recomputes its own PCs from its
own genotype matrix (step 4).

    python3 step2_sphere_rotate.py \\
        --dosage-prefix $UKB_DIR/dosage/chr1_dosage \\
        --groups ancestry_groups.csv \\
        --roster $UKB_DIR/roster.txt \\
        --pheno-csv $UKB_DIR/pheno.csv \\
        --seed 0 --ad-coding pm1 \\
        --out-dir output/
"""
import argparse
import sys

import numpy as np
import pandas as pd



def _sphere2(Z):
    """Two iterations of SPHERE (k = 2)."""
    return sphere(sphere(Z))


PHENO_COL = "AD"
COVARS = ["sex", "age"]


def rotate_within_ancestry(M, groups, seed):
    """SPHERE block-diagonal rotation: one independent key per cluster.

    For each ancestry cluster g, seeds the RNG at ``seed + g * 1000``
    and applies two SPHERE iterations.  The same seed is used for
    genotype blocks and for phenotype/covariates, so identical rotation
    keys are applied to both.
    """
    M_out = M.astype(float, copy=True)
    for g in np.unique(groups):
        mask = (groups == g)
        np.random.seed(seed + g * 1000)
        M_out[mask] = _sphere2(M_out[mask])
    return M_out


def main():
    ap = argparse.ArgumentParser(
        description="SPHERE within-ancestry rotation of genotype + phenotype")
    ap.add_argument("--dosage-prefix", required=True,
                    help="PLINK2 pgen/pvar/psam prefix")
    ap.add_argument("--groups", required=True,
                    help="ancestry_groups.csv from step 1")
    ap.add_argument("--roster", required=True, help="roster.txt (FID IID)")
    ap.add_argument("--pheno-csv", required=True,
                    help="CSV with IID, AD, sex, age")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=2000,
                    help="column block width for streaming rotation")
    ap.add_argument("--ad-coding", choices=("01", "pm1"), default="pm1",
                    help="AD coding: '01' keeps {0,1}; 'pm1' maps to {-1,+1} "
                         "before rotation. OLS t/p are identical either way.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    # ── Load roster, ancestry groups, phenotype ──
    rst = pd.read_csv(args.roster, sep=r"\s+", header=None,
                      names=["FID", "IID"], dtype=str)
    roster = rst["IID"].to_numpy()
    anc = pd.read_csv(args.groups, dtype={"IID": str})
    assert (anc["IID"].to_numpy() == roster).all(), "ancestry/roster order mismatch"
    groups = anc["group"].to_numpy().astype(int)

    ph = pd.read_csv(args.pheno_csv, dtype={"IID": str}).set_index("IID").reindex(roster)
    y = ph[PHENO_COL].to_numpy(np.float64)
    if args.ad_coding == "pm1":
        y = 2.0 * y - 1.0
    covars = ph[COVARS].to_numpy(np.float64)

    # ── Co-rotate phenotype + covariates ──
    YC = rotate_within_ancestry(
        np.column_stack([y, covars]), groups, args.seed)
    y_synth, cov_synth = YC[:, 0], YC[:, 1:]

    print(f"[rotate] n={len(roster):,}  groups={len(np.unique(groups))}  "
          f"ad_coding={args.ad_coding}")
    print(f"[rotate] phenotype + covariates co-rotated")
    print(f"[rotate] genotype rotation requires streaming through pgen blocks")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AD GWAS via OLS on all filtered variants, independently in each lane.

Each lane uses its own PCs (from step 4); phenotype and non-PC covariates
are co-rotated with the genotypes (step 2).

Model (each lane independently):
    AD ~ SNP_j + sex + age + PC_1 + ... + PC_40

The regression uses Frisch–Waugh–Lovell: residualise y and SNP on the
covariates, then the slope of the residualised SNP on the residualised y
gives the exact multiple-regression beta, SE and p.

    python3 step5_gwas.py \\
        --dosage-prefix $UKB_DIR/dosage/chr1_dosage \\
        --pc-real gw_pc_real.npy --pc-synth gw_pc_synth.npy \\
        --pheno-csv $UKB_DIR/pheno.csv \\
        --groups ancestry_groups.csv --roster $UKB_DIR/roster.txt \\
        --npc 40 --ad-coding pm1 --out-dir output/gwas
"""
import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GWASResult:
    beta: np.ndarray
    se: np.ndarray
    p: np.ndarray


def gwas_ols(Z, y, C):
    """Frisch-Waugh-Lovell OLS: exact multiple-regression beta/SE/p.

    Residualise y and each column of Z on the covariate matrix C, then
    regress the residualised SNP on the residualised y. This gives the
    same beta as the full model (y ~ SNP + C) but avoids inverting a
    (p_covars x p_covars) matrix per SNP.
    """
    n, p = Z.shape
    # QR of covariates for numerical stability
    Q, _ = np.linalg.qr(np.column_stack([np.ones(n), C]))
    # Residualise y
    y_res = y - Q @ (Q.T @ y)
    # Residualise Z block-wise (memory)
    beta = np.empty(p)
    se = np.empty(p)
    pval = np.empty(p)
    dof = n - Q.shape[1] - 1
    for j in range(p):
        z_j = Z[:, j]
        z_res = z_j - Q @ (Q.T @ z_j)
        zz = z_res @ z_res
        if zz < 1e-30:
            beta[j] = se[j] = pval[j] = np.nan
            continue
        b = (z_res @ y_res) / zz
        resid = y_res - b * z_res
        s2 = (resid @ resid) / dof
        se_b = np.sqrt(s2 / zz)
        t = b / se_b if se_b > 0 else 0.0
        beta[j] = b
        se[j] = se_b
        pval[j] = 2.0 * stats.t.sf(abs(t), dof)
    return GWASResult(beta=beta, se=se, p=pval)


def main():
    ap = argparse.ArgumentParser(description="AD GWAS via OLS, real vs SPHERE")
    ap.add_argument("--dosage-prefix", required=True)
    ap.add_argument("--pc-real", required=True)
    ap.add_argument("--pc-synth", required=True)
    ap.add_argument("--pheno-csv", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--roster", required=True)
    ap.add_argument("--npc", type=int, default=40)
    ap.add_argument("--ad-coding", choices=("01", "pm1"), default="pm1")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load sample metadata
    rst = pd.read_csv(args.roster, sep=r"\s+", header=None,
                      names=["FID", "IID"], dtype=str)
    roster = rst["IID"].to_numpy()
    ph = pd.read_csv(args.pheno_csv, dtype={"IID": str}).set_index("IID").reindex(roster)

    y_real = ph["AD"].to_numpy(np.float64)
    if args.ad_coding == "pm1":
        y_real = 2.0 * y_real - 1.0
    cov_real = ph[["sex", "age"]].to_numpy(np.float64)

    pc_real = np.load(args.pc_real)[:, :args.npc]
    pc_synth = np.load(args.pc_synth)[:, :args.npc]
    C_real = np.column_stack([cov_real, pc_real])

    # Co-rotated phenotype/covariates would come from step 2; for the GWAS
    # comparison, the synthetic side needs (y_synth, cov_synth, pc_synth).
    # This script documents the model; the actual streaming GWAS over ~9.7M
    # variants requires the SPHERE key-generation internals for co-rotation
    # (see step2_sphere_rotate.py).

    print(f"[gwas] n={len(roster):,}  npc={args.npc}  ad_coding={args.ad_coding}")
    print(f"[gwas] model: AD ~ SNP + sex + age + PC1..PC{args.npc}")
    print(f"[gwas] real covariate matrix: {C_real.shape}")
    print(f"[gwas] streaming GWAS over ~9.7M variants requires the full pipeline")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 3 — write the rotated joint matrix back out as per-modality CSVs.

Each real CSV is mirrored column for column under sphere_xk/x{k}/, in the same
participant order, so every downstream script can point at either tree without
changing a line. Participants left entirely missing across a modality's real
columns are dropped from that modality's file, which reproduces the block
structure of the real release.

    python3 step3_extract_modality_csvs.py \\
        --joint-dir results/figure4 --k 2 \\
        --real-root data/adrc/real \\
        --out-dir   results/figure4/sphere_xk/x2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adrc_common import ID_COL, MODALITY_REL, SKIP_ID


def extract_sphere_csvs(joint_dir: Path, k: int, real_root: Path, dst: Path) -> None:
    print(f"\nExtracting SPHERE×{k} CSVs → {dst}")
    Z_cont = np.load(joint_dir / f"sphere_joint_matrix_x{k}_cont.npy").astype(np.float64)
    Z_cat  = np.load(joint_dir / f"sphere_joint_matrix_x{k}_cat.npy", allow_pickle=True)
    cols_cont = np.load(joint_dir / "joint_matrix_cols_cont.npy", allow_pickle=True).astype(str)
    cols_cat  = np.load(joint_dir / "joint_matrix_cols_cat.npy",  allow_pickle=True).astype(str)
    syn_ids   = np.load(joint_dir / "joint_matrix_ids.npy",       allow_pickle=True).astype(str)

    id_to_row    = {pid: i for i, pid in enumerate(syn_ids)}
    cont_col_idx = {c: i for i, c in enumerate(cols_cont)}
    cat_col_idx  = {c: i for i, c in enumerate(cols_cat)}

    for prefix, rel_path in MODALITY_REL.items():
        real_path = real_root / rel_path
        if not real_path.exists():
            print(f"  {prefix}: SKIP — {real_path} not found")
            continue
        dst_path = dst / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        real_df   = pd.read_csv(real_path, low_memory=False, dtype={ID_COL: str})
        real_cols = [c for c in real_df.columns if c != ID_COL and c not in SKIP_ID]

        p = f"{prefix}__"
        cont_jcols = {c.split("__", 1)[1]: cont_col_idx[c] for c in cols_cont if c.startswith(p)}
        cat_jcols  = {c.split("__", 1)[1]: cat_col_idx[c]  for c in cols_cat  if c.startswith(p)}

        # ±1 orth columns added at joint-matrix-build time (not in the real CSV)
        orth_extra   = sorted(c for c in cont_jcols if "_orth_" in c and c not in set(real_cols))
        all_out_cols = real_cols + orth_extra

        # Ordered participant rows present in the joint matrix (preserves real_df order)
        pids      = [str(pid) for pid in real_df[ID_COL].values if str(pid) in id_to_row]
        skipped   = len(real_df) - len(pids)
        row_idx   = np.array([id_to_row[pid] for pid in pids], dtype=int)

        data = {ID_COL: pids}
        for col in all_out_cols:
            if col in cont_jcols:
                vals = Z_cont[row_idx, cont_jcols[col]]
                data[col] = np.where(np.isfinite(vals), vals, np.nan)
            elif col in cat_jcols:
                data[col] = Z_cat[row_idx, cat_jcols[col]]
            else:
                data[col] = np.full(len(pids), np.nan)   # in real CSV but not joint matrix

        out_df = pd.DataFrame(data, columns=[ID_COL] + all_out_cols)
        # Drop participants entirely missing across the real data columns
        out_df = out_df.dropna(subset=real_cols, how="all").reset_index(drop=True)
        out_df.to_csv(dst_path, index=False)
        print(f"  {prefix}: {len(out_df)} rows, {len(real_cols)} cols + {len(orth_extra)} orth "
              f"(skipped {skipped}) → {dst_path.name}")


def main():
    ap = argparse.ArgumentParser(description="Extract per-modality CSVs from the rotated joint matrix")
    ap.add_argument("--joint-dir", type=Path, required=True, help="Step 1/2 output directory")
    ap.add_argument("--k", type=int, required=True, help="Which rotation count to extract")
    ap.add_argument("--real-root", type=Path, required=True,
                    help="Real tree, read for the column list and participant order")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    extract_sphere_csvs(args.joint_dir, args.k, args.real_root, args.out_dir)


if __name__ == "__main__":
    main()

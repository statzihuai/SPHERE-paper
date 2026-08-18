#!/usr/bin/env python3
"""
Step 1 — assemble the nine real ADRC modality CSVs into one joint matrix.

Participants are the union of adrc_id over the modalities, so the matrix is
block-missing: whoever was not measured on a modality carries NaN (continuous)
or None (categorical) across that whole block. Step 2 turns exactly that mask
into rotation sites.

Columns are named `<modality>__<column>`; object columns land in the
categorical array, everything else in a float32 continuous array.

    python3 step1_build_joint_matrix.py \\
        --real-root data/adrc/real \\
        --out-dir   results/figure4
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adrc_common import ID_COL, MODALITY_KEYS, MODALITY_REL


def build_joint_matrix(real_root: Path, out_dir: Path) -> None:
    """Assemble all 9 real modality CSVs into the joint matrix arrays.

    Categorical columns (object dtype) → joint_matrix_cat.npy (strings).
    Numeric columns (incl. the pre-computed ±1 orth demographic contrasts) →
    joint_matrix_cont.npy (float32). Column names are prefixed ``key__colname``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading modality CSVs …")
    modality_dfs, cont_col_list, cat_col_list, all_ids = {}, [], [], set()

    for key in MODALITY_KEYS:
        path = real_root / MODALITY_REL[key]
        if not path.exists():
            print(f"  {key}: SKIP — {path} not found")
            continue
        print(f"  {key} … ", end="", flush=True)
        df = pd.read_csv(path, dtype={ID_COL: str}, low_memory=False)
        df = df.drop_duplicates(subset=[ID_COL]).set_index(ID_COL)
        for col in df.columns:
            (cat_col_list if pd.api.types.is_object_dtype(df[col]) else cont_col_list).append((key, col))
        modality_dfs[key] = df
        all_ids.update(df.index.tolist())
        print(f"{df.shape}  (cont={sum(1 for k,_ in cont_col_list if k==key)}, "
              f"cat={sum(1 for k,_ in cat_col_list if k==key)})")

    all_ids = sorted(all_ids)
    n, p_cont, p_cat = len(all_ids), len(cont_col_list), len(cat_col_list)
    print(f"\nParticipants: {n}  |  Continuous: {p_cont}  |  Categorical: {p_cat}")

    ids    = np.array(all_ids)
    Z_cont = np.full((n, p_cont), np.nan, dtype=np.float32)
    Z_cat  = np.full((n, p_cat),  None,   dtype=object)

    print("\nFilling matrices …")
    cont_off = cat_off = 0
    for key in MODALITY_KEYS:
        if key not in modality_dfs:
            continue
        df_aligned    = modality_dfs[key].reindex(all_ids)
        cont_cols_mod = [c for (k, c) in cont_col_list if k == key]
        cat_cols_mod  = [c for (k, c) in cat_col_list  if k == key]
        if cont_cols_mod:
            Z_cont[:, cont_off:cont_off + len(cont_cols_mod)] = \
                df_aligned[cont_cols_mod].values.astype(np.float32)
            cont_off += len(cont_cols_mod)
        if cat_cols_mod:
            cat_vals = df_aligned[cat_cols_mod].values.astype(object)
            for j, col in enumerate(cat_cols_mod):
                cat_vals[pd.isna(df_aligned[col]).values, j] = None
            Z_cat[:, cat_off:cat_off + len(cat_cols_mod)] = cat_vals
            cat_off += len(cat_cols_mod)

    cols_cont = np.array([f"{k}__{c}" for k, c in cont_col_list])
    cols_cat  = np.array([f"{k}__{c}" for k, c in cat_col_list])

    np.save(out_dir / "joint_matrix_cont.npy",      Z_cont)
    np.save(out_dir / "joint_matrix_cat.npy",       Z_cat)
    np.save(out_dir / "joint_matrix_ids.npy",       ids)
    np.save(out_dir / "joint_matrix_cols_cont.npy", cols_cont)
    np.save(out_dir / "joint_matrix_cols_cat.npy",  cols_cat)
    print(f"\nJoint matrix saved → {out_dir}")
    print(f"  cont {Z_cont.shape} float32  |  cat {Z_cat.shape} object  |  {len(ids)} participants")


def main():
    ap = argparse.ArgumentParser(description="Build the ADRC joint matrix from the real CSVs")
    ap.add_argument("--real-root", type=Path, required=True,
                    help="Directory holding the nine real modality CSVs")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where joint_matrix_*.npy are written")
    args = ap.parse_args()
    build_joint_matrix(args.real_root, args.out_dir)


if __name__ == "__main__":
    main()

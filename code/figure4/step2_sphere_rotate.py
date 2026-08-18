#!/usr/bin/env python3
"""
Step 2 — SPHERE rotation of the joint matrix, k times, per missingness site.

Rows are grouped by which modalities they carry; sites below --min-site-size
are merged into the site that costs the least rarity-weighted contamination.
Every site is then rotated as one block over all of its columns at once
(continuous and categorical together), which is what makes the cross-modality
covariance in panels f/g survive the rotation. Within a site,

    Z*'Z* = Z'Z  and column means are preserved exactly,

and the NaN/None mask is restored afterwards, so the released matrix has the
same block-missing shape as the real one.

    python3 step2_sphere_rotate.py \\
        --joint-dir results/figure4 \\
        --ks 1 2 3 4 5 --seed 0
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np


from adrc_common import KS_DEFAULT, MIN_SITE_SIZE, MODALITY_KEYS


def _col_mode(arr):
    vals = [v for v in arr if v is not None]
    return Counter(vals).most_common(1)[0][0] if vals else None


_is_none_vec = np.vectorize(lambda x: x is None)


def build_modality_sites(nan_mask_cont, none_mask_cat, prefixes_cont, prefixes_cat,
                         min_size=MIN_SITE_SIZE):
    """Group rows by unique modality-presence pattern, then merge small sites
    using a post-merge pairwise-purity (rarity-weighted contamination) cost."""
    n = nan_mask_cont.shape[0]

    presence = {}
    for m in MODALITY_KEYS:
        has_cont = np.zeros(n, dtype=bool)
        has_cat  = np.zeros(n, dtype=bool)
        cont_idx = np.where(prefixes_cont == m)[0]
        if cont_idx.size:
            has_cont = (~nan_mask_cont[:, cont_idx]).any(axis=1)
        cat_idx = np.where(prefixes_cat == m)[0]
        if cat_idx.size:
            has_cat = (~none_mask_cat[:, cat_idx]).any(axis=1)
        presence[m] = has_cont | has_cat

    mod_sizes = {m: int(presence[m].sum()) for m in MODALITY_KEYS}

    site_map: dict = {}
    for i in range(n):
        combo = tuple(m for m in MODALITY_KEYS if presence[m][i])
        site_map.setdefault(combo, []).append(i)

    print(f"  Unique missingness patterns: {len(site_map)}")
    print(f"  Size dist: {dict(sorted(Counter(len(v) for v in site_map.values()).items()))}")

    max_size = max(mod_sizes.values()) if mod_sizes else 1
    rarity   = {m: max_size / (mod_sizes[m] + 1) for m in MODALITY_KEYS}

    def merge_cost(src_rows, tgt_rows):
        all_rows = src_rows + tgt_rows
        n_total  = len(all_rows)
        merged   = [m for m in MODALITY_KEYS if any(presence[m][r] for r in all_rows)]
        total = 0.0
        for i, m1 in enumerate(merged):
            for m2 in merged[i + 1:]:
                n_both = sum(1 for r in all_rows if presence[m1][r] and presence[m2][r])
                total += (1.0 - n_both / n_total) * rarity[m1] * rarity[m2]
        return total

    while True:
        smalls = {c for c, r in site_map.items() if len(r) < min_size}
        if not smalls:
            break
        best_cost, best_src, best_tgt = float("inf"), None, None
        for src_combo in smalls:
            src_rows = site_map[src_combo]
            for tgt_combo, tgt_rows in site_map.items():
                if tgt_combo == src_combo:
                    continue
                c = merge_cost(src_rows, tgt_rows)
                if c < best_cost:
                    best_cost, best_src, best_tgt = c, src_combo, tgt_combo
        if best_src is None:
            break
        print(f"    merge {best_src} (n={len(site_map[best_src])}) → "
              f"{best_tgt} (n={len(site_map[best_tgt])})  cost={best_cost:.3f}")
        merged_combo = tuple(m for m in MODALITY_KEYS if m in set(best_src) | set(best_tgt))
        site_map[merged_combo] = site_map.pop(best_src) + site_map.pop(best_tgt)

    print(f"  After merging (min={min_size}): {len(site_map)} sites")
    print(f"  Size dist: {dict(sorted(Counter(len(v) for v in site_map.values()).items()))}")
    return site_map


def sphere_pass(Z_cont, Z_cat, rng, nan_mask_cont, none_mask_cat,
                prefixes_cont, prefixes_cat, min_site_size=MIN_SITE_SIZE):
    """One SPHERE rotation pass — per-site joint rotation of all columns.
    Z*ᵀZ* = ZᵀZ within each site; NaN/None masks are restored afterward."""
    from sphere import sphere   # code/sphere.py — the SPHERE core algorithm

    Zstar_cont = Z_cont.copy()
    Zstar_cat  = Z_cat.copy()
    col_means  = np.nanmean(Z_cont, axis=0)
    col_modes  = [_col_mode(Z_cat[:, j]) for j in range(Z_cat.shape[1])]

    site_map = build_modality_sites(nan_mask_cont, none_mask_cat,
                                    prefixes_cont, prefixes_cat, min_size=min_site_size)

    for combo, rows in site_map.items():
        if not combo:
            continue
        rows_arr = np.array(rows)
        if len(rows_arr) < 2:
            continue
        n_site   = len(rows_arr)
        cont_idx = np.where(np.isin(prefixes_cont, list(combo)))[0]
        cat_idx  = np.where(np.isin(prefixes_cat,  list(combo)))[0]

        # Impute continuous: NaN → global column mean
        sub_cont = (Z_cont[np.ix_(rows_arr, cont_idx)].copy() if cont_idx.size
                    else np.empty((n_site, 0), dtype=np.float64))
        if cont_idx.size and np.isnan(sub_cont).any():
            for ci, gci in enumerate(cont_idx):
                m = np.isnan(sub_cont[:, ci])
                if m.any():
                    sub_cont[m, ci] = col_means[gci] if np.isfinite(col_means[gci]) else 0.0

        # Impute categorical: None → global column mode
        sub_cat = (Z_cat[np.ix_(rows_arr, cat_idx)].copy() if cat_idx.size
                   else np.empty((n_site, 0), dtype=object))
        none_sub = _is_none_vec(sub_cat)
        for ci, gci in enumerate(cat_idx):
            m = none_sub[:, ci]
            if m.any() and col_modes[gci] is not None:
                sub_cat[m, ci] = col_modes[gci]

        # Combine (continuous first, then categorical) and rotate jointly
        if cont_idx.size and cat_idx.size:
            Z_full = np.concatenate([sub_cont, sub_cat], axis=1)
        elif cat_idx.size:
            Z_full = sub_cat
        elif cont_idx.size:
            Z_full = sub_cont
        else:
            continue

        seed = int(rng.integers(0, 2**31 - 1))
        np.random.seed(seed)                       # sphere() uses the global RNG
        Z_full_rot = sphere(Z_full, categorical_cols=None)

        if cont_idx.size:
            Zstar_cont[np.ix_(rows_arr, cont_idx)] = Z_full_rot[:, :cont_idx.size].astype(np.float64)
        if cat_idx.size:
            Zstar_cat[np.ix_(rows_arr, cat_idx)] = Z_full_rot[:, cont_idx.size:]

    # Restore original NaN / None masks
    Zstar_cont[nan_mask_cont] = np.nan
    nr, nc = np.where(none_mask_cat)
    for r, c in zip(nr, nc):
        Zstar_cat[r, c] = None
    return Zstar_cont, Zstar_cat


def run_sphere_xk(joint_dir: Path, ks, min_site_size, seed=None) -> None:
    """Apply SPHERE cumulatively k=1..max(ks); save sphere_joint_matrix_x{k}_{cont,cat}.npy."""
    print("\nLoading joint matrix …")
    Z_cont    = np.load(joint_dir / "joint_matrix_cont.npy").astype(np.float64)
    Z_cat     = np.load(joint_dir / "joint_matrix_cat.npy",       allow_pickle=True)
    cols_cont = np.load(joint_dir / "joint_matrix_cols_cont.npy", allow_pickle=True).astype(str)
    cols_cat  = np.load(joint_dir / "joint_matrix_cols_cat.npy",  allow_pickle=True).astype(str)
    print(f"  Continuous:  {Z_cont.shape}  NaN fraction:  {np.isnan(Z_cont).mean():.3f}")
    print(f"  Categorical: {Z_cat.shape}   None fraction: {_is_none_vec(Z_cat).mean():.3f}")

    prefixes_cont = np.array([c.split("__")[0] for c in cols_cont])
    prefixes_cat  = np.array([c.split("__")[0] for c in cols_cat])
    nan_mask_cont = np.isnan(Z_cont)
    none_mask_cat = _is_none_vec(Z_cat)

    master_rng = np.random.default_rng(seed)       # seed=None → fresh OS entropy
    Zc, Zk = Z_cont.copy(), Z_cat.copy()
    max_k = max(ks)
    for k in range(1, max_k + 1):
        print(f"\nRotation {k}/{max_k} …", flush=True)
        Zc, Zk = sphere_pass(Zc, Zk, rng=master_rng,
                             nan_mask_cont=nan_mask_cont, none_mask_cat=none_mask_cat,
                             prefixes_cont=prefixes_cont, prefixes_cat=prefixes_cat,
                             min_site_size=min_site_size)
        if k in ks:
            np.save(joint_dir / f"sphere_joint_matrix_x{k}_cont.npy", Zc.astype(np.float32))
            np.save(joint_dir / f"sphere_joint_matrix_x{k}_cat.npy",  Zk)
            print(f"  Saved x{k}: cont {Zc.shape}  cat {Zk.shape}  "
                  f"(NaN preserved: {(np.isnan(Zc) == nan_mask_cont).all()}, "
                  f"None preserved: {(_is_none_vec(Zk) == none_mask_cat).all()})")


def main():
    ap = argparse.ArgumentParser(description="Apply SPHERE k times to the ADRC joint matrix")
    ap.add_argument("--joint-dir", type=Path, required=True,
                    help="Directory holding joint_matrix_*.npy (step 1 output); the "
                         "rotated matrices are written next to them")
    ap.add_argument("--ks", type=int, nargs="+", default=KS_DEFAULT,
                    help="Rotation counts to save (the pass is cumulative)")
    ap.add_argument("--min-site-size", type=int, default=MIN_SITE_SIZE)
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed of the master RNG; omit for OS entropy")
    args = ap.parse_args()
    run_sphere_xk(args.joint_dir, sorted(set(args.ks)),
                  min_site_size=args.min_site_size, seed=args.seed)


if __name__ == "__main__":
    main()

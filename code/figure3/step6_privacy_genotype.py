#!/usr/bin/env python3
"""
Anonymeter privacy evaluation of SPHERE-rotated genotypes, per ancestry
cluster (groups × 10 independent draws).

Each draw samples a 200-variant pool from the shared pruned frame, then
runs three Anonymeter attacks (singling-out, linkability, inference) on
min(2000, n_cluster) rows per cluster. ColShuffle and Real anchors are
computed per cell for two-anchor normalisation.

Requires anonymeter==1.0.0 (Python < 3.12).

    python3 step6_privacy_genotype.py \\
        --pruned-dir $OUT/pca \\
        --groups ancestry_groups.csv \\
        --roster $OUT/roster.txt \\
        --out cluster_privacy.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Unified privacy spec ──────────────────────────────────────────────────
N_DRAWS = 10
POOL_SIZE = 200          # variants per draw
NFEAT = 20               # LK/INF random-20 of the pool
N_ROWS = 2000            # row subsample per cluster
N_ATTACKS = 500
SO_N_COLS = 3
SO_MAX_ATTEMPTS = 20_000
SEED = 0
METRICS = ("singling_out", "linkability", "inference")

# RNG stream IDs for independent per-cell seeds
STREAM_POOL, STREAM_ROWS, STREAM_FEATS, STREAM_EVAL = 1, 2, 3, 4


def cell_rng(stream, draw, unit):
    """Deterministic RNG for one stream of one (draw, unit) cell."""
    return np.random.default_rng([SEED, stream, draw, unit + 1])


def reseed(sd):
    """Reset all three of anonymeter's randomness sources."""
    import anonymeter.evaluators.singling_out_evaluator as _so
    _orig = np.random.default_rng
    np.random.seed(sd)
    _so.rng = np.random.default_rng(sd)
    np.random.default_rng = (lambda *a, **k: _orig(sd)
                             if (not a and not k) else _orig(*a, **k))
    return _orig


def run_3axis(ori, syn, all_cols, feats, n_attacks, seed):
    """Run Anonymeter singling-out, linkability and inference attacks."""
    from anonymeter.evaluators import (SinglingOutEvaluator,
                                       LinkabilityEvaluator,
                                       InferenceEvaluator)
    half = len(feats) // 2
    r = {}

    _orig = reseed(seed * 97 + 1)
    try:
        ev = SinglingOutEvaluator(ori=ori, syn=syn, n_attacks=N_ATTACKS,
                                  n_cols=SO_N_COLS, max_attempts=SO_MAX_ATTEMPTS)
        ev.evaluate(mode="multivariate")
        r["singling_out"] = float(ev.risk().value)
    except Exception:
        r["singling_out"] = float("nan")

    reseed(seed * 97 + 2)
    try:
        ev = LinkabilityEvaluator(ori=ori, syn=syn, n_attacks=n_attacks,
                                  aux_cols=(feats[:half], feats[half:]))
        ev.evaluate()
        r["linkability"] = float(ev.risk().value)
    except Exception:
        r["linkability"] = float("nan")

    reseed(seed * 97 + 3)
    try:
        ev = InferenceEvaluator(ori=ori, syn=syn, n_attacks=n_attacks,
                                aux_cols=feats[1:], secret=feats[0])
        ev.evaluate()
        r["inference"] = float(ev.risk().value)
    except Exception:
        r["inference"] = float("nan")

    np.random.default_rng = _orig
    return r


def two_anchor(sphere_val, real_val, colshuf_val):
    """Two-anchor normalisation: Real = 0%, ColShuffle = 100%."""
    d = colshuf_val - real_val
    if abs(d) < 1e-10:
        return float("nan")
    raw = (sphere_val - real_val) / d * 100
    return float(np.clip(raw, 0, 100))


# ── Worker ────────────────────────────────────────────────────────────────
_G = {}


def _init_worker(pruned_dir, grp_arr, chroms):
    _G["pruned_dir"] = pruned_dir
    _G["grp"] = grp_arr
    _G["chroms"] = chroms


def _process(payload):
    di, unit = payload
    pruned_dir = _G["pruned_dir"]
    grp = _G["grp"]
    chroms = _G["chroms"]
    t0 = time.time()

    # Variant pool: drawn per draw (shared across clusters for I/O efficiency)
    pool_rng = cell_rng(STREAM_POOL, di, 0)

    # Count total pruned variants across chromosomes
    col_counts = []
    for c in chroms:
        ar = np.load(f"{pruned_dir}/chr{c}_pruned_real.npy", mmap_mode="r")
        col_counts.append(ar.shape[1])
    total_p = sum(col_counts)
    offsets = np.cumsum([0] + col_counts)

    pool_idx = np.sort(pool_rng.choice(total_p, size=POOL_SIZE, replace=False))

    # Read pooled variants for both lanes
    n = len(grp)
    Zr = np.empty((n, POOL_SIZE), dtype=np.float32)
    Zs = np.empty((n, POOL_SIZE), dtype=np.float32)
    for ci, c in enumerate(chroms):
        mask = (pool_idx >= offsets[ci]) & (pool_idx < offsets[ci + 1])
        if not mask.any():
            continue
        local_cols = pool_idx[mask] - offsets[ci]
        out_cols = np.flatnonzero(mask)
        ar = np.load(f"{pruned_dir}/chr{c}_pruned_real.npy", mmap_mode="r")
        sr = np.load(f"{pruned_dir}/chr{c}_pruned_synth.npy", mmap_mode="r")
        Zr[:, out_cols] = ar[:, local_cols]
        Zs[:, out_cols] = sr[:, local_cols]

    # Row subsample for this cluster
    ridx = np.flatnonzero(grp == unit)
    n_unit = len(ridx)
    row_rng = cell_rng(STREAM_ROWS, di, unit)
    take = row_rng.choice(n_unit, size=min(N_ROWS, n_unit), replace=False)
    rsel = ridx[take]

    cols = [f"c{k}" for k in range(POOL_SIZE)]
    real = pd.DataFrame(np.asarray(Zr[rsel], dtype=float), columns=cols)
    syn = pd.DataFrame(np.asarray(Zs[rsel], dtype=float), columns=cols)
    cs = pd.DataFrame({c: real[c].to_numpy()[row_rng.permutation(len(real))]
                       for c in cols})

    # Features drawn per cell (not per draw)
    frng = cell_rng(STREAM_FEATS, di, unit)
    feats = list(frng.choice(cols, size=min(NFEAT, len(cols)), replace=False))

    ev_seed = int(cell_rng(STREAM_EVAL, di, unit).integers(1, 2 ** 24))
    r_real = run_3axis(real, real, cols, feats, N_ATTACKS, ev_seed)
    r_cs = run_3axis(real, cs, cols, feats, N_ATTACKS, ev_seed)
    r_sph = run_3axis(real, syn, cols, feats, N_ATTACKS, ev_seed)

    row = {"draw": di, "unit": int(unit), "n_unit": int(n_unit),
           "n_rows_used": int(len(rsel)),
           "elapsed_s": round(time.time() - t0, 1)}
    for m in METRICS:
        row[m] = two_anchor(r_sph[m], r_real[m], r_cs[m])
        row[f"{m}_raw"] = {"sphere": r_sph[m], "real": r_real[m],
                           "cs": r_cs[m]}
    return row


def main():
    ap = argparse.ArgumentParser(
        description="Per-cluster Anonymeter privacy of SPHERE genotypes")
    ap.add_argument("--pruned-dir", required=True,
                    help="directory with chr{c}_pruned_{real,synth}.npy")
    ap.add_argument("--groups", required=True,
                    help="ancestry_groups.csv with IID, group columns")
    ap.add_argument("--roster", required=True,
                    help="roster.txt (FID IID, whitespace-separated)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--chrs", default="1-22",
                    help="chromosome range (default 1-22)")
    args = ap.parse_args()

    lo, hi = (args.chrs.split("-") + [None])[:2]
    chroms = list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]

    roster = pd.read_csv(args.roster, sep=r"\s+", header=None,
                         names=["FID", "IID"], dtype=str)["IID"].to_numpy()
    anc = pd.read_csv(args.groups, dtype={"IID": str})
    anc = anc.set_index("IID").reindex(roster).reset_index()
    groups = anc["group"].to_numpy().astype(int)
    n = len(roster)
    group_ids = sorted(np.unique(groups).tolist())

    print(f"[privacy] n={n:,}  groups={len(group_ids)}  "
          f"draws={N_DRAWS}  pool={POOL_SIZE}  rows={N_ROWS}")
    print(f"[privacy] attacks: SO(n_cols={SO_N_COLS}, max_att={SO_MAX_ATTEMPTS}) "
          f"LK/INF(nfeat={NFEAT})  n_attacks={N_ATTACKS}")
    for g in group_ids:
        print(f"    group {g:2d}  n={int((groups == g).sum()):,}")

    jobs = [(di, u) for di in range(N_DRAWS) for u in group_ids]
    print(f"[privacy] {len(jobs)} jobs ({N_DRAWS} draws × {len(group_ids)} clusters)")

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                             initargs=(args.pruned_dir, groups, chroms)) as ex:
        futs = [ex.submit(_process, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"  [{k}/{len(jobs)}] draw {r['draw']} group {r['unit']} "
                  f"(n={r['n_rows_used']:,}): "
                  + ", ".join(f"{m[:4]}={r[m]:.1f}" for m in METRICS)
                  + f"  [{r['elapsed_s']}s]", flush=True)

    # Aggregate per-unit
    by_unit = {}
    for u in group_ids:
        sub = [r for r in rows if r["unit"] == u]
        e = {"n_unit": sub[0]["n_unit"], "n_draws": len(sub), "axes": {}}
        for m in METRICS:
            v = np.array([r[m] for r in sub if r[m] == r[m]], dtype=float)
            e["axes"][m] = {
                "mean": float(np.mean(v)) if v.size else float("nan"),
                "sd": float(np.std(v)) if v.size else float("nan"),
                "min": float(np.min(v)) if v.size else float("nan"),
                "n_finite": int(v.size)}
        by_unit[str(u)] = e

    # Grand summary: per-draw mean across clusters, then mean ± SD across draws
    grand = {}
    for m in METRICS:
        draw_means = []
        for di in range(N_DRAWS):
            vals = [r[m] for r in rows if r["draw"] == di and r[m] == r[m]]
            if vals:
                draw_means.append(float(np.mean(vals)))
        grand[m] = float(np.mean(draw_means)) if draw_means else float("nan")
        grand[f"{m}_sd"] = float(np.std(draw_means)) if draw_means else float("nan")

    summary = {
        "n": int(n), "n_draws": N_DRAWS, "pool_size": POOL_SIZE,
        "n_rows": N_ROWS, "n_attacks": N_ATTACKS, "nfeat": NFEAT,
        "groups": group_ids,
        "grand_mean": grand,
        "by_unit": by_unit,
        "per_job": sorted(rows, key=lambda r: (r["draw"], r["unit"]))}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== within-ancestry privacy (two-anchor %, mean±SD over draws) ===")
    print(f"{'unit':>5} {'n':>8} " + " ".join(f"{m[:4]:>16}" for m in METRICS))
    for u in group_ids:
        e = by_unit[str(u)]
        cells = []
        for m in METRICS:
            a = e["axes"][m]
            cells.append(f"{a['mean']:7.1f}±{a['sd']:5.1f}")
        print(f"{u:>5} {e['n_unit']:>8,} " + " ".join(f"{c:>16}" for c in cells))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

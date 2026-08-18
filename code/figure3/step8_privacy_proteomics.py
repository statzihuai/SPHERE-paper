#!/usr/bin/env python3
"""
Anonymeter privacy evaluation of SPHERE-rotated Olink proteomics.
10 independent runs with features and ColShuffle re-drawn each run.

Requires anonymeter==1.0.0 (Python < 3.12).

    python3 step8_privacy_proteomics.py \\
        --proteomics-csv $UKB_DIR/olink.csv \\
        --sphere-npy output/proteomics/prot_sphere.npy \\
        --out prot_privacy.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

# ── Unified privacy spec ──────────────────────────────────────────────────
N_ATTACKS = 500
NRUNS = 10
NFEAT = 20
SO_N_COLS = 3
MAX_ATTEMPTS = 20_000
N_ROWS = 2000
SEED = 42
META_COLS = {"IID", "age", "sex"}
METRICS = ("singling_out", "linkability", "inference")


def reseed(sd=SEED):
    """Reset all three of anonymeter's randomness sources."""
    import anonymeter.evaluators.singling_out_evaluator as _so
    import anonymeter.evaluators.linkability_evaluator as _lk
    _orig = np.random.default_rng
    np.random.seed(sd)
    _so.rng = np.random.default_rng(sd)
    _lk.np.random.default_rng = lambda *a, **k: _orig(sd)


def _run_one(args_tuple):
    """One run: 3-axis attack on real vs SPHERE."""
    from anonymeter.evaluators import (SinglingOutEvaluator,
                                       LinkabilityEvaluator,
                                       InferenceEvaluator)
    run, Z_real, Z_synth, seed = args_tuple
    n, p = Z_real.shape
    n_use = min(N_ROWS, n)
    rng = np.random.default_rng(seed + run)

    idx = rng.choice(n, size=n_use, replace=False)
    idx.sort()
    cols = [f"p{i}" for i in range(p)]
    ori = pd.DataFrame(Z_real[idx], columns=cols)
    syn = pd.DataFrame(Z_synth[idx], columns=cols)

    feats = rng.choice(cols, size=min(NFEAT, p), replace=False).tolist()
    half = len(feats) // 2
    secret = feats[0]
    aux = [c for c in cols if c != secret]

    results = {}
    for name, make_ev in [
        ("singling_out", lambda: SinglingOutEvaluator(
            ori, syn, n_attacks=N_ATTACKS, n_cols=SO_N_COLS,
            max_attempts=MAX_ATTEMPTS)),
        ("linkability", lambda: LinkabilityEvaluator(
            ori, syn, n_attacks=N_ATTACKS,
            aux_cols=(feats[:half], feats[half:]))),
        ("inference", lambda: InferenceEvaluator(
            ori, syn, aux_cols=aux[:19], secret=secret,
            n_attacks=N_ATTACKS)),
    ]:
        try:
            reseed(seed + run)
            ev = make_ev()
            ev.evaluate()
            results[name] = float(max(0.0, ev.risk().value))
        except Exception as e:
            print(f"  run={run} {name}: {e}")
            results[name] = None
    return run, results


def main():
    ap = argparse.ArgumentParser(
        description="Proteomics Anonymeter privacy evaluation")
    ap.add_argument("--proteomics-csv", required=True)
    ap.add_argument("--sphere-npy", required=True,
                    help="SPHERE-rotated protein matrix from prot_sphere.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.proteomics_csv)
    prot_cols = [c for c in df.columns if c not in META_COLS]
    Z_real = df[prot_cols].to_numpy(np.float64)
    Z_synth = np.load(args.sphere_npy)
    n, p = Z_real.shape
    assert Z_synth.shape == Z_real.shape, "shape mismatch"
    print(f"[privacy-prot] n={n:,}  p={p:,}  runs={NRUNS}")

    jobs = [(r, Z_real, Z_synth, SEED) for r in range(NRUNS)]
    per_run = []
    agg = {m: [] for m in METRICS}

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for run, res in pool.map(_run_one, jobs):
            per_run.append({"run": run, **res})
            for m in METRICS:
                if res[m] is not None:
                    agg[m].append(res[m])

    summary = {}
    for m in METRICS:
        vals = agg[m]
        summary[m] = float(np.mean(vals)) if vals else None
        summary[f"{m}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None

    output = {"summary": summary,
              "per_run": sorted(per_run, key=lambda r: r["run"])}
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  SO={summary['singling_out']:.1f}±{summary['singling_out_sd']:.1f}  "
          f"LK={summary['linkability']:.1f}±{summary['linkability_sd']:.1f}  "
          f"INF={summary['inference']:.1f}±{summary['inference_sd']:.1f}")
    print(f"[privacy-prot] -> {args.out}")


if __name__ == "__main__":
    main()

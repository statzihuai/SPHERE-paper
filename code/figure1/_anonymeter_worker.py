#!/usr/bin/env python3
"""
Anonymeter privacy-attack worker for Figure 1 panel b.

Runs in a separate interpreter (anonymeter requires Python <3.13) and is
launched by panel_b_privacy.py; not meant to be imported directly.

Args (command line):
    real_dir    data/benchmark/real
    synth_dir   data/benchmark/synthetic_full
    out_csv     output path
    methods     comma-separated method names; 'Real' is added automatically

Output columns: dataset, n, p, method, singling_out, linkability, inference

Requires anonymeter==1.0.0 (pinned for reproducibility).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from anonymeter.evaluators import (SinglingOutEvaluator,
                                   LinkabilityEvaluator,
                                   InferenceEvaluator)
import anonymeter.evaluators.singling_out_evaluator as so_mod
import anonymeter.evaluators.linkability_evaluator as lk_mod

SEED = 42
N_ATTACKS = 500      # attacks per evaluator
MAX_ROWS = 2000      # rows handed to each evaluator (attack cost is superlinear)
SO_COLS = 3          # predicate width for the singling-out attack
# anonymeter's default (1e7) lets the singling-out search run until it times out on
# large or integer-valued datasets, which silently returns NaN. 50k terminates.
SO_MAX_ATTEMPTS = 50_000
N_LINK_FEATS = 20    # auxiliary columns for linkability, split in half

_ORIG_RNG = np.random.default_rng


def reseed(seed=SEED):
    """Reset anonymeter's internal RNG sources for reproducibility."""
    np.random.seed(seed)                                            # pandas .sample
    so_mod.rng = np.random.default_rng(seed)                        # singling-out module
    lk_mod.np.random.default_rng = lambda *a, **k: _ORIG_RNG(seed)  # linkability module


def risk_of(name, make_evaluator):
    """Run an evaluator, returning its risk in [0, 1], or None on failure.

    `make_evaluator` is a thunk (not an instance) so that reseed() runs
    before the constructor, which itself consumes randomness.
    """
    try:
        reseed()
        ev = make_evaluator()
        ev.evaluate()
        return float(max(0.0, ev.risk().value))
    except Exception as e:
        print(f"    {name}: {type(e).__name__}: {e}", flush=True)
        return None


def main():
    real_dir = Path(sys.argv[1])
    synth_dir = Path(sys.argv[2])
    out_csv = Path(sys.argv[3])
    methods = sys.argv[4].split(",")

    rows = []
    for di, rfile in enumerate(sorted(real_dir.glob("*.npy"))):
        name = rfile.stem.split("__")[0]        # filename: {name}__n{n}_p{p}.npy
        Z = np.load(str(rfile)).astype(float)
        n, ncols = Z.shape
        n_atk = min(MAX_ROWS, n)
        cols = [f"c{i}" for i in range(ncols)]
        ori = pd.DataFrame(Z, columns=cols)

        # Per-dataset, seeded choice of the attacker's auxiliary columns and of the
        # secret to be inferred. Seeded so the attack is the same for every method.
        sel = np.random.default_rng(di)
        link_feats = sel.choice(cols, size=min(N_LINK_FEATS, ncols),
                                replace=False).tolist()
        half_lk = len(link_feats) // 2
        secret_col = sel.choice(cols)
        aux_inf = [c for c in cols if c != secret_col]

        print(f"  {name} (n={n}, p={ncols})", flush=True)
        # 'Real' = ori vs ori: the self-disclosure ceiling every method is scored against.
        for m in list(methods) + ["Real"]:
            if m == "Real":
                syn = ori.copy()
            else:
                sp = synth_dir / f"{name}__{m}.npy"
                if not sp.exists():
                    rows.append([name, n, ncols, m, None, None, None]); continue
                Zs = np.load(str(sp)).astype(float)
                if np.isnan(Zs).any():
                    rows.append([name, n, ncols, m, None, None, None]); continue
                syn = pd.DataFrame(Zs, columns=cols)

            o, s = ori.head(n_atk), syn.head(n_atk)
            so = risk_of("SinglingOut", lambda: SinglingOutEvaluator(
                o, s, n_attacks=N_ATTACKS, n_cols=SO_COLS,
                max_attempts=SO_MAX_ATTEMPTS))
            lk = risk_of("Linkability", lambda: LinkabilityEvaluator(
                o, s, n_attacks=N_ATTACKS,
                aux_cols=(link_feats[:half_lk], link_feats[half_lk:])))
            inf_ = risk_of("Inference", lambda: InferenceEvaluator(
                o, s, aux_cols=aux_inf, secret=secret_col, n_attacks=N_ATTACKS))
            rows.append([name, n, ncols, m, so, lk, inf_])

    df = pd.DataFrame(rows, columns=["dataset", "n", "p", "method",
                                     "singling_out", "linkability", "inference"])
    df.to_csv(str(out_csv), index=False)
    print(f"Saved {len(df)} rows -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()

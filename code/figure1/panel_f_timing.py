#!/usr/bin/env python3
"""
Figure 1 panel f — computing time.

Wall-clock cost of synthesising each dataset, run once per method under a
timer. The figure plots runtime relative to ColShuffle.

Note: timing is hardware-dependent and not expected to reproduce exactly;
only the ordering and order of magnitude are meaningful.

Usage:
    python3 code/figure1/panel_f_timing.py [--methods ...] [--skip-deep]
"""
import time

import numpy as np
import pandas as pd

import _common as C

TIMING_SEED = 42


def compute(datasets, methods):
    rows = []
    for d in datasets:
        Z = d["Z"]
        n = Z.shape[0]
        for m in methods:
            if C._skip(m, n, d["p"]):
                rows.append([d["name"], n, d["p"], m, np.nan]); continue
            t0 = time.perf_counter()
            try:
                np.random.seed(TIMING_SEED)
                C.METHODS[m](Z)                  # result discarded; we want the cost
            except Exception as e:
                print(f"  ! {d['name']}/{m}: {type(e).__name__}: {e} -> NaN", flush=True)
                rows.append([d["name"], n, d["p"], m, np.nan]); continue
            rows.append([d["name"], n, d["p"], m,
                         (time.perf_counter() - t0) * 1000])
    return pd.DataFrame(rows, columns=["dataset", "n", "p", "method", "runtime_ms"])


def main():
    args = C.build_parser("Figure 1 panel g — synthesiser runtime").parse_args()
    # stages=set(): this panel re-runs the synthesisers itself and needs no cache.
    datasets, methods = C.setup(args, stages=set(),
                                description="Figure 1 panel g — computing time")
    print("\n> Panel g (timing) — serial, wall-clock")
    df = compute(datasets, methods)
    C.save_csv(C.RES / "panel_f_timing.csv", df, methods)


if __name__ == "__main__":
    main()

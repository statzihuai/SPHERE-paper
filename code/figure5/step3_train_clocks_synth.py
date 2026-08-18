#!/usr/bin/env python3
"""
Refit the organ clocks on SPHERE-synthetic proteomics.

The alphas are taken from the real fit (step 1) rather than reselected.
Because train and test were rotated within-partition with Age co-rotated,
the frozen-alpha LASSO reproduces the real coefficients and the real
Pearson r bit-exact; MAE, being an L1 quantity, matches only approximately.

Writes synth_clock_perf.csv, synth_clock_weights.csv, synth_predictions.csv
and real_vs_synth_identity.csv (panel b).

    python3 step3_train_clocks_synth.py \\
        --synth-parquet output/organaging/synth_proteomics.parquet \\
        --frozen-alpha output/organaging/frozen_alpha.json \\
        --panels-json organ_panels.json \\
        --real-perf output/organaging/real_clock_perf.csv \\
        --real-weights output/organaging/real_clock_weights.csv \\
        --out-dir output/organaging
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Reuse the training logic from step 1
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step1_train_organ_clocks import fit_all_clocks  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Refit organ clocks on SPHERE synthetic with frozen alphas")
    ap.add_argument("--synth-parquet", required=True)
    ap.add_argument("--frozen-alpha", required=True)
    ap.add_argument("--panels-json", required=True)
    ap.add_argument("--real-perf", required=True)
    ap.add_argument("--real-weights", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    panels = json.loads(open(args.panels_json).read())
    synth = pd.read_parquet(args.synth_parquet)
    frozen = json.loads(open(args.frozen_alpha).read())

    print("[fit ] synthetic, frozen alphas from real fit:")
    res = fit_all_clocks(synth, panels, frozen_alphas=frozen)
    res["perf"].to_csv(f"{args.out_dir}/synth_clock_perf.csv", index=False)
    res["weights"].to_csv(f"{args.out_dir}/synth_clock_weights.csv", index=False)
    res["preds"].to_csv(f"{args.out_dir}/synth_predictions.csv", index=False)

    # ── Real vs synthetic identity comparison ──
    rp = pd.read_csv(args.real_perf)
    sp = res["perf"]
    perf = rp.merge(sp, on="clock", suffixes=("_real", "_synth"))
    perf["dr_test"] = perf["test_r_synth"] - perf["test_r_real"]
    perf["dr_train"] = perf["train_r_synth"] - perf["train_r_real"]

    rw = pd.read_csv(args.real_weights)
    sw = res["weights"]
    w = rw.merge(sw, on=["clock", "protein"], suffixes=("_real", "_synth"))
    coef_r = (w.groupby("clock")
              .apply(lambda g: np.corrcoef(g["weight_real"], g["weight_synth"])[0, 1]
                     if g["weight_real"].std() > 0 else np.nan)
              .rename("coef_r"))

    ident = perf[["clock", "test_r_real", "test_r_synth",
                  "dr_test", "dr_train"]].merge(coef_r, on="clock")
    ident.to_csv(f"{args.out_dir}/real_vs_synth_identity.csv", index=False)

    print("\n[identity] real vs synthetic:")
    print(ident.to_string(index=False))
    print(f"\n[write] -> {args.out_dir}")


if __name__ == "__main__":
    main()

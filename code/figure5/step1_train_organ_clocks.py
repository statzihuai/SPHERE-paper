#!/usr/bin/env python3
"""
Train organ-aging clocks on real UKB Olink proteomics (Oh et al. recipe).

Per organ: z-score the panel proteins using train-set statistics, run
LassoCV (5-fold) to score the alpha path, take the largest alpha reaching
95% of the best CV R2 (the most-regularised acceptable model), and refit a
plain Lasso at that alpha. The selected alphas are written out and reused
verbatim on the synthetic data in step 3, so CV-based alpha selection
cannot diverge between the two lanes.

Writes real_clock_perf.csv, real_clock_weights.csv, real_predictions.csv
and frozen_alpha.json.

    python3 step1_train_organ_clocks.py \\
        --proteomics-csv $DATA_DIR/olink_with_split.csv \\
        --panels-json organ_panels.json \\
        --out-dir output/organaging
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV

SEED = 0
CV_FOLDS = 5
PERF_FRACTION = 0.95   # paper: largest alpha reaching 95% of best CV R2

ORGANS = ["Brain", "Artery", "Liver", "Immune", "Intestine", "Lung",
          "Heart", "Pancreas", "Muscle", "Adipose", "Kidney"]


def select_alpha_95rule(lasso_cv, y_train):
    """Largest alpha whose 5-fold CV R2 >= 95% of best CV R2."""
    mean_mse = lasso_cv.mse_path_.mean(axis=1)
    var_y = float(np.var(y_train))
    r2_path = 1.0 - mean_mse / var_y
    thresh = PERF_FRACTION * r2_path.max()
    ok = r2_path >= thresh
    return float(lasso_cv.alphas_[ok].max())


def fit_all_clocks(df, panels, frozen_alphas=None):
    """Fit every organ clock; return weights, perf, predictions, alphas.

    frozen_alphas : dict clock->alpha to reuse. If None, alpha is selected on
                    this df's train split via LassoCV + the 95% rule.
    """
    is_tr = (df["split"] == "train").values
    y = df["age"].values.astype(float)
    y_tr, y_te = y[is_tr], y[~is_tr]

    weights_rows, perf_rows = [], []
    preds = pd.DataFrame({"iid": df["iid"].values,
                           "split": df["split"].values, "age": y})
    alphas_used = {}

    for clock, prots in panels.items():
        prots = [p for p in prots if p in df.columns]
        X = df[prots].values.astype(float)
        Xtr = X[is_tr]
        # z-score with train statistics only
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        Xs_tr = Xs[is_tr]

        if frozen_alphas is None:
            cv = LassoCV(cv=CV_FOLDS, random_state=SEED, n_jobs=-1,
                         max_iter=20000).fit(Xs_tr, y_tr)
            alpha = select_alpha_95rule(cv, y_tr)
        else:
            alpha = float(frozen_alphas[clock])
        alphas_used[clock] = alpha

        model = Lasso(alpha=alpha, max_iter=50000).fit(Xs_tr, y_tr)
        pred = model.predict(Xs)
        preds[f"{clock}_pred"] = pred

        pr_tr, pr_te = pred[is_tr], pred[~is_tr]
        train_r = float(np.corrcoef(pr_tr, y_tr)[0, 1])
        test_r = float(np.corrcoef(pr_te, y_te)[0, 1])
        perf_rows.append({
            "clock": clock, "alpha": alpha,
            "n_features": len(prots),
            "n_nonzero": int(np.sum(model.coef_ != 0)),
            "train_r": train_r, "test_r": test_r,
            "train_mae": float(np.mean(np.abs(pr_tr - y_tr))),
            "test_mae": float(np.mean(np.abs(pr_te - y_te))),
        })
        weights_rows.append({"clock": clock, "protein": "__intercept__",
                             "weight": float(model.intercept_)})
        for p, w in zip(prots, model.coef_):
            weights_rows.append({"clock": clock, "protein": p,
                                 "weight": float(w)})
        print(f"  {clock:12s} alpha={alpha:.5f}  nnz={int(np.sum(model.coef_!=0)):4d}  "
              f"test_r={test_r:.4f}  test_mae={perf_rows[-1]['test_mae']:.3f}")

    return {
        "weights": pd.DataFrame(weights_rows),
        "perf": pd.DataFrame(perf_rows),
        "preds": preds,
        "alphas": alphas_used,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Train organ-aging clocks on real proteomics (Oh et al. recipe)")
    ap.add_argument("--proteomics-csv", required=True)
    ap.add_argument("--panels-json", required=True,
                    help="JSON {organ: [protein_list]}")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    panels = json.loads(open(args.panels_json).read())
    df = pd.read_csv(args.proteomics_csv)
    df.columns = [str(c).lower() for c in df.columns]
    n_tr = int((df["split"] == "train").sum())
    n_te = int((df["split"] == "test").sum())
    print(f"[data] {df.shape}  train={n_tr}  test={n_te}")

    print("[fit ] LassoCV(5-fold) + 95% rule, per clock:")
    res = fit_all_clocks(df, panels)

    res["perf"].to_csv(f"{args.out_dir}/real_clock_perf.csv", index=False)
    res["weights"].to_csv(f"{args.out_dir}/real_clock_weights.csv", index=False)
    res["preds"].to_csv(f"{args.out_dir}/real_predictions.csv", index=False)
    with open(f"{args.out_dir}/frozen_alpha.json", "w") as f:
        json.dump(res["alphas"], f, indent=2)
    print(f"[write] -> {args.out_dir}")


if __name__ == "__main__":
    main()

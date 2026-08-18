#!/usr/bin/env python3
"""
Figure 1 panel e — machine-learning utility (TSTR).

Trains five regressors (EN, SVM, RF, GB, MLP) on synthetic data and
evaluates R² on held-out REAL rows. The split seed matches the one used
during synthesis, so no test-row leakage occurs.

Usage:
    python3 code/figure1/panel_e_utility.py [--methods ...] [--skip-deep] [--workers N]
"""
import time

import numpy as np
import pandas as pd

import _common as C

N_SPLITS = 5
MODEL_NAMES = ("EN", "SVM", "RF", "GB", "MLP")
# Models whose optimisation/regularisation is sensitive to the target scale.
Y_SCALE = {"EN", "SVM", "MLP"}


def _models():
    from sklearn.linear_model import ElasticNet
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    return {
        "EN":  lambda: ElasticNet(alpha=0.01, max_iter=2000),
        "SVM": lambda: SVR(kernel="rbf"),
        "RF":  lambda: RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=1),
        "GB":  lambda: GradientBoostingRegressor(n_estimators=50, random_state=0),
        "MLP": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000,
                                    early_stopping=True, n_iter_no_change=20,
                                    random_state=0),
    }


def _one(args):
    """Worker: all methods x all models for one (dataset, split)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    di, name, n, p, Z, s, methods = args
    models = _models()

    # Same seed as the synthesis worker -> test set = complement of the synth train.
    rng = np.random.default_rng(di * 1000 + s * 100 + 7)
    idx = rng.permutation(n)
    ntr = int(0.8 * n)
    Z_tr, Z_te = Z[idx[:ntr]], Z[idx[ntr:]]
    y_te, X_te = Z_te[:, 0], Z_te[:, 1:]

    # Both scalers fitted on the REAL train split, shared by every method.
    sc = StandardScaler().fit(Z_tr[:, 1:])
    sc_y = StandardScaler().fit(Z_tr[:, [0]])
    X_te_std = sc.transform(X_te)

    rows = []
    for m in methods:
        r2 = {k: np.nan for k in MODEL_NAMES}
        path = C.SYNTH_E / f"{name}__{m}__split{s}.npy"
        if C._skip(m, n, p) or not path.exists():
            rows.append([name, n, p, m, s] + [r2[k] for k in MODEL_NAMES]); continue
        Zs = np.load(path)
        if np.isnan(Zs).all():
            rows.append([name, n, p, m, s] + [r2[k] for k in MODEL_NAMES]); continue

        ys = Zs[:, 0]
        Xs_std = sc.transform(Zs[:, 1:])
        ys_std = sc_y.transform(Zs[:, [0]]).ravel()

        for mn in MODEL_NAMES:
            try:
                if mn in Y_SCALE:
                    mdl = models[mn]().fit(Xs_std, ys_std)
                    pred = sc_y.inverse_transform(
                        mdl.predict(X_te_std)[:, None]).ravel()
                else:
                    mdl = models[mn]().fit(Xs_std, ys)
                    pred = mdl.predict(X_te_std)
                r2[mn] = float(r2_score(y_te, pred))
            except Exception:
                pass                      # this model on this cell only -> NaN
        rows.append([name, n, p, m, s] + [r2[k] for k in MODEL_NAMES])
    return rows


def compute(datasets, methods, workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(di, d["name"], d["n"], d["p"], d["Z"], s, methods)
            for di, d in enumerate(datasets) for s in range(N_SPLITS)]
    all_rows = []
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed({ex.submit(_one, j): j for j in jobs}):
            all_rows.extend(fut.result())
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  [{done:3d}/{len(jobs)}]  {time.perf_counter()-t0:6.1f}s",
                      flush=True)
    return pd.DataFrame(all_rows, columns=["dataset", "n", "p", "method", "sim",
                                           *MODEL_NAMES])


def main():
    args = C.build_parser("Figure 1 panel f — TSTR machine-learning utility").parse_args()
    datasets, methods = C.setup(args, stages={"E"},
                                description="Figure 1 panel f — utility for machine learning (TSTR)")
    print("\n> Panel f (TSTR)")
    df = compute(datasets, methods, workers=args.workers)
    C.save_csv(C.RES / "panel_e_utility.csv", df, methods)


if __name__ == "__main__":
    main()

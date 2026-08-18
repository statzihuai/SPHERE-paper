"""
Shared layer for Figure 1 panel scripts.

Provides the synthesiser registry (SPHERE, DP, ROMM, …), raw-data loader,
and synthetic-data cache. Each panel metric lives in its own panel_*.py.

Panel -> file map:
    b  privacy     panel_b_privacy.py    -> panel_b_privacy.csv
    c  fidelity    panel_c_fidelity.py   -> panel_c_fidelity.csv
    d  inference   panel_d_inference.py  -> panel_d_inference.csv
    e  ML utility  panel_e_utility.py    -> panel_e_utility.csv
    f  runtime     panel_f_timing.py     -> panel_f_timing.csv

Usage (each script is standalone):
    python3 code/figure1/panel_c_fidelity.py [--methods ...] [--skip-deep] [--workers N]
"""
# Single-threaded BLAS. Must precede `import numpy`: the panel scripts fan out over
# a process pool, and an oversubscribed BLAS makes them slower, not faster.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # <repo>/code/figure1
CODE = HERE.parent                              # <repo>/code
ROOT = CODE.parent                              # <repo>
RES  = HERE                                      # panel CSVs written here
DATA = ROOT / "data" / "benchmark"

REAL_DIR   = DATA / "real"
SYNTH_FULL = DATA / "synthetic_full"            # whole dataset      (panels c, d)
SYNTH_D    = DATA / "synthetic_D"               # [y_sim | X] copies (panel e)
SYNTH_E    = DATA / "synthetic_E"               # 80% train splits   (panel f)


# ══════════════════════════════════════════════════════════════════════════════
# Synthesiser registry — the single source of truth for what each method does
# ══════════════════════════════════════════════════════════════════════════════
def _real(Z):
    return Z.astype(float, copy=True)


# SPHERE is available as a macOS desktop app (https://github.com/statzihuai/SPHERE)
# and a command-line tool (https://github.com/statzihuai/sphere-cli).
def _sphere1(Z):
    from sphere import sphere as _sphere
    return _sphere(Z)


def _sphere2(Z):
    from sphere import sphere as _sphere
    return _sphere(_sphere(Z))


def _colshuffle(Z):
    Zs = Z.astype(float, copy=True)
    for j in range(Zs.shape[1]):
        Zs[:, j] = np.random.permutation(Zs[:, j])
    return Zs


def _dp(Z, eps=17.14, delta=1e-5):
    Z = Z.astype(float)
    r = Z.max(0) - Z.min(0)
    sig = np.sqrt(2 * np.log(1.25 / delta)) * r / eps
    return Z + np.random.randn(*Z.shape) * sig


def _mixup_beta(Z, alpha=0.4):
    Z = Z.astype(float)
    lam = np.random.beta(alpha, alpha)
    return lam * Z + (1 - lam) * Z[np.random.permutation(len(Z))]


def _musig_est(Z):
    Z = Z.astype(float)
    return np.random.multivariate_normal(Z.mean(0), np.cov(Z.T), len(Z))


def _romm(Z):
    Z = Z.astype(float)
    Q, _ = np.linalg.qr(np.random.randn(len(Z), len(Z)))
    return Q @ Z


def _synthpop(Z):
    """Synthpop-style sequential CART synthesis (pure Python; no R dependency)."""
    from sklearn.tree import DecisionTreeRegressor
    Z = Z.astype(float)
    n, p = Z.shape
    Zs = np.zeros_like(Z)
    Zs[:, 0] = np.random.choice(Z[:, 0], n, replace=True)
    for j in range(1, p):
        tree = DecisionTreeRegressor(min_samples_leaf=5)
        tree.fit(Z[:, :j], Z[:, j])
        lr = tree.apply(Z[:, :j])
        ls = tree.apply(Zs[:, :j])
        for leaf in np.unique(ls):
            ms = ls == leaf
            mr = lr == leaf
            if not mr.any():
                mr = np.ones(n, bool)
            Zs[ms, j] = np.random.choice(Z[mr, j], ms.sum(), replace=True)
    return Zs


def _ctgan(Z, epochs=50):
    """CTGAN — Conditional Tabular GAN (Xu et al., NeurIPS 2019)."""
    from ctgan import CTGAN
    Z = Z.astype(float)
    n, p = Z.shape
    cols = [f"c{j}" for j in range(p)]
    df = pd.DataFrame(Z, columns=cols)
    mdl = CTGAN(epochs=epochs, verbose=False)
    mdl.fit(df)
    return mdl.sample(n)[cols].values.astype(float)


def _gaussian_copula(Z):
    """GaussianCopulaSynthesizer from SDV."""
    import warnings
    warnings.filterwarnings("ignore")
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import Metadata

    Z = Z.astype(float)
    n, p = Z.shape
    cols = ["y"] + [f"x{j}" for j in range(p - 1)]
    df = pd.DataFrame(Z, columns=cols)
    meta = Metadata.detect_from_dataframe(df, table_name="data")
    gc = GaussianCopulaSynthesizer(meta.tables["data"])
    gc.fit(df)
    sampled = gc.sample(n, output_file_path=None)
    return sampled[cols].values.astype(float)


def _tabddpm(Z, num_timesteps=100, steps=500, lr=2e-3, batch_size=512):
    """TabDDPM — Kotelnikov et al., NeurIPS 2023.

    Unconditional Gaussian diffusion on continuous Z. Loads the upstream tab-ddpm
    source from the directory named by TAB_DDPM_DIR (default /tmp/tab-ddpm).
    """
    import contextlib, io
    import torch
    sys.path.insert(0, os.environ.get("TAB_DDPM_DIR", "/tmp/tab-ddpm"))
    from tab_ddpm import GaussianMultinomialDiffusion
    from tab_ddpm.modules import MLPDiffusion

    torch.manual_seed(0)
    Z = Z.astype(float)
    n, d = Z.shape
    device = torch.device("cpu")
    mean, std = Z.mean(0), Z.std(0) + 1e-8
    Zn = ((Z - mean) / std).astype(np.float32)

    denoiser = MLPDiffusion(
        d_in=d, num_classes=0, is_y_cond=False,
        rtdl_params={"d_layers": [256, 256, 256], "dropout": 0.0},
        dim_t=128,
    ).to(device)
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array([0]),
        num_numerical_features=d,
        denoise_fn=denoiser,
        num_timesteps=num_timesteps,
        scheduler="cosine",
        device=device,
    ).to(device)

    X_tr = torch.tensor(Zn)
    y_dum = torch.zeros(n, dtype=torch.long)
    opt = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        idx = np.random.choice(n, min(batch_size, n), replace=False)
        lm, lg = diffusion.mixed_loss(
            X_tr[idx].to(device), {"y": y_dum[idx].to(device)})
        opt.zero_grad(); (lm + lg).backward(); opt.step()

    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        samples, _ = diffusion.sample(n, torch.ones(1))
    return samples.numpy() * std + mean


METHODS = {
    "Real":            _real,
    "SPHERE":          _sphere1,
    "SPHERE-x2":       _sphere2,
    "ColShuffle":      _colshuffle,
    "DP":              _dp,
    "Mixup-Beta(0.4)": _mixup_beta,
    "muSig-est":       _musig_est,
    "Synthpop":        _synthpop,
    "ROMM":            _romm,
    "CTGAN":           _ctgan,
    "TabDDPM":         _tabddpm,
    "GaussianCopula":  _gaussian_copula,
}
DEEP_METHODS = ("CTGAN", "TabDDPM")

# Size ceilings above which a method is not attempted (it would dominate the whole
# run). Rows above the ceiling are recorded as NaN in every panel, not dropped.
SLOW_N = {"ROMM": 2000, "CTGAN": 2000, "TabDDPM": 2000, "GaussianCopula": 2000}
SLOW_P = {"Synthpop": 500, "muSig-est": 500}


def _skip(method, n, p):
    return ((method in SLOW_N and n > SLOW_N[method]) or
            (method in SLOW_P and p > SLOW_P[method]))


# ══════════════════════════════════════════════════════════════════════════════
# Raw data
# ══════════════════════════════════════════════════════════════════════════════
def load_datasets():
    """The 33 benchmark datasets. Each .npy is [y | X]; see data/benchmark/README.md."""
    if not REAL_DIR.exists():
        raise FileNotFoundError(
            f"{REAL_DIR} not found. The raw benchmark data ships with this "
            f"repository under data/benchmark/ — see data/benchmark/README.md.")
    out = []
    for path in sorted(REAL_DIR.glob("*.npy")):
        Z = np.load(path)
        name = path.stem.split("__n")[0]
        out.append({"name": name, "Z": Z, "n": Z.shape[0], "p": Z.shape[1] - 1})
    return out


def generate_simY(di, Z):
    """Generate null/signal Y vectors for panel e (inference) on the fly.

    Deterministic: the same (di, Z) always produces the same output.
    Recipe documented in data/benchmark/README.md.

    Parameters
    ----------
    di : int       Alphabetical index of the dataset among the 33.
    Z  : ndarray   Shape (n, p+1) — column 0 is the original target, 1..p features.

    Returns
    -------
    dict with keys: X (n, p), y_null (n, 5), y_signal (n, 5), beta (p,), n_sig (int).
    """
    X_raw = Z[:, 1:]
    n, p = X_raw.shape

    # Log1p-transform non-negative columns (matches the cached npz recipe)
    X = X_raw.copy()
    for j in range(p):
        if np.all(X[:, j] >= 0):
            X[:, j] = np.log1p(X[:, j])

    n_sig = min(5, max(1, p // 4))
    beta = np.zeros(p)
    beta[:n_sig] = 0.27

    y_null = np.empty((n, 5))
    y_signal = np.empty((n, 5))
    for s in range(5):
        seed = di * 200 + s * 7 + 13
        rng = np.random.default_rng(seed)
        y_null[:, s] = rng.standard_normal(n)
        y_signal[:, s] = X @ beta + rng.standard_normal(n)

    return {"X": X, "y_null": y_null, "y_signal": y_signal,
            "beta": beta, "n_sig": n_sig}


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic-data cache — shared by panels c, d, e, f
# ══════════════════════════════════════════════════════════════════════════════
def _cached(path, fn, Z, seed):
    """Synthesise Z with fn under `seed`, memoised on disk.

    The seed is set with the legacy np.random global because several baselines
    (ColShuffle, DP, Mixup, muSig-est, ROMM, Synthpop) draw from it internally.
    """
    if path.exists():
        return np.load(path)
    np.random.seed(seed)
    try:
        Zs = fn(Z)
    except NotImplementedError:
        # SPHERE is not distributed with this release (see code/sphere.py).
        # Fail loudly: silently substituting NaN would drop every SPHERE row from
        # the panels and yield a figure that looks complete but is not.
        raise
    except Exception as e:
        # A baseline can legitimately fail on a given dataset (e.g. a deep model
        # on a wide input). Record it as NaN, but say so.
        print(f"  ! {path.stem}: {type(e).__name__}: {e} -> NaN", flush=True)
        Zs = np.full(Z.shape, np.nan)
    np.save(path, Zs)
    return Zs


def _gen_one(args):
    """Worker: synthesise the requested stages for one (dataset, method)."""
    di, name, Z, n, p, method, stages = args
    if _skip(method, n, p):
        return method, name, "skipped"
    fn = METHODS[method]

    # 'full' — the whole dataset, once. Panels c (privacy) and d (fidelity).
    if "full" in stages:
        _cached(SYNTH_FULL / f"{name}__{method}.npy", fn, Z, seed=di * 100 + 77)

    # 'D' — [y_sim | X] for each (simulation, null/signal). Panel d (inference).
    if "D" in stages:
        simY = generate_simY(di, Z)
        X_proc = simY["X"]
        for s in range(5):
            for k, y_key in enumerate(("y_null", "y_signal")):
                y = simY[y_key][:, s]
                Z_task = np.column_stack([y, X_proc])
                _cached(
                        SYNTH_D / f"{name}__{method}__sim{s}_y{k}.npy",
                        fn, Z_task,
                        seed=di * 200 + s * 100 + 2 + k * 1000)

    # 'E' — the 80% train split of each simulation. Panel f (train-on-synthetic).
    if "E" in stages:
        for s in range(5):
            rng = np.random.default_rng(di * 1000 + s * 100 + 7)
            idx = rng.permutation(n)
            Z_tr = Z[idx[: int(0.8 * n)]]
            _cached(
                SYNTH_E / f"{name}__{method}__split{s}.npy",
                fn, Z_tr,
                seed=di * 1000 + s * 100 + 7)
    return method, name, "done"


def generate_all(datasets, methods, workers, stages):
    """Populate the synthetic cache for `stages` (subset of {'full','D','E'}).

    Idempotent: _cached() returns early for anything already on disk, so a panel
    run that follows another only synthesises what is genuinely missing.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    for d in (SYNTH_FULL, SYNTH_D, SYNTH_E):
        d.mkdir(parents=True, exist_ok=True)
    jobs = [(di, d["name"], d["Z"], d["n"], d["p"], m, stages)
            for di, d in enumerate(datasets) for m in methods]
    print(f"\n> Synthesise stages {sorted(stages)} "
          f"({len(jobs)} jobs, {workers} workers)")
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed({ex.submit(_gen_one, j): j for j in jobs}):
            method, name, status = fut.result()
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  [{done:4d}/{len(jobs)}]  {time.perf_counter()-t0:6.1f}s"
                      f"  last: {name}/{method} ({status})", flush=True)
    print(f"  total: {time.perf_counter()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════
def save_csv(path, new_df, methods):
    """Write new_df to path, merging with any existing CSV.

    Rows whose 'method' is in `methods` are replaced; all other rows of the
    existing file are kept, so `--methods` reruns never destroy other methods.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old[~old["method"].isin(methods)], new_df],
                           ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(path, index=False)
    print(f"  saved {path.name}  ({len(merged)} rows, {len(methods)} method(s) updated)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI shared by every panel script
# ══════════════════════════════════════════════════════════════════════════════
def build_parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="subset of methods to (re)compute (default: all)")
    ap.add_argument("--skip-deep", action="store_true",
                    help=f"exclude {', '.join(DEEP_METHODS)}")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) // 2),
                    help="parallel workers")
    return ap


def resolve_methods(args):
    """Method list for this run. 'Real' is always included — it is the reference
    row every panel is scored against."""
    methods = list(METHODS) if args.methods is None else list(args.methods)
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown method(s): {unknown}\nknown: {list(METHODS)}")
    if args.skip_deep:
        methods = [m for m in methods if m not in DEEP_METHODS]
    if "Real" not in methods:
        methods = ["Real"] + methods
    return methods


def setup(args, stages, description):
    """Common preamble: load the datasets and fill the synthetic cache."""
    methods = resolve_methods(args)
    print("=" * 70)
    print(description)
    print("=" * 70)
    print(f"Methods ({len(methods)}): {methods}")
    datasets = load_datasets()
    print(f"Datasets: {len(datasets)}")
    if stages:
        generate_all(datasets, methods, workers=args.workers, stages=stages)
    return datasets, methods

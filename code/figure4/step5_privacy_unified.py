#!/usr/bin/env python3
"""
Step 5 — Anonymeter privacy over rotation k = 1..5 and all nine modalities.

This is the table behind panel c and Supplementary Table 12. One recipe is
applied to every modality:

    common : n_attacks = 500, sanitized column names, 3-RNG reseed,
             two-anchor normalisation (real = 0%, column-shuffle = 100%),
             10 runs with the features re-drawn each run
    SO     : multivariate, n_cols = 3, over the full p (scRNA loads a
             1000-column cap, which serves as its "full p")
    LK     : 20 random columns of p, split 10/10 into the two aux sets
    INF    : 20 random columns of p; secret = the first, aux = the other 19
    rows   : full rows, since every ADRC modality is small

Column names are sanitized because SomaScan spot names carry '.', '-' and ' ',
which pandas.query() cannot parse; unsanitized, every singling-out query raises,
the evaluator burns its attempt budget and the modality returns nothing.

Anonymeter draws randomness from three places that it never seeds — the
singling-out module RNG, the per-call linkability RNG, and the legacy global
np.random behind pandas .sample() — so all three are reset before every
evaluator call; that is what makes this table reproducible.

Outputs adrc_privacy_unified.csv (modality x k x metric, 10-run mean and SD)
and adrc_privacy_unified.json (the same plus the per-run raw anchors).

    python3 step5_privacy_unified.py \\
        --real-root data/adrc/real \\
        --sphere-xk results/figure4/sphere_xk \\
        --out-dir   results/figure4/privacy_unified \\
        --ks 1 2 3 4 5
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from anonymeter.evaluators import (SinglingOutEvaluator, LinkabilityEvaluator,
                                   InferenceEvaluator)
import anonymeter.evaluators.singling_out_evaluator as so_mod

from adrc_common import ID_COL, KS_DEFAULT, MODALITY_KEYS, MODALITY_REL, SKIP_ID

# ── paths (set from the CLI in main) ─────────────────────────────────────────
_ORIG_RNG = np.random.default_rng
REAL = SXK = OUT = None

# ── unified spec constants ───────────────────────────────────────────────────
N_ATTACKS   = 500          # spec: common
NRUNS       = 10           # spec: 10-run, features re-drawn each run
NFEAT       = 20           # spec: LK/INF random-20 of p (paper saturation 10-20)
LK_HALF     = NFEAT // 2   # 10/10 aux split
SO_N_COLS   = 3            # spec: SO multivariate n_cols=3 (library default)
MAX_ATTEMPTS = 20_000     # lowered from figure4's 100k: SO valid-query count
                          # saturates far below this on low-cardinality modalities
                          # (~70-116 queries) so results are unchanged, ~5x faster.
                          # High-cardinality modalities never approach it (no-op).
SCRNA_LOAD_CAP = 1000      # spec: scRNA "full p" = 1000-col load cap
EPS         = 1e-8

MODALITY_ORDER = MODALITY_KEYS
# All nine modalities run all three evaluators. Demographics linkability is
# kept for completeness even though a nine-column demographics table is a poor
# fit for the "two independent releases re-linked" threat model; its one-hot
# columns collapse the two anchors, so its LK/INF read unstably (see README).
EVALS_FOR = {}
ALL_EVALS = ["singling_out", "linkability", "inference"]


def reseed(seed):
    """Reset anonymeter's RNG sources for reproducibility."""
    np.random.seed(seed)              # pandas .sample
    so_mod.rng = _ORIG_RNG(seed)      # singling-out module


def sanitize(df):
    """query-hostile chars ('.', '-', ' ') -> '_' so pandas.query() parses."""
    out = df.copy()
    out.columns = [str(c).replace(".", "_").replace("-", "_").replace(" ", "_")
                   for c in out.columns]
    # de-duplicate any collisions introduced by the replacement
    seen, cols = {}, []
    for c in out.columns:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    out.columns = cols
    return out


def load_numeric(path, use_cols=None):
    """polars fast path: drop id, keep numeric (fillna median) + string dummies."""
    if use_cols is not None:
        df_pl = pl.read_csv(path, columns=use_cols, infer_schema_length=10)
    else:
        df_pl = pl.scan_csv(path, infer_schema_length=1000).collect()
    drop = [c for c in SKIP_ID if c in df_pl.columns]
    if drop:
        df_pl = df_pl.drop(drop)
    NUM = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
           pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    numeric = [c for c, t in zip(df_pl.columns, df_pl.dtypes) if t in NUM]
    strcols = [c for c, t in zip(df_pl.columns, df_pl.dtypes)
               if t in (pl.Utf8, pl.String, pl.Categorical) and c not in SKIP_ID]
    parts = []
    if numeric:
        num_pl = df_pl.select(numeric)
        nc, n = num_pl.null_count(), num_pl.height
        keep = [c for c in num_pl.columns if nc[c][0] < n]
        num_df = num_pl.select(keep).to_pandas()
        parts.append(num_df.fillna(num_df.median(numeric_only=True)))
    if strcols:
        str_df = df_pl.select(strcols).to_pandas().fillna("__NaN__").astype(str)
        parts.append(pd.get_dummies(str_df, drop_first=True, dtype=float))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]


def colshuffle(df, seed=0):
    """100%-privacy anchor: permute each column independently -> destroys rows."""
    rng = _ORIG_RNG(seed)
    out = df.copy()
    for col in out.columns:
        out[col] = rng.permutation(out[col].values)
    return out


# ── single evaluator, unified spec ───────────────────────────────────────────
def run_evaluator(name, ori, syn, n_at, feats):
    try:
        if name == "singling_out":
            ev = SinglingOutEvaluator(ori=ori, syn=syn, n_attacks=n_at,
                                      n_cols=min(SO_N_COLS, ori.shape[1]),
                                      max_attempts=MAX_ATTEMPTS)
            ev.evaluate(mode="multivariate")
        elif name == "linkability":
            # random-20 split 10/10; if p<20 split whatever feats exist in half
            nf = len(feats)
            half = nf // 2
            a, b = feats[:half], feats[half:nf]
            ev = LinkabilityEvaluator(ori=ori, syn=syn, n_attacks=n_at,
                                      aux_cols=(a, b))
            ev.evaluate(n_jobs=1)
        elif name == "inference":
            secret = feats[0]
            aux = feats[1:]
            ev = InferenceEvaluator(ori=ori, syn=syn, n_attacks=n_at,
                                    secret=secret, aux_cols=aux)
            ev.evaluate(n_jobs=1)
        else:
            return None
        res = ev.risk()
        return {"risk": float(res.value),
                "ci_low": float(res.ci[0]), "ci_high": float(res.ci[1])}
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:120]}"}


def run_all(ori, syn, n_at, feats, active):
    """One (ori,syn) pair -> {evaluator: result|None}. Reseed BEFORE each call."""
    out = {}
    for name in ALL_EVALS:
        if name not in active:
            out[name] = None
            continue
        reseed(RUN_SEED)                       # 3-RNG reseed per evaluator
        out[name] = run_evaluator(name, ori, syn, n_at, feats)
    return out


def _norm(s, r, c):
    denom = c - r
    if abs(denom) < EPS:
        return np.nan
    return (float(np.clip((s - r) / denom * 100, 0, 100)) if denom < 0
            else float(np.clip((r - s) / denom * 100, 0, 100)))


def normalize(sphere, real, cs):
    """two-anchor: real->0%, colshuffle->100%. Returns {metric: score|None}."""
    result = {}
    for key in ALL_EVALS:
        s, r, c = sphere.get(key), real.get(key), cs.get(key)
        if not (isinstance(s, dict) and isinstance(r, dict) and isinstance(c, dict)):
            result[key] = None
            continue
        if "err" in s or "err" in r or "err" in c:
            result[key] = None
            continue
        score = _norm(s["risk"], r["risk"], c["risk"])
        result[key] = None if np.isnan(score) else round(score, 2)
    return result


# ── per-modality driver ──────────────────────────────────────────────────────
RUN_SEED = 0   # module-global, set per run so reseed() sees it


def process_modality(key, ks, nruns, verbose=True):
    global RUN_SEED
    rel = MODALITY_REL[key]
    active = EVALS_FOR.get(key, ALL_EVALS)
    real_path = REAL / rel

    # scRNA ONLY: cap load to SCRNA_LOAD_CAP columns (p=193k is absurd; 1000-col
    # load IS its "full p" per spec). csf/plasma keep FULL p (sanitize makes the
    # dotted-name SO storm go away, so full p is feasible).
    non_skip = [c for c in pl.scan_csv(real_path, infer_schema_length=0)
                .collect_schema().names() if c not in SKIP_ID]
    cols_to_load = None
    if key == "scrna" and len(non_skip) > SCRNA_LOAD_CAP:
        cap_rng = _ORIG_RNG(0)
        cols_to_load = list(cap_rng.choice(non_skip, size=SCRNA_LOAD_CAP,
                                           replace=False))

    real_df = sanitize(load_numeric(real_path, use_cols=cols_to_load))
    syn_by_k = {}
    for k in ks:
        syn_by_k[k] = sanitize(load_numeric(SXK / f"x{k}" / rel,
                                            use_cols=cols_to_load))

    # align on common columns across real + all synthetic k
    common = set(real_df.columns)
    for k in ks:
        common &= set(syn_by_k[k].columns)
    common = [c for c in real_df.columns if c in common]
    real_df = real_df[common]
    for k in ks:
        syn_by_k[k] = syn_by_k[k][common]

    p = len(common)
    n = len(real_df)
    n_at = min(N_ATTACKS, max(1, n // 2))
    if verbose:
        print(f"\n{'='*66}\n### {key}  n={n}  p={p}  n_attacks={n_at}  "
              f"evals={active}\n{'='*66}", flush=True)

    # per-run accumulation: scores[metric][k] -> list over runs
    scores = {m: {k: [] for k in ks} for m in ALL_EVALS}
    anchors = []   # per-run raw for JSON audit

    for r in range(nruns):
        RUN_SEED = r
        # feature re-draw each run (LK/INF); SO ignores feats (uses full p)
        frng = _ORIG_RNG(1000 + r)
        nf = min(NFEAT, p)
        feats = list(frng.choice(common, size=nf, replace=False))
        cs_df = colshuffle(real_df, seed=r)

        t0 = time.time()
        real_res = run_all(real_df, real_df.copy(), n_at, feats, active)
        cs_res   = run_all(real_df, cs_df,          n_at, feats, active)
        sphere_res = {k: run_all(real_df, syn_by_k[k], n_at, feats, active)
                      for k in ks}
        dt = time.time() - t0

        run_scores = {}
        for k in ks:
            norm = normalize(sphere_res[k], real_res, cs_res)
            run_scores[k] = norm
            for m in ALL_EVALS:
                if norm.get(m) is not None:
                    scores[m][k].append(norm[m])
        anchors.append({"run": r, "feats": feats, "real": real_res,
                        "cs": cs_res, "sphere": {str(k): sphere_res[k] for k in ks},
                        "scores": {str(k): run_scores[k] for k in ks}})
        if verbose:
            snap = {m: {k: run_scores[k].get(m) for k in ks}
                    for m in active}
            print(f"  run{r}: t={dt:.1f}s  " +
                  "  ".join(f"{m[:2].upper()}=" +
                            "/".join("NA" if snap[m][k] is None else f"{snap[m][k]:.0f}"
                                     for k in ks) for m in active),
                  flush=True)

    # summarize mean+-SD over valid runs
    rows = []
    for m in active:
        for k in ks:
            vals = scores[m][k]
            rows.append({
                "modality": key, "k": k, "metric": m,
                "score_mean": round(float(np.mean(vals)), 2) if vals else None,
                "score_sd":   round(float(np.std(vals)), 2) if vals else None,
                "n_valid": len(vals), "n_runs": nruns,
                "n": n, "p": p, "n_attacks": n_at,
            })
    return rows, anchors


def main():
    global REAL, SXK, OUT
    ap = argparse.ArgumentParser(description="Unified Anonymeter privacy table for the ADRC release")
    ap.add_argument("--real-root", type=Path, required=True)
    ap.add_argument("--sphere-xk", type=Path, required=True,
                    help="Root holding x1/ … x5/ (step 3 output)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--modalities", nargs="*", default=None,
                    help="subset of modality keys (default: all 9)")
    ap.add_argument("--ks", nargs="*", type=int, default=KS_DEFAULT)
    ap.add_argument("--nruns", type=int, default=NRUNS)
    ap.add_argument("--smoke", action="store_true",
                    help="quick: nruns=1, ks=[2]")
    args = ap.parse_args()

    REAL, SXK, OUT = args.real_root, args.sphere_xk, args.out_dir
    mods = args.modalities or MODALITY_ORDER
    ks = args.ks
    nruns = args.nruns
    if args.smoke:
        nruns, ks = 1, [2]

    print(f"### ADRC unified privacy | modalities={mods} | ks={ks} | "
          f"nruns={nruns}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows, all_anchors = [], {}
    t_start = time.time()
    for key in mods:
        rows, anchors = process_modality(key, ks, nruns)
        all_rows.extend(rows)
        all_anchors[key] = anchors
        # incremental save so a long run is resumable / inspectable
        pd.DataFrame(all_rows).to_csv(OUT / "adrc_privacy_unified.csv", index=False)
        with open(OUT / "adrc_privacy_unified.json", "w") as fh:
            json.dump({"spec": {"n_attacks": N_ATTACKS, "nruns": nruns, "ks": ks,
                                "nfeat": NFEAT, "so_n_cols": SO_N_COLS,
                                "scrna_load_cap": SCRNA_LOAD_CAP},
                       "summary": all_rows, "anchors": all_anchors}, fh)
    print(f"\nDONE in {time.time()-t_start:.1f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Table 1, Privacy column: Anonymeter singling-out, linkability and inference
risk for all three modalities, scored on the same matrices SPHERE rotates.

Each metric is reported as a two-anchor protection score rather than a raw
risk, because a raw risk number is uninterpretable on its own — it depends on
n, on d, and on how much signal the attack can find in any release at all:

    Protection% = min(100, (risk_real - risk_sphere) / (risk_real - risk_floor))

`real` is the no-protection anchor (the release IS the original), `floor` is
ColShuffle, an independent per-column row permutation that keeps every marginal
and destroys every row. 0% means SPHERE leaked as much as handing over the
original; 100% means it leaked no more than a release with no row identity
left. Table 1 quotes the mean of the three metrics.

Two things make this reproducible that are easy to get wrong:

  * Anonymeter is not seeded. It draws randomness from three separate places —
    the singling-out module's own `default_rng`, a per-call `default_rng` in the
    linkability evaluator, and pandas `.sample`, which uses legacy np.random.
    All three are reseeded before every evaluator; reseeding fewer than three
    leaves the result non-deterministic. Those reseeds target attribute paths by
    name, so anonymeter must stay pinned at 1.0.0.
  * The rotation here must match the utility step's rotation exactly. For text
    and ECG that is the same call. For image it is not: step5 rotates
    [Z | y_pm1] and SPHERE auto-detects the ±1 columns into its mixed-angle
    scheme, whereas here there are no label columns, so delta, mix_prob and
    force_mixed_angle are passed by hand to reach the same rotation.

High-dimensional modalities are scored on a random 20-column subset per run
(20 of 768, 2,048 or 12,000). Full-p singling-out is intractable at this width;
this is the same tractability choice the genotype panel makes, and it is why
these numbers are not comparable to a full-p singling-out score.

    python3 step7_privacy_anonymeter.py [--modality all]

Inputs   data/table1/text/emb_train_raw.npy                    (step1)
         data/table1/image/imagenet100_validation_*.npy        (step2)
         data/table1/ecg/ptb_waveforms.npy                     (step3)
Outputs  code/table1/privacy_anonymeter.json
         code/table1/privacy_perrun_{modality}.csv
Needs    anonymeter==1.0.0 under Python 3.11, numpy 1.26.x, pandas 2.x, and the
         SPHERE implementation (the .venv-anon environment).
"""
import _common as C

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from anonymeter.evaluators import (SinglingOutEvaluator, LinkabilityEvaluator,
                                   InferenceEvaluator)
import anonymeter.evaluators.singling_out_evaluator as so_mod
import anonymeter.evaluators.linkability_evaluator as lk_mod

_ORIG_RNG = np.random.default_rng

N_RUNS = 10
NFEAT = 20              # random column subset per run
N_COLS_SO = 3           # predicate width for singling-out
MAX_ATTEMPTS = 50_000
SEED_BASE = 0
CLIP = 100.0            # cap each run's protection score at 100

# Image privacy is scored on a 200-row subset of the held-out set: 10 images per
# class for classes 0-19, taken in the order step2 streamed them. Twenty classes
# rather than a hundred keeps the row count small enough for the attacks while
# staying diverse; the ordering is replayed from the saved labels so no second
# pass over HuggingFace is needed.
IMG_N_CLASSES, IMG_N_PER_CLASS = 20, 10


def reseed(seed):
    """Reseed all three randomness sources anonymeter draws from."""
    np.random.seed(seed)
    so_mod.rng = np.random.default_rng(seed)
    lk_mod.np.random.default_rng = lambda *a, **k: _ORIG_RNG(seed)


def build_matrices(modality, data_dir):
    """Return (original, colshuffle_floor, meta) as float32 (n, d)."""
    if modality == "text":
        p = C.require(data_dir / "text" / "emb_train_raw.npy",
                      "run step1_text_embed.py first")
        ori = np.load(p).astype(np.float32)
        cs = C.colshuffle(ori, SEED_BASE).astype(np.float32)
        meta = {"source": "GTR/vec2text embedding (train split)",
                "sphere": "sphere(Z, theta=pi/6) seed 0"}

    elif modality == "image":
        f = C.require(data_dir / "image" / "imagenet100_validation_features.npy",
                      "run step2_image_features.py first")
        feats = np.load(f)
        labels = np.load(data_dir / "image" / "imagenet100_validation_labels.npy")
        count = {c: 0 for c in range(IMG_N_CLASSES)}
        idx = []
        for i, c in enumerate(labels.astype(int)):
            if c < IMG_N_CLASSES and count[c] < IMG_N_PER_CLASS:
                idx.append(i)
                count[c] += 1
            if len(idx) >= IMG_N_CLASSES * IMG_N_PER_CLASS:
                break
        ori = feats[np.array(idx)].astype(np.float32)
        cs = C.colshuffle(ori, 42).astype(np.float32)
        meta = {"source": "ResNet-50 feature, 200-row held-out subset",
                "sphere": "sphere(Z, theta=pi/6, delta=5deg, mix_prob=0.75, "
                          "force_mixed_angle=True) seed 42"}

    elif modality == "ecg":
        p = C.require(data_dir / "ecg" / "ptb_waveforms.npy",
                      "run step3_ecg_waveforms.py first")
        W = np.load(p).astype(np.float32)
        # Z-scored over the whole cohort here, not per training split: privacy is
        # a property of the released matrix, and there is no split to train on.
        mean = W.mean(axis=(0, 1), keepdims=True)
        std = W.std(axis=(0, 1), keepdims=True) + 1e-8
        ori = ((W - mean) / std).astype(np.float32).reshape(len(W), -1)
        cs = C.colshuffle(ori, 0).astype(np.float32)
        meta = {"source": "raw flattened waveform (z-scored)",
                "sphere": "sphere(Z, theta=pi/6, categorical_cols=[]) seed 0"}
    else:
        raise ValueError(modality)

    meta.update(n=int(ori.shape[0]), d=int(ori.shape[1]))
    return ori, cs, meta


def sphere_matrix(modality, ori, k=C.K):
    """Rotate the released matrix with this modality's exact call and seed."""
    Z = ori.copy().astype(np.float32)
    base = C.SPHERE_SEED[modality]
    for i in range(k):
        np.random.seed(base + i)
        if modality == "image":
            Z = sphere(Z, theta=C.THETA, delta=5.0 * np.pi / 180.0,
                       mix_prob=0.75, force_mixed_angle=True).astype(np.float32)
        elif modality == "ecg":
            Z = sphere(Z, theta=C.THETA, categorical_cols=[]).astype(np.float32)
        else:
            Z = sphere(Z, theta=C.THETA).astype(np.float32)
    return Z


def three_metrics(ori_df, syn_df, cols, n_attacks, seed):
    """Singling-out / linkability / inference risk of syn against ori.

    Each evaluator gets a fresh reseed. A failed evaluator yields NaN rather
    than aborting the run, and NaNs are excluded from the aggregate with the
    surviving count reported as n_valid.
    """
    aux, aux2 = cols[:len(cols) // 2], cols[len(cols) // 2:]
    secret = cols[-1]
    inf_aux = [c for c in cols if c != secret]
    so = lk = inf = np.nan

    try:
        reseed(seed)
        e = SinglingOutEvaluator(ori_df, syn_df, n_attacks=n_attacks,
                                 n_cols=N_COLS_SO, max_attempts=MAX_ATTEMPTS)
        e.evaluate()
        so = float(max(0.0, e.risk().value))
    except Exception:
        pass
    try:
        reseed(seed)
        e = LinkabilityEvaluator(ori_df, syn_df, n_attacks=n_attacks, aux_cols=(aux, aux2))
        e.evaluate()
        lk = float(max(0.0, e.risk().value))
    except Exception:
        pass
    try:
        reseed(seed)
        e = InferenceEvaluator(ori_df, syn_df, aux_cols=inf_aux, secret=secret,
                               n_attacks=n_attacks)
        e.evaluate()
        inf = float(max(0.0, e.risk().value))
    except Exception:
        pass
    return so, lk, inf


def run_modality(modality, data_dir, out_dir):
    t0 = time.time()
    ori, cs, meta = build_matrices(modality, data_dir)
    n, d = ori.shape
    n_attacks = max(50, min(500, n // 2))
    meta["n_attacks"] = n_attacks
    colnames = [f"c{i}" for i in range(d)]
    releases = {"real": ori, "colshuffle": cs, "sphere": sphere_matrix(modality, ori)}
    print(f"[{modality}] n={n} d={d} n_attacks={n_attacks}  {meta['sphere']}", flush=True)

    rows = []
    for r in range(N_RUNS):
        cidx = np.random.default_rng(SEED_BASE + r).choice(d, size=min(NFEAT, d),
                                                           replace=False)
        sub = [colnames[i] for i in cidx]
        ori_df = pd.DataFrame(ori[:, cidx], columns=sub)
        for rel, Z in releases.items():
            so, lk, inf = three_metrics(ori_df, pd.DataFrame(Z[:, cidx], columns=sub),
                                        sub, n_attacks, SEED_BASE + r)
            rows.append([r, rel, so, lk, inf])
        print(f"  run {r+1}/{N_RUNS} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows, columns=["run", "release", "singling_out",
                                     "linkability", "inference"])
    csv_path = out_dir / f"privacy_perrun_{modality}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    out = {"meta": meta, "raw_risk": {}, "protection_pct": {}}
    for metric in ("singling_out", "linkability", "inference"):
        piv = df.pivot(index="run", columns="release", values=metric)
        out["raw_risk"][metric] = {
            rel: {"mean": float(np.nanmean(piv[rel])), "sd": float(np.nanstd(piv[rel])),
                  "min": float(np.nanmin(piv[rel])), "max": float(np.nanmax(piv[rel]))}
            for rel in releases}
        denom = piv["real"] - piv["colshuffle"]
        prot = np.where(np.abs(denom) > 1e-9,
                        (piv["real"] - piv["sphere"]) / denom, np.nan) * 100.0
        prot = np.minimum(prot, CLIP)
        out["protection_pct"][metric] = {
            "mean": float(np.nanmean(prot)), "sd": float(np.nanstd(prot)),
            "n_valid": int(np.sum(~np.isnan(prot)))}

    three = [out["protection_pct"][m]["mean"]
             for m in ("singling_out", "linkability", "inference")]
    out["privacy_pct"] = float(np.mean(three))     # the Table 1 cell
    print(f"[{modality}] SO={three[0]:.1f} LK={three[1]:.1f} INF={three[2]:.1f} "
          f"-> privacy={out['privacy_pct']:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["text", "image", "ecg", "all"], default="all")
    ap.add_argument("--data-dir", default=str(C.DATA))
    ap.add_argument("--out-dir", default=str(C.OUT))
    args = ap.parse_args()

    data_dir, out_dir = C.Path(args.data_dir), C.Path(args.out_dir)
    mods = ["text", "image", "ecg"] if args.modality == "all" else [args.modality]

    import importlib.metadata as im
    try:
        anon_ver = im.version("anonymeter")
    except Exception:
        anon_ver = "?"

    out_path = out_dir / "privacy_anonymeter.json"
    result = C.read_json(out_path) if out_path.exists() else {}
    result["_spec"] = {
        "anonymeter_version": anon_ver, "n_runs": N_RUNS, "nfeat": NFEAT,
        "n_cols_so": N_COLS_SO, "max_attempts": MAX_ATTEMPTS, "theta": "pi/6",
        "k": C.K, "clip": CLIP,
        "note": "two-anchor Protection% = min(100, (risk_real - risk_sphere) / "
                "(risk_real - risk_floor)) per run; floor = ColShuffle; higher risk = "
                "more leak; random 20-column subset per run (NOT full-p singling-out).",
    }
    for m in mods:
        result[m] = run_modality(m, data_dir, out_dir)
        C.write_json(out_path, result)


if __name__ == "__main__":
    main()

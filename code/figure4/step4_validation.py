#!/usr/bin/env python3
"""
Step 4 — privacy and utility validation of the released rotation (panels a, d, e).

Writes adrc_validation.js, the metrics cache the renderer reads:

  * per-modality Anonymeter risk (singling-out / linkability / inference) for
    real, SPHERE x k and a column-shuffled control, over n_draws draws, each
    draw two-anchor normalised so real reads 0 and the shuffle reads 100;
  * AD-vs-control OLS run over ols_seeds random feature draws per modality,
    giving the beta and -log10 p correlations of panels d and e;
  * a 2-component PCA embedding of real vs synthetic per modality (panel a).

Anonymeter seeds three separate generators (its singling-out module RNG, the
per-call linkability RNG, and the legacy global RNG behind pandas .sample);
all three are pinned per draw, so a run with --seed is reproducible.

    python3 step4_validation.py \\
        --real-root data/adrc/real \\
        --sphere-xk results/figure4/sphere_xk \\
        --out-js    results/figure4/adrc_validation.js \\
        --ks 1 2 3 4 5 --seed 0
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from adrc_common import ID_COL, KS_DEFAULT, MODALITY_REL, RELEASE_K, SKIP_ID


def compute_validation(real_root: Path, sphere_xk_root: Path, out_js: Path,
                       ks, release_k=RELEASE_K, draw_size=20, n_draws=10,
                       attack_frac=0.5, ols_feats=50, ols_seeds=100,
                       pca_feats=500, seed=None) -> dict:
    """Recompute privacy + utility metrics and write them as adrc_validation.js.
    Requires anonymeter (privacy) and polars (fast wide-CSV reads)."""
    import polars as pl
    from anonymeter.evaluators import (SinglingOutEvaluator, LinkabilityEvaluator,
                                        InferenceEvaluator)

    KS         = sorted(ks)
    ATTACK_FRAC = max(0.01, min(0.99, attack_frac))
    EPS        = 1e-8
    vrng       = np.random.default_rng(seed)         # seed=None → fresh entropy

    def sphere_root(k):
        return sphere_xk_root / f"x{k}"

    MODALITIES = [
        dict(key="demographics", label="Demographics & Diagnosis",
             real=real_root / MODALITY_REL["demographics"],
             syn_rel=MODALITY_REL["demographics"], n_attacks=50, n_draws=10,
             evaluators=["singling_out", "inference"]),   # linkability N/A
        dict(key="cognitive", label="Cognitive Scores",
             real=real_root / MODALITY_REL["cognitive"], syn_rel=MODALITY_REL["cognitive"], n_attacks=50),
        dict(key="biomarkers", label="Blood Biomarkers",
             real=real_root / MODALITY_REL["biomarkers"], syn_rel=MODALITY_REL["biomarkers"], n_attacks=50),
        dict(key="imaging_amyloid", label="Amyloid PET + Thickness",
             real=real_root / MODALITY_REL["imaging_amy"], syn_rel=MODALITY_REL["imaging_amy"], n_attacks=50),
        dict(key="imaging_tau", label="Tau PET + Thickness",
             real=real_root / MODALITY_REL["imaging_tau"], syn_rel=MODALITY_REL["imaging_tau"], n_attacks=50),
        dict(key="wgs", label="Whole Genome Seq.",
             real=real_root / MODALITY_REL["wgs"], syn_rel=MODALITY_REL["wgs"], n_attacks=50),
        dict(key="scrna", label="PBMC scRNAseq",
             real=real_root / MODALITY_REL["scrna"], syn_rel=MODALITY_REL["scrna"], n_attacks=50),
        dict(key="csf", label="CSF Proteomics",
             real=real_root / MODALITY_REL["csf"], syn_rel=MODALITY_REL["csf"], n_attacks=50),
        dict(key="plasma", label="Plasma Proteomics",
             real=real_root / MODALITY_REL["plasma"], syn_rel=MODALITY_REL["plasma"], n_attacks=50),
    ]

    # ── CSV loading (polars fast path, pandas fallback) ──────────────────────────
    def load_numeric_fast(path, use_cols=None):
        try:
            if use_cols is not None:
                df_pl = pl.read_csv(path, columns=use_cols, infer_schema_length=10)
            else:
                df_pl = pl.scan_csv(path, infer_schema_length=1000).collect()
        except Exception as e:
            print(f"    polars load failed ({e}), falling back to pandas")
            df = pd.read_csv(path, usecols=(use_cols if use_cols else None), low_memory=False)
            df = df.drop(columns=[c for c in SKIP_ID if c in df.columns], errors="ignore")
            df = df.select_dtypes(include="number").dropna(axis=1, how="all")
            return df.fillna(df.median())
        drop_cols = [c for c in SKIP_ID if c in df_pl.columns]
        if drop_cols:
            df_pl = df_pl.drop(drop_cols)
        NUM = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
               pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        numeric_cols = [c for c, t in zip(df_pl.columns, df_pl.dtypes) if t in NUM]
        string_cols  = [c for c, t in zip(df_pl.columns, df_pl.dtypes)
                        if t in (pl.Utf8, pl.String, pl.Categorical) and c not in SKIP_ID]
        if not numeric_cols and not string_cols:
            return pd.DataFrame()
        parts = []
        if numeric_cols:
            num_pl = df_pl.select(numeric_cols)
            null_counts, n_rows = num_pl.null_count(), num_pl.height
            keep = [c for c in num_pl.columns if null_counts[c][0] < n_rows]
            num_df = num_pl.select(keep).to_pandas()
            num_df = num_df.fillna(num_df.median(numeric_only=True))
            parts.append(num_df)
        if string_cols:
            str_df = df_pl.select(string_cols).to_pandas().fillna("__NaN__").astype(str)
            parts.append(pd.get_dummies(str_df, drop_first=True, dtype=float))
        return pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]

    def get_cols(path):
        return set(pl.scan_csv(path, infer_schema_length=0).collect_schema().names())

    def _add_binary_orth_cols(df):
        for ocol in list(df.columns):
            if (ocol in SKIP_ID or pd.api.types.is_numeric_dtype(df[ocol])
                    or pd.api.types.is_bool_dtype(df[ocol])):
                continue
            pres = df[ocol].notna().values
            strv = df[ocol].astype(str).values
            for v in sorted(df[ocol].dropna().unique(), key=str):
                df[f"{ocol}_orth_{v}"] = np.where(
                    pres, np.where(strv == str(v), 1.0, -1.0), np.nan).astype(np.float32)
        return df

    def sample_cols(cols, nn, rng=None):
        if len(cols) <= nn:
            return cols
        rng = rng or vrng
        return [cols[i] for i in sorted(rng.choice(len(cols), size=nn, replace=False))]

    def colshuffle(df, seed=0):
        rng = np.random.default_rng(seed)
        out = df.copy()
        for col in out.columns:
            out[col] = rng.permutation(out[col].values)
        return out

    def embed_pca(real, syn):
        comb = StandardScaler().fit_transform(np.vstack([real, syn]))
        emb  = PCA(n_components=2).fit_transform(comb)
        return emb[:len(real)], emb[len(real):]

    def fmt_pca(re_, sy_):
        return ([{"x": round(float(x), 3), "y": round(float(y), 3), "s": 0} for x, y in re_]
                + [{"x": round(float(x), 3), "y": round(float(y), 3), "s": 1} for x, y in sy_])

    def _run_one_evaluator(name, ori, syn, n_at, cols):
        try:
            if name == "singling_out":
                ev = SinglingOutEvaluator(ori=ori, syn=syn, n_attacks=n_at,
                                          n_cols=min(3, ori.shape[1]), max_attempts=100_000)
                ev.evaluate(mode="multivariate")
            elif name == "linkability":
                nh = len(cols) // 2
                ev = LinkabilityEvaluator(ori=ori, syn=syn, n_attacks=n_at,
                                          aux_cols=(cols[:nh], cols[nh:]))
                ev.evaluate(n_jobs=1)
            elif name == "inference":
                secret = cols[0]
                aux = [c for c in cols if c != secret][:min(5, len(cols) - 1)]
                ev = InferenceEvaluator(ori=ori, syn=syn, n_attacks=n_at, secret=secret, aux_cols=aux)
                ev.evaluate(n_jobs=1)
            else:
                return None
            res = ev.risk()
            return {"risk": round(float(res.value), 6),
                    "ci_low": round(float(res.ci[0]), 6),
                    "ci_high": round(float(res.ci[1]), 6)}
        except Exception as e:
            print(f"    {name} failed: {e}")
            return None

    def run_anonymeter_raw(ori, syn, n_attacks, evaluators=None):
        ori, syn = ori.reset_index(drop=True), syn.reset_index(drop=True)
        cols = list(ori.columns)
        n_at = min(n_attacks, max(1, min(len(ori), len(syn)) // 2))
        active = evaluators if evaluators is not None else ["singling_out", "linkability", "inference"]
        return {name: (_run_one_evaluator(name, ori, syn, n_at, cols) if name in active else None)
                for name in ("singling_out", "linkability", "inference")}

    def average_raw(results_list):
        out = {}
        for name in ("singling_out", "linkability", "inference"):
            vals = [r[name] for r in results_list if r.get(name) is not None]
            out[name] = (None if not vals else
                         {kk: round(float(np.mean([v[kk] for v in vals])), 6)
                          for kk in ("risk", "ci_low", "ci_high")})
        return out

    def normalize_score(sphere_raw, real_raw, cs_raw):
        if sphere_raw is None or real_raw is None or cs_raw is None:
            return None
        def _norm(s, r, c):
            denom = c - r
            if abs(denom) < EPS:
                return np.nan
            return (float(np.clip((s - r) / denom * 100, 0, 100)) if denom < 0
                    else float(np.clip((r - s) / denom * 100, 0, 100)))
        result = {}
        for key in ("singling_out", "linkability", "inference"):
            s_d, r_d, c_d = sphere_raw.get(key), real_raw.get(key), cs_raw.get(key)
            if s_d is None or r_d is None or c_d is None:
                result[key] = None
                continue
            score = _norm(s_d["risk"], r_d["risk"], c_d["risk"])
            ci_lo = _norm(s_d["ci_high"], r_d["ci_low"], c_d["ci_high"])
            ci_hi = _norm(s_d["ci_low"], r_d["ci_high"], c_d["ci_low"])
            result[key] = {
                "score":   round(score, 1) if not np.isnan(score) else None,
                "ci_low":  round(float(np.clip(min(ci_lo, ci_hi), 0, 100)), 1) if not np.isnan(ci_lo) else None,
                "ci_high": round(float(np.clip(max(ci_lo, ci_hi), 0, 100)), 1) if not np.isnan(ci_hi) else None,
                "sphere_risk": round(s_d["risk"] * 100, 1),
                "real_risk":   round(r_d["risk"] * 100, 1),
                "cs_risk":     round(c_d["risk"] * 100, 1),
            }
        return result

    # ── Phase 1: load real data + anchors ────────────────────────────────────────
    print("=" * 60 + "\nPhase 1: Loading real data + computing anchors")
    modality_cache = {}
    for m in MODALITIES:
        key, real_path = m["key"], m["real"]
        if not real_path.exists():
            print(f"  {key}: SKIP — real data not found: {real_path}")
            continue
        print(f"\n  {key}: loading real …", end=" ", flush=True)
        non_skip = [c for c in get_cols(real_path) if c not in SKIP_ID]
        cols_to_load = (list(vrng.choice(non_skip, size=pca_feats * 2, replace=False))
                        if len(non_skip) > pca_feats * 2 else None)
        real_df = load_numeric_fast(real_path, use_cols=cols_to_load)
        print(f"{real_df.shape}")

        # Augment real_df with ±1 orth columns (mirror build_joint_matrix), detecting
        # categorical columns from header + first row only (never load all 193k cols).
        import csv as _csv
        cat_in_file = []
        try:
            with open(real_path, newline="", encoding="utf-8") as f:
                rdr = _csv.reader(f); header = next(rdr, []); row1 = next(rdr, [])
            for col, val in zip(header, row1):
                if col in SKIP_ID:
                    continue
                try:
                    float(val)
                except (ValueError, TypeError):
                    cat_in_file.append(col)
        except Exception:
            pass
        if cat_in_file:
            raw = pd.read_csv(real_path, low_memory=False)
            for ocol in list(raw.columns):
                if (ocol in SKIP_ID or pd.api.types.is_numeric_dtype(raw[ocol])
                        or pd.api.types.is_bool_dtype(raw[ocol])):
                    continue
                pres = raw[ocol].notna().values
                if len(pres) != len(real_df):
                    continue
                for v in sorted(raw[ocol].dropna().unique(), key=str):
                    oname = f"{ocol}_orth_{v}"
                    if oname not in real_df.columns:
                        real_df[oname] = np.where(
                            pres, np.where(raw[ocol].astype(str).values == str(v), 1.0, -1.0),
                            np.nan).astype(np.float32)
            del raw

        all_cols  = list(real_df.columns)
        eval_cols = sample_cols(all_cols, pca_feats)
        if key == "demographics":
            orth_and_age = [c for c in all_cols if "_orth_" in c or c == "age_at_visit"]
            if orth_and_age:
                eval_cols = orth_and_age
                print(f"    demographics: evaluating {len(orth_and_age)} cols (age + ±1 orth)")

        draw_size_m = min(draw_size, len(eval_cols))
        n_draws_m   = m.get("n_draws", n_draws)
        draw_cols   = [list(vrng.choice(eval_cols, size=draw_size_m, replace=False))
                       for _ in range(n_draws_m)]

        print(f"    Computing anchors ({n_draws_m} draws × {draw_size_m} features) …")
        real_draws, cs_draws = [], []
        for di, dc in enumerate(draw_cols):
            real_sub = real_df[dc].reset_index(drop=True)
            cs_sub   = colshuffle(real_sub, seed=di)
            n_at     = min(m["n_attacks"], max(1, int(len(real_sub) * ATTACK_FRAC)))
            ev = m.get("evaluators")
            np.random.seed(di); real_draws.append(run_anonymeter_raw(real_sub, real_sub.copy(), n_at, ev))
            np.random.seed(di); cs_draws.append(run_anonymeter_raw(real_sub, cs_sub, n_at, ev))
        modality_cache[key] = dict(m=m, real_df=real_df, eval_cols=eval_cols,
                                   draw_cols=draw_cols, draw_size=draw_size_m,
                                   cols_to_load=cols_to_load,
                                   real_raw=average_raw(real_draws), cs_raw=average_raw(cs_draws))

    # ── Phase 2: per-modality progression over k ─────────────────────────────────
    print("\n" + "=" * 60 + f"\nPhase 2: Per-modality progression (k = {KS})")
    progression_by_key = {key: [] for key in modality_cache}
    pca_data = {}
    for k in KS:
        sroot = sphere_root(k)
        print(f"\nk = {k}  ({sroot})")
        for key, cache in modality_cache.items():
            m = cache["m"]
            syn_path = sroot / m["syn_rel"]
            if not syn_path.exists():
                print(f"  {key}: SKIP — {syn_path} not found")
                continue
            print(f"\n  {key} (k={k}) …", end=" ", flush=True)
            syn_df = load_numeric_fast(syn_path, use_cols=cache["cols_to_load"])
            real_df = cache["real_df"]
            draw_cols, draw_size_m, eval_cols = cache["draw_cols"], cache["draw_size"], cache["eval_cols"]
            cols = sorted(set(real_df.columns) & set(syn_df.columns))
            if len(cols) < 2:
                print("fewer than 2 common cols — SKIP")
                continue
            avail = set(cols)
            eff_draw_cols = [
                [c for c in dc if c in avail] or
                list(np.random.default_rng(i).choice(cols, size=min(draw_size_m, len(cols)), replace=False))
                for i, dc in enumerate(draw_cols)]
            print(f"{syn_df.shape}  common={len(cols)}")

            if k == release_k:
                pca_cols = sample_cols([c for c in eval_cols if c in avail], pca_feats)
                try:
                    re_, sy_ = embed_pca(real_df[pca_cols].values, syn_df[pca_cols].values)
                    pca_data[key] = fmt_pca(re_, sy_)
                except Exception as e:
                    print(f"    PCA FAILED: {e}")
                    pca_data[key] = []

            sphere_draws = []
            for di, dc in enumerate(eff_draw_cols):
                real_sub = real_df[dc].reset_index(drop=True)
                syn_sub  = syn_df[dc].reset_index(drop=True)
                n_at = min(m["n_attacks"], max(1, int(min(len(real_sub), len(syn_sub)) * ATTACK_FRAC)))
                np.random.seed(di)
                sphere_draws.append(run_anonymeter_raw(real_sub, syn_sub, n_at, m.get("evaluators")))
            metrics_k = normalize_score(average_raw(sphere_draws), cache["real_raw"], cache["cs_raw"])
            progression_by_key[key].append({"k": k, "metrics": metrics_k})
            del syn_df
            gc.collect()

    # ── Phase 3: OLS utility at the release point (y = AD orth ~ each feature) ────
    print("\n" + "=" * 60 + "\nPhase 3: OLS utility (β + p-value preservation)")

    def _find_ad_orth_col(df_):
        return (next((c for c in df_.columns if c.startswith("diagnosis_consensus_orth_")
                      and "probable" in c.lower() and "alzheimer" in c.lower()), None)
                or next((c for c in df_.columns if c.startswith("diagnosis_consensus_orth_")
                         and "alzheimer" in c.lower()), None))

    def _ols_betas(X, y):
        from scipy import stats as _st
        betas, ses, pvals = [], [], []
        for j in range(X.shape[1]):
            x = X[:, j]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 4:
                betas.append(np.nan); ses.append(np.nan); pvals.append(np.nan); continue
            xm, ym = x[mask], y[mask]
            Xd = np.column_stack([np.ones(mask.sum()), xm])
            b, *_ = np.linalg.lstsq(Xd, ym, rcond=None)
            res = ym - Xd @ b
            dof = mask.sum() - 2
            s2 = (res @ res) / dof if dof > 0 else np.nan
            se = np.sqrt(max(s2 / max(np.sum((xm - xm.mean()) ** 2), 1e-30), 0)) if np.isfinite(s2) else np.nan
            t  = b[1] / se if (np.isfinite(se) and se > 0) else np.nan
            pv = 2 * _st.t.sf(abs(t), df=dof) if np.isfinite(t) else np.nan
            betas.append(b[1]); ses.append(se); pvals.append(pv)
        return np.array(betas), np.array(ses), np.array(pvals)

    from scipy.stats import pearsonr as _pearsonr
    utility_by_key = {}
    _dem = next((m for m in MODALITIES if m["key"] == "demographics"), None)
    _dem_real_df = _dem_syn_df = None
    _DEM_AD_REAL = _DEM_AD_SYN = None
    if _dem:
        _drp, _dsp = _dem["real"], sphere_root(release_k) / _dem["syn_rel"]
        if _drp.exists() and _dsp.exists():
            _dem_real_df = _add_binary_orth_cols(pd.read_csv(_drp, low_memory=False, dtype={ID_COL: str}))
            _dem_syn_df  = pd.read_csv(_dsp, low_memory=False, dtype={ID_COL: str})
            _DEM_AD_REAL = _find_ad_orth_col(_dem_real_df)
            _DEM_AD_SYN  = _find_ad_orth_col(_dem_syn_df)

    for m in MODALITIES:
        key = m["key"]
        real_path, syn_path = m["real"], sphere_root(release_k) / m["syn_rel"]
        if not (real_path.exists() and syn_path.exists()):
            print(f"  {key}: SKIP — file missing"); continue
        real_df = _add_binary_orth_cols(pd.read_csv(real_path, low_memory=False, dtype={ID_COL: str}))
        syn_df  = pd.read_csv(syn_path, low_memory=False, dtype={ID_COL: str})

        oc_real, oc_syn = _find_ad_orth_col(real_df), _find_ad_orth_col(syn_df)
        if oc_real is None and _dem_real_df is not None and _DEM_AD_REAL:
            real_df = real_df.merge(_dem_real_df[[ID_COL, _DEM_AD_REAL]].drop_duplicates(ID_COL),
                                    on=ID_COL, how="left"); oc_real = _DEM_AD_REAL
        if oc_syn is None and _dem_syn_df is not None and _DEM_AD_SYN:
            syn_df = syn_df.merge(_dem_syn_df[[ID_COL, _DEM_AD_SYN]].drop_duplicates(ID_COL),
                                  on=ID_COL, how="left"); oc_syn = _DEM_AD_SYN
        if oc_real is None or oc_syn is None:
            print(f"  {key}: SKIP — no AD orth column"); continue

        real_df["_y"] = real_df[oc_real].astype(float)
        syn_df["_y"]  = syn_df[oc_syn].astype(float)
        protected = SKIP_ID | {ID_COL, "_y", oc_real}
        feat_cols = [c for c in real_df.columns
                     if c not in protected and "_orth_" not in c
                     and pd.api.types.is_numeric_dtype(real_df[c]) and c in syn_df.columns]
        n_real, n_syn = len(real_df), len(syn_df)
        n_feat = min(ols_feats, len(feat_cols))
        if n_feat < 2 or n_real < 10 or n_syn < 10:
            print(f"  {key}: SKIP — too few samples/features"); continue

        y_real, y_syn = real_df["_y"].values.astype(float), syn_df["_y"].values.astype(float)
        bc_list, pc_list = [], []
        for s in range(ols_seeds):
            cols_draw = list(np.random.default_rng(s).choice(feat_cols,
                             size=min(ols_feats, len(feat_cols)), replace=False))
            b_r, _, p_r = _ols_betas(real_df[cols_draw].values.astype(float), y_real)
            b_s, _, p_s = _ols_betas(syn_df[cols_draw].values.astype(float), y_syn)
            ok = np.isfinite(b_r) & np.isfinite(b_s) & np.isfinite(p_r) & np.isfinite(p_s)
            if ok.sum() < 2:
                continue
            br, bs = b_r[ok], b_s[ok]
            pr, ps = np.clip(p_r[ok], 1e-300, 1), np.clip(p_s[ok], 1e-300, 1)
            _bc = _pearsonr(br, bs)[0] if len(br) > 1 else np.nan
            _pc = _pearsonr(-np.log10(pr), -np.log10(ps))[0] if len(pr) > 1 else np.nan
            if np.isfinite(_bc): bc_list.append(_bc)
            if np.isfinite(_pc): pc_list.append(_pc)
        if not bc_list:
            print(f"  {key}: SKIP — too few finite results"); continue
        bc, pc = float(np.mean(bc_list)), float(np.mean(pc_list)) if pc_list else np.nan
        utility_by_key[key] = {
            "beta_corr": round(max(0.0, bc) * 100 if np.isfinite(bc) else 0.0, 1),
            "pval_corr": round(max(0.0, pc) * 100 if np.isfinite(pc) else 0.0, 1),
            "n_features": n_feat, "n_real": n_real, "n_syn": n_syn}
        print(f"  {key}: beta_corr={utility_by_key[key]['beta_corr']}%  "
              f"pval_corr={utility_by_key[key]['pval_corr']}%")

    # ── Phase 4: assemble + write ────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\nPhase 4: Assembling adrc_validation.js")
    validation = {}
    for key, cache in modality_cache.items():
        prog = progression_by_key.get(key, [])
        latest = (next((p["metrics"] for p in prog if p["k"] == release_k and p["metrics"]), None)
                  or next((p["metrics"] for p in prog if p["metrics"]), None))
        avg_privacy = None
        if latest:
            scores = [latest[ek]["score"] for ek in ("singling_out", "linkability", "inference")
                      if latest.get(ek) and latest[ek].get("score") is not None]
            avg_privacy = round(sum(scores) / len(scores), 1) if scores else None
        validation[key] = {
            "label": cache["m"]["label"], "n_real": int(len(cache["real_df"])),
            "n_syn": int(len(cache["real_df"])), "n_cols": cache["draw_size"],
            "umap": pca_data.get(key, []), "metrics": latest,
            "avg_privacy_score": avg_privacy, "progression": prog,
            "utility": utility_by_key.get(key, None)}

    out_js.parent.mkdir(parents=True, exist_ok=True)
    with open(out_js, "w", encoding="utf-8") as f:
        f.write("window.ADRC_VALIDATION=")
        f.write(json.dumps(validation, separators=(",", ":"), ensure_ascii=False))
        f.write(";")
    print(f"Wrote {out_js}  ({out_js.stat().st_size // 1024} KB)")

def main():
    ap = argparse.ArgumentParser(description="Recompute the ADRC privacy + utility metrics cache")
    ap.add_argument("--real-root", type=Path, required=True)
    ap.add_argument("--sphere-xk", type=Path, required=True,
                    help="Root holding x1/ … x5/ (step 3 output)")
    ap.add_argument("--out-js", type=Path, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=KS_DEFAULT)
    ap.add_argument("--release-k", type=int, default=RELEASE_K)
    ap.add_argument("--n-draws", type=int, default=10, help="Anonymeter draws per modality")
    ap.add_argument("--ols-seeds", type=int, default=100, help="Feature draws per modality for panels d/e")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    compute_validation(args.real_root, args.sphere_xk, args.out_js, sorted(set(args.ks)),
                       release_k=args.release_k, n_draws=args.n_draws,
                       ols_seeds=args.ols_seeds, seed=args.seed)


if __name__ == "__main__":
    main()

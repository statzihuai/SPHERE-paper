#!/usr/bin/env python3
"""
Organ age gap -> incident disease and all-cause mortality (Oh et al. recipe).

  1. Organ age gap: dage_organ = z-scored residual of OLS(predicted ~ age),
     taken from the real clock predictions of step 1.
  2. Merge incident-disease time-to-event, all-cause mortality and
     covariates (Sex, APOE4, APOE2, entry ages).
  3. Rotate the full analysis matrix jointly with SPHERE: event columns via
     encode-decode so their marginal counts stay exact, time and age columns
     clipped at zero, everything else continuous.
  4. Run the identical delayed-entry Cox model (age timescale, adjusted for
     Sex) on the real and the synthetic matrix.

The Cox partial likelihood is not orthogonally invariant, so the two lanes
give concordant hazard ratios rather than identical ones.

Input schemas:
    --predictions-csv  real_predictions.csv from step 1
    --tte-csv          IID, age_at_baseline, last_known_age,
                       {disease}_event, {disease}_time
    --mortality-csv    IID, baseline_age_mort, final_age, mortality_event
    --covariates-csv   IID, Sex, APOE4, APOE2

Writes disease_hr.csv (panel c), mortality_burden.csv and
youthful_burden.csv (panel d).

    python3 step4_survival_analysis.py \\
        --predictions-csv output/organaging/real_predictions.csv \\
        --tte-csv $DATA_DIR/tte.csv \\
        --mortality-csv $DATA_DIR/mortality.csv \\
        --covariates-csv $DATA_DIR/covariates.csv \\
        --out-dir output/organaging
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd



def _sphere2(Z, categorical_cols):
    """Two iterations of SPHERE (k = 2), with the event columns encode-decoded."""
    for _ in range(2):
        Z = np.asarray(sphere(Z, categorical_cols=categorical_cols), dtype=float)
    return Z

ORGANS = ["Brain", "Artery", "Liver", "Immune", "Intestine", "Lung",
          "Heart", "Pancreas", "Muscle", "Adipose", "Kidney"]

ORGAN_DISEASE_PAIRS = [
    ("Brain", "alzheimer_disease"), ("Brain", "all_cause_dementia"),
    ("Heart", "heart_failure"), ("Heart", "atrial_fibrillation_or_flutter"),
    ("Lung", "emphysema_copd"), ("Kidney", "chronic_kidney_disease"),
    ("Liver", "chronic_liver_disease"), ("Pancreas", "type2_diabetes"),
    ("Artery", "ischemic_heart_disease"), ("Artery", "cerebrovascular_disease"),
    ("Immune", "rheumatoid_arthritis"), ("Immune", "leukemia"),
]

# Pairs the published forest omits: all_cause_dementia overlaps
# alzheimer_disease as an outcome, and Immune -> leukemia is null in the real
# lane (HR 0.997, p=0.97). The concordance quoted in the figure is computed
# over the remaining 10 displayed pairs; the 12-pair value is also reported.
FOREST_DROP = {("Brain", "all_cause_dementia"), ("Immune", "leukemia")}

SEED = 42
THRESH_SD = 1.5   # |organ age gap| in SD that counts as aged / youthful


def compute_gaps(preds):
    """dage_organ = z-scored residual of OLS(pred_age ~ chrono_age)."""
    age = preds["age"].to_numpy(float)
    A = np.column_stack([np.ones_like(age), age])
    out = pd.DataFrame({"iid": preds["iid"].to_numpy()})
    for organ in ORGANS:
        y = preds[f"{organ}_pred"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        z = (resid - resid.mean()) / resid.std(ddof=0)
        out[f"dage_{organ}"] = z
    return out


def apply_sphere(df):
    """SPHERE the full analysis matrix: event cols as categorical, rest continuous."""
    cols = list(df.columns)
    event_cols = [c for c in cols if c.endswith("_event") or c == "mortality_event"]
    time_cols = [c for c in cols if c.endswith("_time")
                 or c in ("age_at_baseline", "last_known_age",
                          "baseline_age_mort", "final_age")]
    ev_idx = [cols.index(c) for c in event_cols]

    Z = df.to_numpy(float)
    means = np.nanmean(Z, axis=0)
    for j in range(Z.shape[1]):
        m = np.isnan(Z[:, j])
        Z[m, j] = means[j]

    np.random.seed(SEED)
    Zs = _sphere2(np.asarray(Z, dtype=float), categorical_cols=ev_idx)

    syn = pd.DataFrame(np.asarray(Zs, dtype=object), columns=cols, index=df.index)
    for c in event_cols:
        syn[c] = syn[c].astype(int)
    for c in time_cols:
        syn[c] = syn[c].astype(float).clip(lower=0)
    for c in cols:
        if c not in event_cols and c not in time_cols:
            syn[c] = syn[c].astype(float)

    for c in event_cols:
        assert int(df[c].sum()) == int(syn[c].sum()), f"marginal drift {c}"
    print(f"[sphere] synth {syn.shape}; event marginals exact ({len(event_cols)} cols)")
    return syn


def cox_disease(df, organ, disease, cov="Sex"):
    """Delayed-entry Cox: dage_organ -> disease, adjust Sex, age timescale."""
    from lifelines import CoxPHFitter
    dage = f"dage_{organ}"
    ev, dur, ent = f"{disease}_event", f"{disease}_time", "age_at_baseline"
    need = [dage, ev, dur, ent, cov]
    if any(c not in df.columns for c in need):
        return None
    sub = df[need].dropna()
    sub = sub[(sub[dur] > sub[ent]) & sub[ev].isin([0, 1])]
    nev = int(sub[ev].sum())
    if nev < 10 or len(sub) < 100:
        return None
    try:
        cph = CoxPHFitter()
        cph.fit(sub, duration_col=dur, event_col=ev, entry_col=ent,
                formula=f"{dage} + {cov}")
        r = cph.summary.loc[dage]
        return dict(hr=float(np.exp(r["coef"])),
                    ci_lo=float(np.exp(r["coef lower 95%"])),
                    ci_hi=float(np.exp(r["coef upper 95%"])),
                    pval=float(r["p"]), n=len(sub), n_events=nev)
    except Exception as e:
        print(f"  cox fail {organ}->{disease}: {e}")
        return None


PENALIZER_FALLBACK = 0.1   # ridge fallback for the near-separated 8+ aged band


def _fit_cox_robust(sub, form, dur, ev, ent):
    """Delayed-entry Cox; if the top band near-separates (8+ aged organs,
    ~150 people at ~70% mortality -> divide-by-zero in log-partial-likelihood),
    retry the WHOLE model with a small ridge penalty so every band stays on the
    same regularization. Returns (fitter, penalizer_used)."""
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError
    try:
        cph = CoxPHFitter()
        cph.fit(sub, duration_col=dur, event_col=ev, entry_col=ent, formula=form)
        return cph, 0.0
    except (ConvergenceError, ValueError, np.linalg.LinAlgError) as e:
        print(f"  [near-separation -> ridge {PENALIZER_FALLBACK}] {e}")
        cph = CoxPHFitter(penalizer=PENALIZER_FALLBACK)
        cph.fit(sub, duration_col=dur, event_col=ev, entry_col=ent, formula=form)
        return cph, PENALIZER_FALLBACK


def _burden(df, label, kind, penalizer=None):
    """Multi-organ burden -> all-cause mortality (Oh et al. Fig 5 recipe).

    kind='aged' (>=+SD) or 'young' (<=-SD). Bands 1 / 2-4 / 5-7 / 8+ (aged) and
    1 / 2-4 / 5-7 (youthful) are mutually exclusive and compared against the
    0-organ reference. The 8+ aged tail near-separates on real data (n~150,
    ~70% mortality), so the aged model is fit with a FIXED small ridge on BOTH
    real and synth (penalizer arg) to keep them on the same footing -- otherwise
    real would be ridge-shrunk while synth converges unpenalized and the HRs
    would not be comparable. penalizer=None uses the robust auto fallback
    (youthful bands converge unpenalized)."""
    dage_cols = [f"dage_{o}" for o in ORGANS]
    ent, dur, ev = "baseline_age_mort", "final_age", "mortality_event"
    sub = df[dage_cols + [ent, dur, ev, "Sex"]].dropna()
    sub = sub[sub[dur] > sub[ent]].copy()
    if kind == "aged":
        cnt = (sub[dage_cols] >= THRESH_SD).sum(axis=1)
        cuts = [1, 2, 5, 8]
        strata = [("1", 1, 1), ("2-4", 2, 2), ("5-7", 3, 5), ("8+", 4, 8)]
    else:
        cnt = (sub[dage_cols] <= -THRESH_SD).sum(axis=1)
        cuts = [1, 2, 5]
        strata = [("1", 1, 1), ("2-4", 2, 2), ("5-7", 3, 5)]
    # mutually-exclusive banding vs cat-0 reference; drop any band with <20
    # people so the design matrix stays full rank.
    cat = np.zeros(len(sub), dtype=int)
    for i, c in enumerate(cuts, start=1):
        cat[cnt.to_numpy() >= c] = i
    strata = [(name, k, k_rep) for (name, k, k_rep) in strata
              if int((cat == k).sum()) >= 20]
    for name, k, k_rep in strata:
        sub[f"cat_{k}"] = (cat == k).astype(int)
    form = " + ".join(f"cat_{k}" for _, k, _ in strata) + " + Sex"
    if penalizer is None:
        cph, pen = _fit_cox_robust(sub, form, dur, ev, ent)
    else:
        from lifelines import CoxPHFitter
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(sub, duration_col=dur, event_col=ev, entry_col=ent, formula=form)
        pen = penalizer
    rows = []
    for name, k, k_rep in strata:
        try:
            r = cph.summary.loc[f"cat_{k}"]
        except KeyError:
            continue
        rows.append(dict(kind=kind, stratum=name, k=k_rep, label=label,
                         hr=float(np.exp(r["coef"])),
                         ci_lo=float(np.exp(r["coef lower 95%"])),
                         ci_hi=float(np.exp(r["coef upper 95%"])),
                         pval=float(r["p"]),
                         n=int((cat == k).sum()), penalizer=pen))
        print(f"  {label:5s} {kind:5s} {name:4s} HR={rows[-1]['hr']:.2f} "
              f"[{rows[-1]['ci_lo']:.2f}-{rows[-1]['ci_hi']:.2f}] "
              f"p={rows[-1]['pval']:.1e} n={rows[-1]['n']} pen={pen}")
    return pd.DataFrame(rows)


def run_youthful_organ(df, label):
    """Youthful brain / immune / both -> mortality (Oh: 0.60/0.58/0.44)."""
    from lifelines import CoxPHFitter
    ent, dur, ev = "baseline_age_mort", "final_age", "mortality_event"
    sub = df[["dage_Brain", "dage_Immune", ent, dur, ev, "Sex"]].dropna()
    sub = sub[sub[dur] > sub[ent]].copy()
    sub["yb"] = (sub["dage_Brain"] <= -THRESH_SD).astype(int)
    sub["yi"] = (sub["dage_Immune"] <= -THRESH_SD).astype(int)
    sub["yboth"] = ((sub["yb"] == 1) & (sub["yi"] == 1)).astype(int)
    rows = []
    for name, col in [("youthful_brain", "yb"), ("youthful_immune", "yi"),
                      ("youthful_both", "yboth")]:
        cph = CoxPHFitter()
        cph.fit(sub[[col, ent, dur, ev, "Sex"]], duration_col=dur,
                event_col=ev, entry_col=ent, formula=f"{col} + Sex")
        r = cph.summary.loc[col]
        rows.append(dict(target=name, label=label,
                         hr=float(np.exp(r["coef"])),
                         ci_lo=float(np.exp(r["coef lower 95%"])),
                         ci_hi=float(np.exp(r["coef upper 95%"])),
                         pval=float(r["p"]), n=int(sub[col].sum())))
        print(f"  {label:5s} {name:16s} HR={rows[-1]['hr']:.2f} "
              f"[{rows[-1]['ci_lo']:.2f}-{rows[-1]['ci_hi']:.2f}] "
              f"p={rows[-1]['pval']:.1e} n={rows[-1]['n']}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Organ age-gap survival analysis: real vs SPHERE")
    ap.add_argument("--predictions-csv", required=True)
    ap.add_argument("--tte-csv", required=True)
    ap.add_argument("--mortality-csv", required=True)
    ap.add_argument("--covariates-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. compute organ-age gaps
    preds = pd.read_csv(args.predictions_csv)
    gaps = compute_gaps(preds)

    # 2. merge analysis dataset
    meta = pd.read_csv(args.covariates_csv)
    meta = meta.rename(columns={"IID": "iid"})   # keep Sex/APOE4/APOE2 case
    diseases = sorted({d for _, d in ORGAN_DISEASE_PAIRS})
    tte = pd.read_csv(args.tte_csv)
    tte = tte.rename(columns={tte.columns[0]: "iid"})
    mort = pd.read_csv(args.mortality_csv)
    mort = mort.rename(columns={mort.columns[0]: "iid"})

    df = (gaps.merge(meta, on="iid", how="inner")
              .merge(tte, on="iid", how="inner")
              .merge(mort, on="iid", how="inner")).set_index("iid")
    print(f"[build] analysis dataset: {df.shape}")

    # 3. SPHERE the full analysis matrix
    syn = apply_sphere(df)

    # 4. Cox models on both lanes
    rows = []
    for label, data in [("real", df), ("synth", syn)]:
        for organ, disease in ORGAN_DISEASE_PAIRS:
            r = cox_disease(data, organ, disease)
            if r:
                rows.append({"label": label, "organ": organ,
                             "disease": disease, **r})
                sig = "*" if r["pval"] < 0.05 else ""
                print(f"  [{label}] {organ:12s} -> {disease:30s} "
                      f"HR={r['hr']:.3f} p={r['pval']:.2e}{sig}")

    hr_df = pd.DataFrame(rows)
    hr_df.to_csv(f"{args.out_dir}/disease_hr.csv", index=False)

    # ── multi-organ burden -> all-cause mortality (Oh et al. Fig 5) ──
    # The aged model carries the near-separated 8+ band, so fix the same small
    # ridge on BOTH lanes; youthful bands converge unpenalized.
    print("\n== multi-organ mortality burden ==")
    burden = pd.concat([_burden(df, "real", "aged", penalizer=PENALIZER_FALLBACK),
                        _burden(syn, "synth", "aged", penalizer=PENALIZER_FALLBACK)],
                       ignore_index=True)
    burden.to_csv(f"{args.out_dir}/mortality_burden.csv", index=False)

    print("\n== youthful-organ burden -> mortality ==")
    yb = pd.concat([_burden(df, "real", "young"),
                    _burden(syn, "synth", "young"),
                    run_youthful_organ(df, "real"),
                    run_youthful_organ(syn, "synth")], ignore_index=True)
    yb.to_csv(f"{args.out_dir}/youthful_burden.csv", index=False)

    # log-HR concordance
    real_h = hr_df[hr_df["label"] == "real"].set_index(["organ", "disease"])
    syn_h = hr_df[hr_df["label"] == "synth"].set_index(["organ", "disease"])
    shared = real_h.index.intersection(syn_h.index)
    displayed = [p for p in shared if p not in FOREST_DROP]
    if len(displayed) > 2:
        lr = np.log(real_h.loc[displayed, "hr"].values)
        ls = np.log(syn_h.loc[displayed, "hr"].values)
        print(f"[concordance] log-HR r = {np.corrcoef(lr, ls)[0, 1]:.4f}  "
              f"(n={len(displayed)} displayed pairs)   <- value quoted in panel c")
    if len(shared) > 2:
        lr = np.log(real_h.loc[shared, "hr"].values)
        ls = np.log(syn_h.loc[shared, "hr"].values)
        r = float(np.corrcoef(lr, ls)[0, 1])
        print(f"\n[concordance] log-HR r = {r:.4f}  (n={len(shared)} pairs)")

    print(f"[write] -> {args.out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Figure 5 rendering — draw the 7-panel composite from the step outputs.

Run steps 1-6 first, then this script:

    a  Conventional aging clock, real vs SPHERE      step1 + step3
    b  per-clock actual/predicted correlation        step3
    c  organ age gap vs disease risk                 step4
    d  extreme agers vs mortality risk               step4
    e  Stanford ADRC pTau217 volcano, real           step5
    f  Stanford ADRC pTau217 volcano, SPHERE         step5
    g  GNPC WSS module vs cognition heatmap          step6

Usage:
    python3 code/figure5/render.py \\
        --organaging-dir output/organaging \\
        --survival-dir output/organaging \\
        --ptau-dir output/ptau217 \\
        --gnpc-csv output/gnpc/gnpc_wss_heatmap.csv \\
        --out-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

DPI, SEED = 300, 0
rng = np.random.default_rng(SEED)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
FS_LAB, FS_TICK, FS_ANNOT, FS_LETTER = 8.5, 7.5, 6.8, 12

# house palette, shared with Figures 2-4
C_REAL, C_S1 = "#2166AC", "#D73027"
CMAP_B = LinearSegmentedColormap.from_list("b", ["#d6e8f7", "#2166AC"])
CMAP_R = LinearSegmentedColormap.from_list("r", ["#fddbc7", "#D73027"])
CMAP_G = LinearSegmentedColormap.from_list("g", ["#2166AC", "#ffffff", "#B2182B"])
C_DIAG, NS, THR = "#bbbbbb", "#bdbdbd", 0.05
YCAP = 15.0     # -log10 p cap for the ADRC volcanos; larger hits are clipped
LOGP_CAP = 30.0  # GNPC heatmap colour scale is symmetric about 0 at +/- this

GNPC_LABEL = {"diagnosis_bin": "AD vs HC", "last_diag_bin": "AD vs HC (last)",
              "MoCA": "MoCA", "last_MoCA": "MoCA (last)"}

DISEASE_LABEL = {
    "alzheimer_disease": "Alzheimer's disease",
    "all_cause_dementia": "all-cause dementia",
    "heart_failure": "heart failure",
    "atrial_fibrillation_or_flutter": "atrial fibrillation",
    "emphysema_copd": "COPD",
    "chronic_kidney_disease": "chronic kidney disease",
    "chronic_liver_disease": "chronic liver disease",
    "type2_diabetes": "type 2 diabetes",
    "ischemic_heart_disease": "ischemic heart disease",
    "cerebrovascular_disease": "cerebrovascular disease",
    "rheumatoid_arthritis": "rheumatoid arthritis",
    "leukemia": "leukemia",
}
# pairs the published forest omits (see FOREST_DROP in step 4)
FOREST_DROP = {("Brain", "all_cause_dementia"), ("Immune", "leukemia")}

# panel d strata, left to right: youthful (most to least) then aged
YOUNG_ORDER = ["5-7", "2-4", "1"]
AGED_ORDER = ["1", "2-4", "5-7", "8+"]


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


def letter(ax, s):
    ax.text(-0.02, 1.03, s, transform=ax.transAxes, fontsize=FS_LETTER,
            fontweight="bold", va="bottom", ha="right")


def banner(fig, x0, x1, y, text):
    """Study-level grouping header centred over [x0, x1]."""
    fig.text((x0 + x1) / 2, y, text, ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="#141414", zorder=6)


# ── panel a ──────────────────────────────────────────────────────────────────
def _density(x, y, basis=5000):
    pts = np.vstack([x, y])
    idx = rng.choice(pts.shape[1], size=min(basis, pts.shape[1]), replace=False)
    d = gaussian_kde(pts[:, idx])(pts)
    return d, d.argsort()


def _jitter(x):
    """Spread integer ages so the density cloud is not banded."""
    if np.allclose(x, np.round(x)):
        return x + rng.uniform(-0.45, 0.45, len(x))
    return x


def panel_a(gs, organ_dir):
    real = pd.read_csv(organ_dir / "real_predictions.csv")
    syn = pd.read_csv(organ_dir / "synth_predictions.csv")
    rt = real[real["split"] == "test"]
    st = syn[syn["split"] == "test"]
    xr, yr = rt["age"].to_numpy(float), rt["Conventional_pred"].to_numpy(float)
    xs_, ys = st["age"].to_numpy(float), st["Conventional_pred"].to_numpy(float)
    lo = min(xr.min(), yr.min(), xs_.min(), ys.min()) - 2
    hi = max(xr.max(), yr.max(), xs_.max(), ys.max()) + 2

    ax = plt.subplot(gs)
    corr = {}
    for x, y, cmap, name in ((xr, yr, CMAP_B, "real"), (xs_, ys, CMAP_R, "synth")):
        corr[name] = float(np.corrcoef(x, y)[0, 1])
        sl, ic = np.polyfit(x, y, 1)
        xj = _jitter(x)
        d, order = _density(xj, y)
        ax.scatter(xj[order], y[order], c=d[order], cmap=cmap, s=2, alpha=0.20,
                   edgecolor="none", rasterized=True, zorder=1,
                   vmin=d.min(), vmax=d.max() * 0.7)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, sl * xs + ic, color="#222222", lw=1.0, zorder=6)
    ax.plot([lo, hi], [lo, hi], color=C_DIAG, lw=0.8, ls="--", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Chronological age (years)", fontsize=FS_LAB)
    ax.set_ylabel("Predicted age (years)", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.05, 0.96,
            f"Real     r = {corr['real']:.3f}\nSPHERE  r = {corr['synth']:.3f}",
            transform=ax.transAxes, va="top", fontsize=FS_ANNOT, linespacing=1.4)
    ax.legend(handles=[Line2D([0], [0], marker="o", color=C_REAL, ls="", ms=5,
                              label="Real"),
                       Line2D([0], [0], marker="o", color=C_S1, ls="", ms=5,
                              label="SPHERE")],
              fontsize=FS_ANNOT, loc="lower right", frameon=False,
              handletextpad=0.4)
    letter(ax, "a")
    ax.set_title("Conventional aging clock", fontsize=8, fontweight="bold",
                 loc="left", pad=6)
    return dict(real_r=corr["real"], synth_r=corr["synth"],
                delta_r=corr["synth"] - corr["real"])


# ── panel b ──────────────────────────────────────────────────────────────────
def panel_b(gs, organ_dir):
    d = pd.read_csv(organ_dir / "real_vs_synth_identity.csv")
    d = d.sort_values("test_r_real").reset_index(drop=True)
    y = np.arange(len(d))
    ax = plt.subplot(gs)
    series = [("test_r_real", "Real", C_REAL, "o", 0.15),
              ("test_r_synth", "SPHERE", C_S1, "^", -0.15)]
    for col, lab, c, mk, off in series:
        ax.scatter(d[col], y + off, marker=mk, s=26, color=c, edgecolor="white",
                   linewidth=0.4, zorder=3, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels(d["clock"], fontsize=FS_TICK)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel("Pearson r", fontsize=FS_LAB)
    ax.grid(axis="x", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(d) - 0.4)
    mx = float((d["test_r_synth"] - d["test_r_real"]).abs().max())
    ax.text(0.03, 0.97, f"max |Δr| = {mx:.0e}", transform=ax.transAxes,
            fontsize=FS_ANNOT, va="top")
    ax.legend(handles=[Line2D([0], [0], marker=mk, color=c, ls="", ms=5.5,
                              markeredgecolor="white", label=lab)
                       for _, lab, c, mk, _ in series],
              fontsize=FS_ANNOT, loc="lower right", frameon=False,
              handletextpad=0.3, labelspacing=0.25)
    letter(ax, "b")
    ax.set_title("Actual age vs predicted age correlation", fontsize=8,
                 fontweight="bold", loc="left", pad=6)
    return dict(max_abs_dev=mx, n_organs=int(len(d)))


# ── panel c ──────────────────────────────────────────────────────────────────
def panel_c(gs, surv_dir):
    d = pd.read_csv(surv_dir / "disease_hr.csv")
    pairs = [(o, di) for o, di in
             d[d["label"] == "real"][["organ", "disease"]].to_numpy().tolist()
             if (o, di) not in FOREST_DROP]
    lr = [np.log(d.query("organ == @o and disease == @di and label == 'real'")
                 .hr.iloc[0]) for o, di in pairs]
    ls = [np.log(d.query("organ == @o and disease == @di and label == 'synth'")
                 .hr.iloc[0]) for o, di in pairs]
    r_cox = float(np.corrcoef(lr, ls)[0, 1])

    ax = plt.subplot(gs)
    y = np.arange(len(pairs))[::-1]
    xmax = 0.0
    for yi, (org, dis) in zip(y, pairs):
        rr = d.query("organ == @org and disease == @dis and label == 'real'").iloc[0]
        ss = d.query("organ == @org and disease == @dis and label == 'synth'").iloc[0]
        ax.plot([rr.ci_lo, rr.ci_hi], [yi + 0.18] * 2, color=C_REAL, lw=1.0, zorder=2)
        ax.scatter(rr.hr, yi + 0.18, marker="o", s=22, color=C_REAL,
                   edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot([ss.ci_lo, ss.ci_hi], [yi - 0.18] * 2, color=C_S1, lw=1.0,
                alpha=0.7, zorder=2)
        ax.scatter(ss.hr, yi - 0.18, marker="s", s=20, facecolor="none",
                   edgecolor=C_S1, linewidth=0.9, alpha=0.9, zorder=3)
        xmax = max(xmax, rr.ci_hi, ss.ci_hi)
        ax.text(rr.ci_hi + 0.03, yi + 0.18, stars(rr.pval), fontsize=6,
                va="center", ha="left", color=C_REAL)
        ax.text(ss.ci_hi + 0.03, yi - 0.18, stars(ss.pval), fontsize=6,
                va="center", ha="left", color=C_S1)
    ax.axvline(1.0, color="#888888", lw=0.7, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([DISEASE_LABEL.get(di, di) for _, di in pairs], fontsize=6.0)
    ax.set_ylim(-0.7, len(pairs) - 0.3)
    ax.set_xlim(0.85, xmax + 0.28)
    ax.set_xlabel("HR", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.97, 0.05, f"r = {r_cox:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FS_ANNOT)
    ax.legend(handles=[Line2D([0], [0], marker="o", color=C_REAL, ls="", ms=5,
                              markeredgecolor="white", label="Real"),
                       Line2D([0], [0], marker="s", color=C_S1, ls="", ms=5,
                              markerfacecolor="none", label="SPHERE")],
              fontsize=FS_ANNOT, loc="lower right", frameon=False,
              bbox_to_anchor=(1.0, 0.14), handletextpad=0.3)
    letter(ax, "c")
    ax.set_title("Organ age gap vs disease risk", fontsize=8, fontweight="bold",
                 loc="left", pad=6)
    return dict(log_hr_r=r_cox, n_pairs=len(pairs))


# ── panel d ──────────────────────────────────────────────────────────────────
def panel_d(gs, surv_dir):
    aged = pd.read_csv(surv_dir / "mortality_burden.csv")
    young = pd.read_csv(surv_dir / "youthful_burden.csv")
    young = young[young["target"].isna()]      # drop the brain/immune rows
    rows = ([("young", s) for s in YOUNG_ORDER] +
            [("aged", s) for s in AGED_ORDER])

    ax = plt.subplot(gs)
    x = np.arange(len(rows))
    lr, ls = [], []
    for xi, (kind, strat) in zip(x, rows):
        src = aged if kind == "aged" else young
        rr = src.query("stratum == @strat and label == 'real'").iloc[0]
        ss = src.query("stratum == @strat and label == 'synth'").iloc[0]
        lr.append(np.log(rr.hr))
        ls.append(np.log(ss.hr))
        for row, off, mk, fc, ec in [(rr, -0.16, "o", C_REAL, C_REAL),
                                     (ss, 0.16, "s", "none", C_S1)]:
            ax.plot([xi + off] * 2, [row.ci_lo, row.ci_hi], color=ec, lw=1.0,
                    alpha=0.8, zorder=2)
            ax.scatter(xi + off, row.hr, marker=mk, s=22, facecolor=fc,
                       edgecolor=ec, linewidth=0.8, zorder=3)
            ax.text(xi + off, row.ci_hi * 1.06, stars(row.pval), fontsize=6,
                    va="bottom", ha="center", color=ec)
    r_d = float(np.corrcoef(lr, ls)[0, 1])
    ax.axhline(1.0, color="#888888", lw=0.7, ls="--", zorder=1)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.set_yticklabels(["0.5", "1", "2", "4", "8"])
    ax.set_ylim(top=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f"youthful {s}" for s in YOUNG_ORDER] +
                       [f"aged {s}" for s in AGED_ORDER],
                       fontsize=FS_TICK, rotation=45, ha="right")
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylabel("HR", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.97, 0.05, f"r = {r_d:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FS_ANNOT)
    ax.legend(handles=[Line2D([0], [0], marker="o", color=C_REAL, ls="", ms=5,
                              label="Real"),
                       Line2D([0], [0], marker="s", color=C_S1, ls="", ms=5,
                              markerfacecolor="none", label="SPHERE")],
              fontsize=FS_ANNOT, loc="upper left", frameon=False,
              handletextpad=0.3)
    letter(ax, "d")
    ax.set_title("Extreme agers vs mortality risk", fontsize=8,
                 fontweight="bold", loc="left", pad=6)
    return dict(n_strata=len(rows), log_hr_r=r_d)


# ── panels e / f ─────────────────────────────────────────────────────────────
def _volcano(ax, beta, pval, padj, title, sig_color, ytop=16.0):
    beta = np.asarray(beta, float)
    pv = np.asarray(pval, float)
    yv = np.minimum(-np.log10(pv), YCAP)
    sig = np.asarray(padj) < THR
    col = np.full(len(beta), NS, dtype=object)
    col[sig] = sig_color
    ax.scatter(beta, yv, c=col, s=4, alpha=0.7, edgecolors="none", rasterized=True)
    if sig.any():
        # dashed line at the raw p that the BH-FDR<0.05 cut maps to
        ax.axhline(min(-np.log10(pv[sig].max()), YCAP), ls="--", lw=0.5, c="grey")
    ax.set_title(title, fontsize=7, fontweight="bold", pad=3)
    ax.text(0.03, 0.97, f"FDR<0.05: n={int(sig.sum())}", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS_ANNOT, color="black")
    ax.set_xlabel("β (pTau217)", fontsize=FS_LAB)
    ax.tick_params(labelsize=6.5)
    ax.set_ylim(0, ytop)      # headroom so points clipped at YCAP stay visible
    ax.set_yticks([0, 5, 10, 15])


def panel_ef(sub, ptau_dir):
    real = pd.read_csv(ptau_dir / "ptau217_glm_real.csv")
    sphere_ = pd.read_csv(ptau_dir / "ptau217_glm_sphere.csv")
    sphere_ = sphere_.set_index("ix").loc[real["ix"]].reset_index()

    ax_e, ax_f = plt.subplot(sub[0]), plt.subplot(sub[1])
    _volcano(ax_e, real["beta"], real["pval"], real["padj"],
             "Stanford ADRC : Real", C_REAL)
    _volcano(ax_f, sphere_["beta"], sphere_["pval"], sphere_["padj"],
             "Stanford ADRC : SPHERE", C_S1)
    xmin = float(min(real["beta"].min(), sphere_["beta"].min()))
    xmax = float(max(real["beta"].max(), sphere_["beta"].max()))
    pad = 0.05 * (xmax - xmin)
    for a in (ax_e, ax_f):
        a.set_xlim(xmin - pad, xmax + pad)
        a.set_ylabel(r"$-\log_{10}(p)$", fontsize=FS_LAB)
    ax_e.set_xlabel("")
    ax_e.tick_params(labelbottom=False)
    letter(ax_e, "e")
    letter(ax_f, "f")
    return dict(n_real=int((real["padj"] < THR).sum()),
                n_sphere=int((sphere_["padj"] < THR).sum()),
                beta_r=float(np.corrcoef(real["beta"], sphere_["beta"])[0, 1]),
                ycap=YCAP)


# ── panel g ──────────────────────────────────────────────────────────────────
def panel_g(gs, gnpc_csv):
    """GNPC WSS module vs cognition heatmap, Real and SPHERE side by side.

    Rows are the WSS pathway modules ordered by mean |LogP| (most associated
    at the top), columns are outcome x arm with a blank spacer between the
    two arms. Fill is -log10(p) signed by beta, so direction is preserved,
    on a symmetric scale clipped at +/- LOGP_CAP; a black cell border marks
    BH-FDR padj<0.05 and a grey border raw pval<0.05.
    """
    dt = pd.read_csv(gnpc_csv)
    dt["arm"] = np.where(dt["dx"].str.contains(r"\(sphere\)"), "sphere", "actual")
    dt["outcome"] = dt["dx"].str.replace(r"\s*\((actual|sphere)\)", "", regex=True)

    outcomes = list(dict.fromkeys(dt["outcome"]))
    order = (dt.groupby("ix")["LogP"].apply(lambda s: s.abs().mean())
             .sort_values(ascending=False).index.tolist())
    cols = ([(o, "actual") for o in outcomes] + [None] +
            [(o, "sphere") for o in outcomes])

    grid = np.full((len(order), len(cols)), np.nan)
    row_of = {m: i for i, m in enumerate(order)}
    borders = []
    for j, key in enumerate(cols):
        if key is None:
            continue
        outcome, arm = key
        sel = dt[(dt["outcome"] == outcome) & (dt["arm"] == arm)]
        for _, row in sel.iterrows():
            i = row_of[row["ix"]]
            grid[i, j] = np.clip(row["LogP"], -LOGP_CAP, LOGP_CAP)
            if row["padj"] < THR:
                borders.append((i, j, "black", 0.7))
            elif row["pval"] < THR:
                borders.append((i, j, "#909090", 0.45))

    ax = plt.subplot(gs)
    im = ax.imshow(grid, cmap=CMAP_G, vmin=-LOGP_CAP, vmax=LOGP_CAP,
                   aspect="auto", interpolation="nearest")
    for i, j, colour, lw in borders:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=colour, linewidth=lw))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["" if k is None else GNPC_LABEL.get(k[0], k[0])
                        for k in cols], fontsize=6.0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=4.2)
    ax.tick_params(length=0, pad=1.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    n_out = len(outcomes)
    for x0, x1, text, colour in ((0, n_out - 1, "Real", C_REAL),
                                 (n_out + 1, len(cols) - 1, "SPHERE", C_S1)):
        ax.plot([x0 - 0.4, x1 + 0.4], [-1.4] * 2, color=colour, lw=2.5,
                clip_on=False)
        ax.text((x0 + x1) / 2, -2.1, text, ha="center", va="bottom",
                fontsize=FS_ANNOT, fontweight="bold", color=colour)

    cax = ax.inset_axes([0.24, -0.075, 0.52, 0.010])
    cb = ax.figure.colorbar(im, cax=cax, orientation="horizontal",
                            ticks=[-LOGP_CAP, 0, LOGP_CAP])
    cb.ax.set_xticklabels([f"−{LOGP_CAP:.0f}", "0", f"{LOGP_CAP:.0f}"],
                          fontsize=5.5)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, width=0.4, pad=1)
    cb.set_label(r"$-\log_{10}(p)\times\mathrm{sign}(\beta)$", fontsize=5.5,
                 labelpad=1.5)
    letter(ax, "g")
    ax.set_title("GNPC", fontsize=8, fontweight="bold", loc="left", pad=6)
    return dict(n_modules=len(order), outcomes=outcomes,
                sig_border="padj<0.05")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Render the Figure 5 composite")
    ap.add_argument("--organaging-dir", required=True,
                    help="step1 + step3 outputs (predictions, identity table)")
    ap.add_argument("--survival-dir", required=True, help="step4 outputs")
    ap.add_argument("--ptau-dir", required=True, help="step5 outputs")
    ap.add_argument("--gnpc-csv", required=True,
                    help="gnpc_wss_heatmap.csv from step6")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    organ_dir = Path(args.organaging_dir)
    surv_dir = Path(args.survival_dir)
    ptau_dir = Path(args.ptau_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 11.45))
    # a-d (UK Biobank) as a tight 2x2 block, then a gap, then e-g (ADRC/GNPC):
    # the stacked volcanos on the left and the tall GNPC heatmap on the right.
    gs_top = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.42,
                              left=0.17, right=0.97, top=0.959, bottom=0.469,
                              height_ratios=[1.0, 1.15])
    gs_ef = fig.add_gridspec(2, 1, left=0.17, right=0.42, top=0.378,
                             bottom=0.0586, hspace=0.45)
    # the GNPC heatmap carries 45 module labels on its left, so its axes start
    # well right of the volcano column
    gs_g = fig.add_gridspec(1, 1, left=0.665, right=0.985, top=0.368, bottom=0.085)

    summary = {"layout": "a b / c d / (e;f) g"}
    summary["a"] = panel_a(gs_top[0, 0], organ_dir)
    summary["b"] = panel_b(gs_top[0, 1], organ_dir)
    summary["c"] = panel_c(gs_top[1, 0], surv_dir)
    summary["d"] = panel_d(gs_top[1, 1], surv_dir)
    summary["ef"] = panel_ef([gs_ef[0], gs_ef[1]], ptau_dir)
    summary["g"] = panel_g(gs_g[0, 0], args.gnpc_csv)

    pa = gs_top[0, 0].get_position(fig)
    pb = gs_top[0, 1].get_position(fig)
    pe = gs_ef[0].get_position(fig)
    pg = gs_g[0, 0].get_position(fig)
    banner(fig, pa.x0, pb.x1, pa.y1 + 0.022, "UK Biobank (Oh et al.)")
    banner(fig, pe.x0, pg.x1, pe.y1 + 0.02, "Stanford ADRC & GNPC (Butler et al.)")

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"sphere_organaging_figure.{ext}", dpi=DPI,
                    facecolor="white")
    print(f"[render] -> {out_dir}/sphere_organaging_figure.{{pdf,png}}")
    for key, val in summary.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Composite figure 4 — the seven-panel Stanford ADRC figure.

    a  data availability: participant x modality heatmap, real vs SPHERE x2
    b  distributional fidelity: KS / dMean / dVariance / dCorrelation + composite
    c  two-anchor privacy at k = 2 (singling-out / linkability / inference)
    d  AD beta-coefficient preservation at k = 2
    e  AD p-value preservation at k = 2
    f  cross-modality correlation preservation at k = 2
    g  cross-modality p-value preservation at k = 2

This script only draws. Every number it plots is computed by a step script and
passed in as a table: panel a by step 6, panel b by step 7, panel c by step 5,
panels d and e by step 4, panels f and g by step 8. Nothing here touches the
modality CSVs, so the figure can be redrawn without the ADRC data.

    python3 render.py \\
        --availability-csv results/figure4/panel_a_availability.csv \\
        --fidelity-csv     results/figure4/panel_b_fidelity.csv \\
        --privacy-csv      results/figure4/privacy_unified/adrc_privacy_unified.csv \\
        --validation-js    results/figure4/adrc_validation.js \\
        --crossmod-csv     results/figure4/panels_fg_cross_modality.csv \\
        --out              figures/adrc_figure
"""
import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless backend, before pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from adrc_common import CROSS_MODS, DIAG_ORDER, FIG_MOD_ORDER, ID_COL, RELEASE_K

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent               # <repo>/code/figure4
ROOT = HERE.parent.parent                            # <repo>
FIG_DIR = ROOT / "figures"



# Figure aesthetics
CARDINAL, TEAL, GOLD = "#8C1515", "#175E54", "#b6a477"
MOD_COLORS = {"demographics": "#4e79a7", "cognitive": "#e15759", "biomarkers": "#f28e2b",
              "imaging_amyloid": "#76b7b2", "imaging_tau": "#59a14f", "wgs": "#af7aa1",
              "scrna": "#ff9da7", "csf": "#9c755f", "plasma": "#bab0ac"}
MOD_LABELS = {"demographics": "Demographics", "cognitive": "Cognitive", "biomarkers": "Biomarkers",
              "imaging_amyloid": "Amyloid PET", "imaging_tau": "Tau PET", "wgs": "Genotype",
              "scrna": "scRNA", "csf": "CSF Prot.", "plasma": "Plasma Prot."}
MOD_ORDER  = FIG_MOD_ORDER
DIAG_COLORS = {"HC": "#175E54", "MCI": "#b6a477", "AD": "#8C1515", "PD": "#4e79a7",
               "PDMCI": "#af7aa1", "PDD": "#9c755f", "LBD": "#76b7b2", "Other": "#888888"}


def _panel_label(ax, letter, fig, dx=-0.011, dy=0.012):
    pos = ax.get_position()
    fig.text(pos.x0 + dx, pos.y1 + dy, letter, fontsize=11, fontweight="bold",
             va="bottom", ha="right", transform=fig.transFigure)


def render_figure(availability_csv: Path, fidelity_csv: Path, privacy_csv: Path,
                  validation_js: Path, crossmod_csv: Path, out_base: Path,
                  release_k=RELEASE_K, fg_cmap="Blues"):
    """Draw the 7-panel ADRC figure from the step outputs. Nothing is computed
    here beyond laying the tables out."""
    sx = f"SPHERE×{release_k}"

    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "figure.dpi": 150,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})

    # ── Load validation (Panels C/D/E) ───────────────────────────────────────────
    print("Loading validation metrics …")
    vtxt = Path(validation_js).read_text()
    vtxt = re.sub(r"^window\.ADRC_VALIDATION\s*=\s*", "", vtxt.strip()).rstrip(";")
    VAL = json.loads(vtxt)

    # ── Panel A: data availability (step 6) ──────────────────────────────────────
    avail_df = pd.read_csv(availability_csv, dtype={ID_COL: str})
    all_pids_sorted = avail_df[ID_COL].tolist()
    pid_diag = dict(zip(all_pids_sorted, avail_df["diagnosis"]))
    avail_real = avail_df[[f"real_{m}" for m in MOD_ORDER]].to_numpy(dtype=np.float32)
    avail_sphere = avail_df[[f"sphere_{m}" for m in MOD_ORDER]].to_numpy(dtype=np.float32)

    # ── Panel B: distributional fidelity (step 7) ────────────────────────────────
    fid_df = pd.read_csv(fidelity_csv).set_index("modality")
    fidelity = {mod: fid_df.loc[mod, ["ks", "mean", "var", "corr", "composite"]].to_dict()
                for mod in fid_df.index}

    # ── Panel C: two-anchor privacy at k=release_k (g/h style) ───────────────────
    # Values come from the canonical unified-privacy CSV (single source of truth;
    # identical to Supplementary Table S2), NOT the live VAL progression, so the
    # panel and the table can never disagree. Demographics is excluded because its
    # low-cardinality one-hot columns collapse the two anchors (unstable LK/INF).
    PRIV_METRICS = ["singling_out", "linkability", "inference"]
    PRIV_MODS = ["cognitive", "biomarkers", "imaging_amy", "imaging_tau",
                 "wgs", "scrna", "csf", "plasma"]
    _pdf = pd.read_csv(privacy_csv)
    _pk = _pdf[(_pdf["k"] == release_k) & (_pdf["modality"].isin(PRIV_MODS))]
    priv_mean, priv_sd = {}, {}
    for mk in PRIV_METRICS:
        vv = _pk.loc[_pk["metric"] == mk, "score_mean"].to_numpy(dtype=float)
        priv_mean[mk] = float(np.mean(vv))
        priv_sd[mk] = float(np.std(vv, ddof=1))

    # ── Panels D/E: utility (from VAL) ───────────────────────────────────────────
    utility_data = {mod: {"beta_corr": (md.get("utility") or {}).get("beta_corr"),
                          "pval_corr": (md.get("utility") or {}).get("pval_corr")}
                    for mod, md in VAL.items()}

    # ── Panels F/G: cross-modality preservation (step 8) ─────────────────────────
    n_cm = len(CROSS_MODS)
    cm_idx = {m: i for i, m in enumerate(CROSS_MODS)}
    heat_s, heat_p = np.full((n_cm, n_cm), np.nan), np.full((n_cm, n_cm), np.nan)
    for _, row in pd.read_csv(crossmod_csv).iterrows():
        i, j = cm_idx[row["modality_i"]], cm_idx[row["modality_j"]]
        heat_s[i, j] = heat_s[j, i] = row["corr_score"]
        heat_p[i, j] = heat_p[j, i] = row["pval_score"]

    # ── Assemble figure ──────────────────────────────────────────────────────────
    print("Assembling figure …")
    FIG_W, FIG_H = 183 / 25.4, 247 / 25.4
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs_top = gridspec.GridSpec(1, 1, figure=fig, top=0.97, bottom=0.76, left=0.07, right=0.97)
    gs_mid = gridspec.GridSpec(1, 1, figure=fig, top=0.70, bottom=0.56, left=0.07, right=0.97)
    gs_bot = gridspec.GridSpec(1, 3, figure=fig, top=0.49, bottom=0.29, left=0.07, right=0.97, wspace=0.42)
    gs_btm = gridspec.GridSpec(1, 2, figure=fig, top=0.21, bottom=0.01, left=0.07, right=0.97, wspace=0.28)
    ax_A = fig.add_subplot(gs_top[0, 0]); ax_B = fig.add_subplot(gs_mid[0, 0])
    ax_C = fig.add_subplot(gs_bot[0, 0]); ax_D = fig.add_subplot(gs_bot[0, 1])
    ax_E = fig.add_subplot(gs_bot[0, 2]); ax_F = fig.add_subplot(gs_btm[0, 0])
    ax_G = fig.add_subplot(gs_btm[0, 1])

    # Panel A
    ax = ax_A; ax.axis("off"); _panel_label(ax, "a", fig)
    n_parts = len(all_pids_sorted)
    LABEL_W, BAR_W, GAP = 0.09, 0.42, 0.03
    X_L1 = LABEL_W; X_L2 = LABEL_W + BAR_W + GAP
    Y_BOT, Y_TOP, STRIP_H = 0.04, 0.80, 0.055
    STRIP_Y = Y_TOP + 0.02
    row_total_h = (Y_TOP - Y_BOT) / len(MOD_ORDER)
    row_h = row_total_h * 0.80

    def draw_avail(avail_mat, x_left, bar_w, show_counts):
        for ri, mod in enumerate(MOD_ORDER):
            y0 = Y_BOT + ri * row_total_h
            axr = ax.inset_axes([x_left, y0, bar_w, row_h])
            pres = avail_mat[:, MOD_ORDER.index(mod)]
            rgb = np.array(matplotlib.colors.to_rgb(MOD_COLORS[mod]))
            img = np.ones((1, n_parts, 4), dtype=np.float32)
            for xi in range(n_parts):
                img[0, xi, :3] = rgb if pres[xi] > 0 else [0.96, 0.96, 0.96]
            axr.imshow(img, aspect="auto", interpolation="none")
            axr.set_xlim(-0.5, n_parts - 0.5); axr.axis("off")
            if show_counts:
                ax.text(x_left + bar_w + 0.005, y0 + row_h / 2, str(int(pres.sum())),
                        transform=ax.transAxes, ha="left", va="center", fontsize=6.2, color="#666666")
        ax_s = ax.inset_axes([x_left, STRIP_Y, bar_w, STRIP_H])
        simg = np.zeros((1, n_parts, 3), dtype=np.float32)
        for xi, p in enumerate(all_pids_sorted):
            simg[0, xi, :] = matplotlib.colors.to_rgb(DIAG_COLORS.get(pid_diag.get(p, "Other"), "#888"))
        ax_s.imshow(simg, aspect="auto", interpolation="none")
        ax_s.set_xlim(-0.5, n_parts - 0.5); ax_s.axis("off")

    draw_avail(avail_real, X_L1, BAR_W, False)
    draw_avail(avail_sphere, X_L2, BAR_W, True)
    for ri, lbl in enumerate([MOD_LABELS[m] for m in MOD_ORDER]):
        ax.text(X_L1 - 0.01, Y_BOT + ri * row_total_h + row_h / 2, lbl,
                transform=ax.transAxes, ha="right", va="center", fontsize=6.8, color="#333333")
    ax.text(X_L1 - 0.01, STRIP_Y + STRIP_H / 2, "Diagnosis", transform=ax.transAxes,
            ha="right", va="center", fontsize=6.8, color="#333333", fontstyle="italic")
    title_y = STRIP_Y + STRIP_H + 0.015
    ax.text(X_L1 + BAR_W / 2, title_y, "Real", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8, fontweight="bold", color="#2166AC")
    ax.text(X_L2 + BAR_W / 2, title_y, "SPHERE", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8, fontweight="bold", color="#D73027")
    diag_counts = Counter(pid_diag.values())
    ax.legend(handles=[mpatches.Patch(color=DIAG_COLORS[d], label=f"{d}  ({diag_counts.get(d, 0)})")
                       for d in DIAG_ORDER if d in pid_diag.values()],
              loc="lower center", bbox_to_anchor=(0.5, 0.96), fontsize=7.5, frameon=False,
              ncol=4, handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    ax.text(0.5, 1.19, "Data Availability by Participant  (n = 644)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#222222")

    # Panel B
    ax = ax_B
    ax.set_title("Distributional fidelity", fontsize=8.5, fontweight="bold", pad=16)
    _panel_label(ax, "b", fig)
    # Single-hue Purples ramp: signals a related group of fidelity sub-metrics and
    # steps out of the qualitative modality wheel, so no cross-panel hex collision.
    FIDMET = {"ks": ("KS fidelity", "#54278f"), "mean": ("ΔMean", "#756bb1"),
              "var": ("ΔVariance", "#9e9ac8"), "corr": ("ΔCorrelation", "#cbc9e2")}
    mods_c = [m for m in MOD_ORDER[1:] if m in fidelity]
    x_pos = np.arange(len(mods_c)); bw = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bw
    for ki, key in enumerate(["ks", "mean", "var", "corr"]):
        ax.bar(x_pos + offsets[ki], [fidelity[m][key] for m in mods_c], width=bw,
               color=FIDMET[key][1], alpha=0.85, label=FIDMET[key][0])
    comp_vals = [fidelity[m]["composite"] for m in mods_c]
    ax.scatter(x_pos, comp_vals, marker="D", s=18, color="#333", zorder=5,
               label="Composite", linewidths=0.6, edgecolors="white")
    for xi, cv in zip(x_pos, comp_vals):
        ax.text(xi, 101.5, f"{cv:.0f}", ha="center", va="bottom", fontsize=6.8, color="#333", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MOD_LABELS[m].replace(" ", "\n") for m in mods_c], fontsize=8.5)
    ax.set_xlim(-0.55, len(mods_c) - 0.45)
    ax.set_ylabel("Score (%)", fontsize=8); ax.set_ylim(0, 112)
    ax.axhline(100, color="#aaa", lw=0.7, ls="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.0, frameon=False, bbox_to_anchor=(0.5, 1.01), loc="lower center",
              ncol=5, handlelength=0.8, handletextpad=0.4, columnspacing=0.8)

    # Panel C: two-anchor privacy at k=release_k (fig3 g/h style)
    # Three x-columns (Real=0 / SPHERE=mean / ColShuf.=100), three metric markers
    # per column. Value labels + error bars only on the SPHERE column. Numbers are
    # the cross-modality mean±SD from the canonical CSV (== Supplementary Table S2).
    ax = ax_C
    ax.set_title("Privacy", fontsize=8.5, fontweight="bold", pad=3)
    _panel_label(ax, "c", fig)
    _REAL_COL, _SPHERE_COL, _COLSHUF_COL = "#2166AC", "#D73027", "#1A9850"
    anchor_val = {"Real": {m: 0.0 for m in PRIV_METRICS},
                  "SPHERE": priv_mean,
                  "ColShuf.": {m: 100.0 for m in PRIV_METRICS}}
    anchor_sd = {"SPHERE": priv_sd}
    _pos = {"Real": 0, "SPHERE": 1, "ColShuf.": 2}
    _col = {"Real": _REAL_COL, "SPHERE": _SPHERE_COL, "ColShuf.": _COLSHUF_COL}
    _mkr = {"singling_out": "o", "linkability": "s", "inference": "D"}
    _off = {"singling_out": -0.15, "linkability": 0.0, "inference": 0.15}
    _lbl = {"singling_out": "Singling-out", "linkability": "Linkability",
            "inference": "Inference"}
    ax.axhline(100, color="#aaa", ls="--", lw=0.6)
    ax.axhline(0, color="#aaa", ls="--", lw=0.6)
    for meth in ["Real", "SPHERE", "ColShuf."]:
        for m in PRIV_METRICS:
            val = anchor_val[meth][m]
            if val != val:
                continue
            xp = _pos[meth] + _off[m]
            e = anchor_sd.get(meth, {}).get(m, 0.0)
            if e > 0:
                ax.errorbar(xp, val, yerr=e, fmt="none", ecolor=_col[meth],
                            elinewidth=0.9, capsize=2, zorder=4)
            ax.scatter(xp, val, marker=_mkr[m], c=_col[meth], s=34, zorder=5,
                       edgecolors="white", linewidths=0.5)
            if meth == "SPHERE":
                va = "bottom" if val >= 99 else "top"
                yo = (e + 3.0) if val >= 99 else -(e + 3.0)
                ax.text(xp, val + yo, f"{val:.0f}", ha="center", va=va, fontsize=6,
                        color=_col[meth], fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Real", "SPHERE", "ColShuffle"], fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("Score (%)", fontsize=8)
    ax.set_ylim(-10, 120); ax.set_xlim(-0.5, 2.5)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(handles=[Line2D([0], [0], marker=_mkr[m], color="w",
                              markerfacecolor="#555", markersize=5,
                              markeredgecolor="#333", markeredgewidth=0.4,
                              label=_lbl[m]) for m in PRIV_METRICS],
              loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=7,
              frameon=True, handletextpad=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Panels D & E
    for ax, field, title, xlabel, letter in [
        (ax_D, "beta_corr", "β-coefficient preservation", "β-correlation (%)", "d"),
        (ax_E, "pval_corr", "p-value preservation", r"Pearson $r$ of $-\log_{10}(p)$ (%)", "e")]:
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
        _panel_label(ax, letter, fig)
        mods = [m for m in MOD_ORDER if m in utility_data and utility_data[m][field] is not None]
        vals = [utility_data[m][field] for m in mods]
        ypos = np.arange(len(mods))
        bars = ax.barh(ypos, vals, color=[MOD_COLORS[m] for m in mods], alpha=0.85, height=0.65)
        ax.set_yticks(ypos); ax.set_yticklabels([MOD_LABELS[m] for m in mods], fontsize=7.5)
        ax.set_xlabel(xlabel, fontsize=8); ax.set_xlim(0, 103)
        ax.axvline(100, color="#aaa", lw=0.6, ls="--")
        for bar, val in zip(bars, vals):
            ax.text(min(val - 0.5, 101), bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                    va="center", ha="right", fontsize=6.8, color="#222")
        ax.spines[["top", "right"]].set_visible(False); ax.invert_yaxis()

    # Panels F & G — full modality names (match panels a–e), both axes 45°, small font
    cm_labels = [MOD_LABELS[m] for m in CROSS_MODS]
    for ax, data, title, letter in [
        (ax_F, heat_s, "Cross-modality correlation preservation", "f"),
        (ax_G, heat_p, "Cross-modality p-value preservation", "g")]:
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
        _panel_label(ax, letter, fig)
        cmap = plt.get_cmap(fg_cmap).copy(); cmap.set_bad(color="#cccccc")
        im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, vmin=0, vmax=100,
                       interpolation="none")
        ax.set_xticks(range(n_cm))
        ax.set_xticklabels(cm_labels, fontsize=6.5, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(n_cm))
        ax.set_yticklabels(cm_labels, fontsize=6.5, rotation=45, ha="right", va="center", rotation_mode="anchor")
        for i in range(n_cm):
            for j in range(n_cm):
                v = data[i, j]
                if np.isfinite(v):
                    r, g, b, _ = cmap(v / 100.0)
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    tcol = "white" if lum < 0.5 else "black"
                else:
                    tcol = "black"
                ax.text(j, i, f"{v:.0f}" if np.isfinite(v) else "N/A", ha="center", va="center",
                        fontsize=6.2, color=tcol)
        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=5.5)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nFigure saved:\n  {out_base.with_suffix('.pdf')}\n  {out_base.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser(description="Draw the composite ADRC figure from the step outputs")
    ap.add_argument("--availability-csv", type=Path, required=True,
                    help="Step 6 output (panel a)")
    ap.add_argument("--fidelity-csv", type=Path, required=True,
                    help="Step 7 output (panel b)")
    ap.add_argument("--privacy-csv", type=Path, required=True,
                    help="Step 5 output (panel c)")
    ap.add_argument("--validation-js", type=Path, required=True,
                    help="Step 4 output (panels d, e)")
    ap.add_argument("--crossmod-csv", type=Path, required=True,
                    help="Step 8 output (panels f, g)")
    ap.add_argument("--out", type=Path, default=FIG_DIR / "adrc_figure",
                    help="Output basename; .pdf and .png are written")
    ap.add_argument("--release-k", type=int, default=RELEASE_K)
    ap.add_argument("--fg-cmap", type=str, default="Blues")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render_figure(args.availability_csv, args.fidelity_csv, args.privacy_csv,
                  args.validation_js, args.crossmod_csv, args.out,
                  release_k=args.release_k, fg_cmap=args.fg_cmap)


if __name__ == "__main__":
    main()

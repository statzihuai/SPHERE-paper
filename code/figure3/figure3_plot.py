#!/usr/bin/env python3
"""
Render Figure 3 (UK Biobank WGS + proteomics) from pre-computed results.

This is the default entry point — it does not require the SPHERE
implementation or raw UKB data. Steps 1–10 produce the same intermediate
files from scratch; this script only reads and renders them.

Input files (under --source-dir):
    gw_pca_sub.npz         real (n,2), synth (n,2), ancestry (n,)
    gw_eigenvalues.npz     real (40,), synth (40,)
    gwas_miami.npz         chr (m,), pos (m,), nlp_real (m,), nlp_synth (m,)
    prot_pca_sub.npz       real (n,2), synth (n,2), labels (n,)
    pwas.npz               beta_real, nlp_real, beta_synth, nlp_synth
    geno_privacy.json      per-draw cluster privacy (Anonymeter)
    prot_privacy.json      per-draw proteomics privacy (Anonymeter)

    python3 figure3_plot.py --source-dir code/figure3/ --out-stem fig3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ── Style constants ──────────────────────────────────────────────────────────
REAL_COL = "#2166ac"
SPHERE_COL = "#b2182b"
COLSHUF_COL = "#888888"
FS, FST, FSL = 6, 7, 5

MIAMI_CAP = 50
CHR_COLS = ["#2166ac", "#92c5de"]
GW_THRESH = -np.log10(5e-8)

MET = ("singling_out", "linkability", "inference")
MET_LABEL = {"singling_out": "Singling-out", "linkability": "Linkability",
             "inference": "Inference"}
MET_MKR = {"singling_out": "o", "linkability": "s", "inference": "D"}
MET_OFF = {"singling_out": -0.15, "linkability": 0.0, "inference": 0.15}
PRIV_YLIM = (-10.0, 120.0)
PRIV_YTICKS = [0, 20, 40, 60, 80, 100]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FS, length=3, width=0.6)


# ── Panel drawers ────────────────────────────────────────────────────────────

def panel_pca_scatter(ax, scores, labels, title, legend=True):
    """PCA scatter coloured by ancestry cluster (panels a, b)."""
    cmap = plt.get_cmap("tab10")
    ulabs = sorted(np.unique(labels))
    for i, g in enumerate(ulabs):
        m = labels == g
        ax.scatter(scores[m, 0], scores[m, 1], s=0.1, alpha=0.3, rasterized=True,
                   color=cmap(i / max(len(ulabs) - 1, 1)), label=f"C{g}")
    ax.set_xlabel("PC 1", fontsize=FS)
    ax.set_ylabel("PC 2", fontsize=FS)
    ax.set_title(title, fontsize=FST, fontweight="bold", loc="left")
    _style(ax)
    if legend:
        ax.legend(fontsize=3, markerscale=6, ncol=3, loc="upper right",
                  handletextpad=0.2, columnspacing=0.5, frameon=False)


def panel_eigenvalues(ax, ev_real, ev_synth, title):
    """Eigenvalue comparison on a 1:1 line (panel c)."""
    lo = min(ev_real.min(), ev_synth.min()) * 0.9
    hi = max(ev_real.max(), ev_synth.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], ls="--", color="#aaa", lw=0.6, zorder=1)
    ax.scatter(ev_real, ev_synth, s=12, c=SPHERE_COL, edgecolors="white",
               linewidths=0.3, zorder=3)
    rel = np.abs(ev_synth - ev_real) / ev_real
    ax.set_xlabel("Real eigenvalue", fontsize=FS)
    ax.set_ylabel("SPHERE eigenvalue", fontsize=FS)
    ax.set_title(title, fontsize=FST, fontweight="bold", loc="left")
    ax.text(0.05, 0.92, f"max rel diff = {rel.max():.2e}",
            transform=ax.transAxes, fontsize=FSL, color="#555")
    _style(ax)


def panel_miami(ax, ci, pos, nlpr, nlps, n_ind):
    """Miami plot: real on top, SPHERE mirrored below (panel d)."""
    nlpr_c = np.minimum(nlpr, MIAMI_CAP)
    nlps_c = np.minimum(nlps, MIAMI_CAP)
    uchr = sorted(np.unique(ci))
    offsets = {}
    x = np.empty(len(ci), float)
    cur = 0.0
    for c in uchr:
        m = ci == c
        mn, mx = pos[m].min(), pos[m].max()
        x[m] = pos[m] - mn + cur
        offsets[c] = cur + (mx - mn) / 2
        cur += mx - mn + 5e6
    for c in uchr:
        m = ci == c
        col = CHR_COLS[c % 2]
        ax.scatter(x[m], nlpr_c[m], s=0.3, c=col, alpha=0.5, rasterized=True)
        ax.scatter(x[m], -nlps_c[m], s=0.3, c=col, alpha=0.5, rasterized=True)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(GW_THRESH, color="grey", ls="--", lw=0.4)
    ax.axhline(-GW_THRESH, color="grey", ls="--", lw=0.4)
    for c in uchr:
        lab = str(c) if c % 2 == 1 or c > 18 else ""
        ax.text(offsets[c], -MIAMI_CAP - 4, lab, ha="center", fontsize=3.5,
                color="#555")
    n_r = int((nlpr > GW_THRESH).sum())
    n_s = int((nlps > GW_THRESH).sum())
    n_b = int(((nlpr > GW_THRESH) & (nlps > GW_THRESH)).sum())
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=FS)
    ax.set_title(f"d  AD GWAS (n = {n_ind:,}): real ↑ / SPHERE ↓ — "
                 f"{n_r}/{n_s}/{n_b} hits",
                 fontsize=FST, fontweight="bold", loc="left")
    ax.set_ylim(-MIAMI_CAP - 8, MIAMI_CAP + 5)
    ax.set_xlim(x.min() - 1e7, x.max() + 1e7)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(labelsize=FS, length=3, width=0.6)


def panel_volcano(ax, beta, neg_log10p, n_prot, side, colour, label, ymax):
    """PWAS volcano: protein ~ age (panels f, g)."""
    bonf = -np.log10(0.05 / n_prot)
    ax.scatter(beta, neg_log10p, s=1, c=colour, alpha=0.4, rasterized=True)
    ax.axhline(bonf, color="#aaa", ls="--", lw=0.5)
    n_sig = int((neg_log10p > bonf).sum())
    ax.set_xlabel("β (age)", fontsize=FS)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=FS)
    ax.set_title(f"{label}  PWAS (protein ~ age): {side}",
                 fontsize=FST, fontweight="bold", loc="left")
    ax.text(0.95, 0.92, f"{n_sig:,} sig", transform=ax.transAxes,
            fontsize=FSL, ha="right", color=colour)
    ax.set_ylim(-1, ymax)
    _style(ax)


def panel_privacy(ax, dist, title):
    """Two-anchor privacy panel (panels h, i).

    Real = 0 % and ColShuffle = 100 % are anchors by construction. Only the
    SPHERE column is a measurement; it is plotted as mean ± s.d. over the
    10 independent draws.
    """
    col = {"Real": REAL_COL, "SPHERE": SPHERE_COL, "ColShuf.": COLSHUF_COL}
    ax.axhline(100, color="#aaa", ls="--", lw=0.6)
    ax.axhline(0, color="#aaa", ls="--", lw=0.6)

    for meth in ("Real", "SPHERE", "ColShuf."):
        for m in MET:
            xp = {"Real": 0, "SPHERE": 1, "ColShuf.": 2}[meth] + MET_OFF[m]
            if meth != "SPHERE":
                val = 0.0 if meth == "Real" else 100.0
                ax.scatter(xp, val, marker=MET_MKR[m], c=col[meth], s=30,
                           zorder=5, edgecolors="white", linewidths=0.4)
                continue
            d = dist[m]
            val, e = float(d.mean()), float(d.std(ddof=0))
            if e > 0:
                ax.errorbar(xp, val, yerr=e, fmt="none", ecolor=SPHERE_COL,
                            elinewidth=0.9, capsize=2, zorder=4)
            ax.scatter(xp, val, marker=MET_MKR[m], c=SPHERE_COL, s=30,
                       zorder=5, edgecolors="white", linewidths=0.4)
            va = "bottom" if val >= 99 else "top"
            yo = (e + 3.0) if val >= 99 else -(e + 3.0)
            ax.text(xp, val + yo, f"{val:.0f}", ha="center", va=va,
                    fontsize=FSL, color=SPHERE_COL, fontweight="bold")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Real", "SPHERE", "ColShuffle"], fontsize=FS)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(*PRIV_YLIM)
    ax.set_yticks(PRIV_YTICKS)
    ax.set_ylabel("Score (%)", fontsize=FS)
    ax.set_title(title, fontsize=FST, fontweight="bold", loc="left")
    ax.legend(handles=[Line2D([0], [0], marker=MET_MKR[m], color="w",
                              markerfacecolor="#555", markersize=5,
                              markeredgecolor="#333", markeredgewidth=0.4,
                              label=MET_LABEL[m]) for m in MET],
              loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=FSL,
              frameon=True, handletextpad=0.3)
    _style(ax)


# ── Data loaders ─────────────────────────────────────────────────────────────

def _nan_clean(v):
    v = np.asarray(v, float)
    return v[np.isfinite(v)]


def load_geno_privacy(path):
    """Load genotype privacy: per-draw means across 12 ancestry clusters."""
    d = json.load(open(path))
    cells = [j for j in d["per_job"] if j["unit"] >= 0]
    draws = sorted({j["draw"] for j in cells})
    out = {}
    for m in MET:
        dv = []
        for k in draws:
            vv = _nan_clean([j[m] for j in cells if j["draw"] == k])
            dv.append(float(vv.mean()) if len(vv) else np.nan)
        out[m] = _nan_clean(dv)
    return out


def load_prot_privacy(path):
    """Load proteomics privacy: per-run scores.

    Reads the output of step8_privacy_proteomics.py, which stores per-run
    values under ``per_run`` and aggregates under ``summary``.
    """
    d = json.load(open(path))
    runs = d.get("per_run", d)
    if isinstance(runs, dict):
        runs = runs.get("per_run", [])
    out = {}
    for m in MET:
        out[m] = _nan_clean([r[m] for r in runs if m in r])
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Render Figure 3 from pre-computed source data")
    ap.add_argument("--source-dir", required=True, type=Path,
                    help="directory with panel data files (see docstring)")
    ap.add_argument("--out-stem", default="fig3")
    ap.add_argument("--out-dir", default=".", type=Path)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    S = args.source_dir

    # ── Load ─────────────────────────────────────────────────────────────
    pca = np.load(S / "gw_pca_sub.npz")
    sr, ss, anc = pca["real"], pca["synth"], pca["ancestry"]
    n_ind = len(sr)

    eig = np.load(S / "gw_eigenvalues.npz")
    ev_real, ev_synth = eig["real"], eig["synth"]

    gw = np.load(S / "gwas_miami.npz")
    ci, pos = gw["chr"], gw["pos"].astype(float)
    nlpr, nlps = gw["nlp_real"], gw["nlp_synth"]

    ppca = np.load(S / "prot_pca_sub.npz")
    pwas = np.load(S / "pwas.npz")

    geno_dist = load_geno_privacy(S / "geno_privacy.json")
    prot_dist = load_prot_privacy(S / "prot_privacy.json")

    # ── Draw ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.2, 10.2))
    gs = fig.add_gridspec(4, 12, left=0.075, right=0.985, top=0.97, bottom=0.05,
                          hspace=0.5, wspace=2.6,
                          height_ratios=[0.95, 0.8, 0.95, 0.95])

    # Row 1: genotype PCA
    panel_pca_scatter(fig.add_subplot(gs[0, 0:4]), sr, anc,
                      "a  Genotype PCA: real")
    panel_pca_scatter(fig.add_subplot(gs[0, 4:8]), ss, anc,
                      "b  Genotype PCA: SPHERE", legend=False)
    panel_eigenvalues(fig.add_subplot(gs[0, 8:12]), ev_real, ev_synth,
                      f"c  PCA eigenvalues ({len(ev_real)} PCs)")

    # Row 2: GWAS Miami (full width)
    panel_miami(fig.add_subplot(gs[1, 0:12]), ci, pos, nlpr, nlps, n_ind)

    # Row 3: proteomics
    ax_e = fig.add_subplot(gs[2, 0:4])
    pr, ps = ppca["real"], ppca["synth"]
    cmap_e = [REAL_COL if l == 0 else SPHERE_COL for l in ppca["labels"]]
    idx_e = np.random.RandomState(42).permutation(len(pr))
    ax_e.scatter(np.vstack([pr, ps])[idx_e, 0],
                 np.vstack([pr, ps])[idx_e, 1],
                 c=[cmap_e[i] for i in idx_e], s=2, alpha=0.3, rasterized=True)
    ax_e.set_xlabel("PC 1", fontsize=FS)
    ax_e.set_ylabel("PC 2", fontsize=FS)
    ax_e.set_title("e  Proteomics PCA", fontsize=FST, fontweight="bold",
                   loc="left")
    ax_e.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=REAL_COL,
               markersize=4, label="Real"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SPHERE_COL,
               markersize=4, label="SPHERE")],
        fontsize=FSL, frameon=False, loc="upper right", handletextpad=0.3)
    _style(ax_e)

    n_prot = len(pwas["beta_real"])
    ymax = max(pwas["nlp_real"].max(), pwas["nlp_synth"].max()) * 1.05
    panel_volcano(fig.add_subplot(gs[2, 4:8]),
                  pwas["beta_real"], pwas["nlp_real"], n_prot,
                  "real", REAL_COL, "f", ymax)
    panel_volcano(fig.add_subplot(gs[2, 8:12]),
                  pwas["beta_synth"], pwas["nlp_synth"], n_prot,
                  "SPHERE", SPHERE_COL, "g", ymax)

    # Row 4: privacy
    panel_privacy(fig.add_subplot(gs[3, 0:6]), geno_dist,
                  "h  Genotype privacy")
    panel_privacy(fig.add_subplot(gs[3, 6:12]), prot_dist,
                  "i  Proteomics privacy")

    # ── Save ─────────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / args.out_stem
    fig.savefig(f"{stem}.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {stem}.png\n[save] {stem}.pdf")

    # ── Stats JSON ───────────────────────────────────────────────────────
    n_r = int((nlpr > GW_THRESH).sum())
    n_s = int((nlps > GW_THRESH).sum())
    n_b = int(((nlpr > GW_THRESH) & (nlps > GW_THRESH)).sum())
    rel_eig = np.abs(ev_synth - ev_real) / ev_real

    def _ps(d):
        return {m: {"mean": round(float(d[m].mean()), 1),
                     "sd": round(float(d[m].std(ddof=0)), 1),
                     "n": len(d[m])} for m in MET}

    stats = {
        "n_individuals": n_ind,
        "n_variants": int(len(ci)),
        "gwas_hits_real_synth_both": [n_r, n_s, n_b],
        "n_pc": int(len(ev_real)),
        "max_rel_d_eigval": float(rel_eig.max()),
        "panel_h": _ps(geno_dist),
        "panel_i": _ps(prot_dist),
    }
    with open(f"{stem}.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

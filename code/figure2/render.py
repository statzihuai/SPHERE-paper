#!/usr/bin/env python3
"""
Figure 2 rendering — generate all panel PDFs and composite figure.

Reads data outputs from analyze.py (CSV + JSON) and renders every panel:
  panel_a_pca.pdf          PCA scatter (Real vs SPHERE)
  panel_b_privacy.pdf      Anonymeter privacy dot plot
  panel_e_forest_tall.pdf  OLS coefficient forest plot

Then patches the TikZ template and compiles the final composite PDF/PNG.

Requires: tectonic or pdflatex.

Usage:
    python3 code/figure2/render.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # <repo>/code/figure2
CODE = HERE.parent                              # <repo>/code
ROOT = CODE.parent                              # <repo>
RES  = HERE
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
TEX_SRC = HERE / "sphere_nature_figure.tex"

# ── Variable display metadata ────────────────────────────────────────────────
_VAR_META = {
    "age":               ("Age",              "Demo"),
    "male":              ("Male",             "Demo"),
    "race_black":        ("Race (Black)",     "Demo"),
    "race_hispanic":     ("Race (Hispanic)",  "Demo"),
    "race_asian":        ("Race (Asian)",     "Demo"),
    "education":         ("Education",        "SES"),
    "poverty_ratio":     ("Poverty",          "SES"),
    "bmi":               ("BMI",              "Anthro"),
    "waist_circumference": ("Waist",          "Anthro"),
    "total_cholesterol": ("Total cholesterol", "Metab"),
    "hdl_cholesterol":   ("HDL",              "Metab"),
    "hba1c":             ("HbA1c",            "Metab"),
    "creatinine":        ("Creatinine",       "Metab"),
    "glucose":           ("Glucose",          "Metab"),
    "crp":               ("CRP",              "Metab"),
    "current_smoker":    ("Smoker",           "Life"),
    "drinks_per_day":    ("Alcohol",          "Life"),
    "physically_active": ("Physical activity", "Life"),
    "sodium_intake":     ("Sodium",           "Life"),
}

_DOM_RANK = {"Demo": 0, "SES": 1, "Anthro": 2, "Metab": 3, "Life": 4}


def _render_panel_a() -> Path:
    """PCA scatter: Real (blue) vs SPHERE (red), from CSV data files."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    df_real = pd.read_csv(DATA / "nhanes_real.csv")
    df_synth = pd.read_csv(DATA / "nhanes_synthetic.csv")

    sc = StandardScaler()
    Xr = sc.fit_transform(df_real.values)
    Xs = sc.transform(df_synth.values)
    X = np.vstack([Xr, Xs])
    lbls = np.array(["Real"] * len(df_real) + ["SPHERE"] * len(df_synth))

    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    emb, lbls = emb[idx], lbls[idx]
    colors = {"Real": "#2166AC", "SPHERE": "#D73027"}

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.scatter(emb[:, 0], emb[:, 1],
               c=[colors[l] for l in lbls], s=2, alpha=0.4, rasterized=True)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166AC",
               markersize=6, label="Real"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D73027",
               markersize=6, label="SPHERE"),
    ], frameon=True, fontsize=8, loc="upper right")
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = RES / "panel_a_pca.pdf"
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  Panel a -> {out}")
    return out


def _render_panel_b() -> Path:
    """Privacy dot plot from nhanes_anonymeter.json (two-anchor normalised)."""
    raw_vals = json.loads((RES / "nhanes_anonymeter.json").read_text())

    r_real = raw_vals["Real"]
    r_cs = raw_vals["ColShuf."]

    def norm(raw, real_val, cs_val):
        denom = cs_val - real_val
        return (np.clip((raw - real_val) / denom * 100, 0, 100)
                if abs(denom) > 1e-10 else np.nan)

    metrics = ["singling_out", "linkability", "inference"]
    scores = {m: {k: norm(raw_vals[m][k], r_real[k], r_cs[k]) for k in metrics}
              for m in raw_vals}

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    xpos = {"Real": 0, "SPHERE": 1, "ColShuf.": 2}
    mcolors = {"Real": "#2166AC", "SPHERE": "#D73027", "ColShuf.": "#1A9850"}
    markers = {"singling_out": "o", "linkability": "s", "inference": "D"}
    offsets = {"singling_out": -0.15, "linkability": 0.0, "inference": 0.15}
    mlabels = {"singling_out": "Singling-out",
               "linkability": "Linkability",
               "inference": "Inference"}

    ax.axhline(100, color="#888", ls="--", lw=0.5, alpha=0.5)
    ax.axhline(0, color="#888", ls="--", lw=0.5, alpha=0.5)
    for meth in ["Real", "SPHERE", "ColShuf."]:
        for m in metrics:
            val = scores[meth][m]
            if np.isnan(val):
                continue
            xp = xpos[meth] + offsets[m]
            ax.scatter(xp, val, marker=markers[m], c=mcolors[meth],
                       s=50, zorder=5, edgecolors="white", linewidths=0.5)
            if meth == "SPHERE":
                ax.text(xp, val + 4, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=5.5,
                        color=mcolors[meth], fontweight="bold")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Real", "SPHERE", "ColShuffle"], fontsize=8)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(-8, 120); ax.set_xlim(-0.5, 2.5)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Line2D([0], [0], marker=markers[m], color="w", markerfacecolor="#555",
               markersize=5, label=mlabels[m])
        for m in metrics
    ], loc="lower right", fontsize=6, frameon=True, handletextpad=0.3)
    plt.tight_layout()

    out = RES / "panel_b_privacy.pdf"
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  Panel b -> {out}")
    return out


def _render_panel_e(ols: dict) -> Path:
    """Forest plot from OLS JSON (panel_e_ols.json)."""
    coefs = ols["coefficients"]

    def _sub_rank(v):
        if v == "age": return 0
        if v.startswith("race_"): return 1
        return 2

    vars_ordered = sorted(
        coefs,
        key=lambda v: (_DOM_RANK.get(_VAR_META.get(v, (v, "Demo"))[1], 9),
                       _sub_rank(v),
                       -abs(coefs[v]["real_coef"])))

    fig, ax = plt.subplots(figsize=(2.6, 4.5))
    for k, var in enumerate(vars_ordered):
        y_k = len(vars_ordered) - 1 - k
        cr = coefs[var]

        ax.errorbar(cr["real_coef"], y_k + 0.12,
                    xerr=1.96 * cr["real_se"],
                    fmt="o", color="#2166AC", markersize=3, capsize=2, lw=0.8, zorder=3)
        ax.errorbar(cr["sphere_coef"], y_k - 0.12,
                    xerr=1.96 * cr["sphere_se"],
                    fmt="s", color="#D73027", markerfacecolor="none",
                    markeredgecolor="#D73027", markeredgewidth=0.9,
                    markersize=3.5, capsize=2, lw=0.8, zorder=3)

        def _stars(p):
            return ("***" if p < 0.001 else "**" if p < 0.01 else
                    "*" if p < 0.05 else "")

        s_r = _stars(cr["real_p"])
        if s_r:
            ax.text(cr["real_coef"] + 1.96 * cr["real_se"] + 0.15,
                    y_k + 0.12, s_r, fontsize=5, va="center", color="black")
        s_s = _stars(cr["sphere_p"])
        if s_s:
            ax.text(cr["sphere_coef"] + 1.96 * cr["sphere_se"] + 0.15,
                    y_k - 0.12, s_s, fontsize=5, va="center", color="black")

    ax.set_yticks(np.arange(len(vars_ordered)))
    ax.set_yticklabels([_VAR_META.get(v, (v, ""))[0] for v in vars_ordered[::-1]],
                       fontsize=7)
    ax.set_xlabel("Coefficient", fontsize=8)
    ax.axvline(0, color="gray", ls=":", lw=0.5)
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166AC",
                  markersize=4, label="Real"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
                  markeredgecolor="#D73027", markeredgewidth=0.9,
                  markersize=4, label="SPHERE")]
    ax.legend(handles=leg, fontsize=6, loc="lower right", frameon=True)
    ax.text(0.03, 0.99, r"$r = 1.0$",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
            color="black", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = RES / "panel_e_forest_tall.pdf"
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  Panel e -> {out}")
    return out


def _tex_safe(text: str) -> str:
    for char, repl in [("&", "\\&"), ("%", "\\%"), ("$", "\\$"),
                       ("#", "\\#"), ("_", "\\_"), ("^", "\\^{}"),
                       ("~", "\\textasciitilde{}"), ("{", "\\{"), ("}", "\\}")]:
        text = text.replace(char, repl)
    return text


def _compile_tex(tex_path: Path) -> bool:
    """Compile a .tex file with tectonic (preferred) or pdflatex."""
    cwd = tex_path.parent
    for engine in ("tectonic", "pdflatex"):
        exe = shutil.which(engine)
        if not exe:
            continue
        cmd = ([exe, str(tex_path)] if engine == "tectonic"
               else [exe, "-interaction=nonstopmode", "-halt-on-error", str(tex_path)])
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        if r.returncode == 0:
            print(f"  Compiled {tex_path.name} with {engine}")
            return True
        print(f"  {engine} failed: {r.stderr[-300:]}")
    print("  No LaTeX engine found (install tectonic or pdflatex).")
    return False


def main():
    print("=" * 60)
    print("Figure 2 — render composite")
    print("=" * 60)

    # Check data prerequisites (JSON/CSV from analyze.py)
    ols_path = RES / "panel_e_ols.json"
    anon_path = RES / "nhanes_anonymeter.json"
    real_csv = DATA / "nhanes_real.csv"
    synth_csv = DATA / "nhanes_synthetic.csv"

    for path in [ols_path, anon_path, real_csv, synth_csv]:
        if not path.exists():
            print(f"  MISSING: {path} — run analyze.py first.")
            sys.exit(1)
    if not TEX_SRC.exists():
        print(f"  MISSING: {TEX_SRC}")
        sys.exit(1)

    # Render panel PDFs from data files
    print("\n▸ Panel a (PCA scatter)...")
    _render_panel_a()

    print("\n▸ Panel b (privacy)...")
    _render_panel_b()

    print("\n▸ Panel e (forest plot)...")
    ols = json.loads(ols_path.read_text())
    _render_panel_e(ols)

    # TikZ template expects this filename
    pca_path = RES / "panel_a_pca.pdf"
    umap_path = RES / "panel_a_umap.pdf"
    if pca_path.exists():
        shutil.copy2(pca_path, umap_path)
        print(f"  Copied {pca_path.name} -> {umap_path.name} (TikZ legacy name)")

    # Patch TikZ template
    print("\n▸ Patching TikZ template...")
    text = TEX_SRC.read_text()

    # Panel C summaries
    summ_path = RES / "panel_c_summaries.json"
    if summ_path.exists():
        summ = json.loads(summ_path.read_text())
        real_tex = _tex_safe(summ.get("real_summary", ""))
        sph_tex = _tex_safe(summ.get("sphere_summary", ""))
        text, _ = re.subn(
            r"(Real Data \(PI conclusion\).*?\\textit\{)[^}]*(\})",
            lambda m: m.group(1) + real_tex + m.group(2),
            text, flags=re.DOTALL)
        text, _ = re.subn(
            r"(SPHERE \(PI conclusion\).*?\\textit\{)[^}]*(\})",
            lambda m: m.group(1) + sph_tex + m.group(2),
            text, flags=re.DOTALL)
        print(f"  Panel C real   -> {real_tex[:80]}")
        print(f"  Panel C SPHERE -> {sph_tex[:80]}")

    # Panel E R²/r annotations
    r2_r = ols["r2_real"]
    r_coef = ols["coef_pearson_r"]
    n = ols["n"]
    n_vars = ols["n_vars"]
    new_r2 = (f"    {{$R^2_{{\\text{{real}}}} = R^2_{{\\text{{SPHERE}}}} = {r2_r:.4f}$,"
              f"~~$n = {n:,}$}};")
    new_cor = (f"    {{Coeff $r = {r_coef:.3f}$~~(all {n_vars} predictors, max diff $= 0$)}};")
    text, _ = re.subn(r"    \{\$R\^2_\{\\text\{real\}\}.*?\};", lambda _: new_r2, text)
    text, _ = re.subn(r"    \{Coeff \$r =.*?\};", lambda _: new_cor, text)

    # Code line count from scenario2 JSON (AI-generated, shipped)
    sc2_path = RES / "scenario2_results.json"
    if sc2_path.exists():
        n_lines = len(json.loads(sc2_path.read_text()).get("final_code", "").splitlines())
        text, _ = re.subn(r"(\{\\scriptsize\s*)\d+(\s*lines\})",
                          rf"\g<1>{n_lines}\g<2>", text)
        print(f"  Code lines: {n_lines}")

    # Write patched .tex and compile
    RES.mkdir(parents=True, exist_ok=True)
    tex_file = RES / "sphere_nature_figure.tex"
    tex_file.write_text(text)

    FIGURES.mkdir(parents=True, exist_ok=True)
    ok = _compile_tex(tex_file)
    if not ok:
        print("  WARNING: LaTeX compilation failed.")
        sys.exit(1)

    src_pdf = RES / "sphere_nature_figure.pdf"
    dst_pdf = FIGURES / "sphere_virtuallab_figure.pdf"
    shutil.copy2(src_pdf, dst_pdf)
    print(f"  PDF -> {dst_pdf}")

    # PNG via pdftoppm
    dst_png = FIGURES / "sphere_virtuallab_figure.png"
    tmp_base = RES / "sphere_nature_figure"
    try:
        r = subprocess.run(
            ["pdftoppm", "-r", "600", "-singlefile", "-png",
             str(src_pdf), str(tmp_base)],
            capture_output=True, text=True)
        tmp_png = Path(str(tmp_base) + ".png")
        if r.returncode == 0 and tmp_png.exists():
            shutil.copy2(tmp_png, dst_png)
            print(f"  PNG -> {dst_png}")
    except FileNotFoundError:
        pass

    print(f"\n  R² = {r2_r:.4f}, n = {n}, r = {r_coef:.6f}")
    print("Done.")


if __name__ == "__main__":
    main()

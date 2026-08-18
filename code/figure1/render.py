#!/usr/bin/env python3
"""
Figure 1 — 7-panel benchmark figure for SPHERE.

Reads five panel CSVs from code/figure1/ and draws the composite figure.
To regenerate those CSVs from raw data, run the per-panel scripts (panel_b..f).

Usage:
    python3 code/figure1/render.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                       # headless backend; set before pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ── Portable paths ─────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent               # <repo>/code/figure1
CODE = HERE.parent                                   # <repo>/code
ROOT = CODE.parent                                   # <repo>
RES  = HERE                                          # panel CSVs (written by panel_*.py)
FIGS = ROOT / 'figures'
DATA = ROOT / 'data'                                 # raw inputs (panel scripts only)
FIGS.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# PLOT-side globals (from make_figures.py)
# ════════════════════════════════════════════════════════════════════════════════

# ── Nature-style rcParams ─────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          8,
    'axes.labelsize':     7.5,
    'axes.titlesize':     8,
    'axes.titleweight':   'bold',
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'legend.fontsize':    6.5,
    'legend.frameon':     False,
    'figure.dpi':         300,
    'axes.linewidth':     0.6,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.major.size':   2.5,
    'ytick.major.size':   2.5,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.linewidth':     0.3,
    'grid.alpha':         0.4,
    'grid.color':         '#cccccc',
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.02,
})


# ── Shared Nature-style panel label (lowercase bold, top-left corner) ──────────
def _panel_label(ax, letter, dx=-0.14, dy=1.06):
    """Place a lowercase bold panel letter just outside the top-left axes corner."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom', ha='left')


# ── Method order, display labels (short, for x-ticks), colours ─────────────────
METHODS_ORDER = [
    'Real', 'SPHERE', 'SPHERE-x2', 'ColShuffle', 'DP', 'MVN',
    'ROMM', 'Mixup-Beta(0.4)', 'Synthpop', 'CTGAN', 'TabDDPM', 'GaussianCopula',
]
METHOD_LABELS_SHORT = {      # single-line versions for x-tick labels
    'Real':            'Real',
    'SPHERE':          'SPHERE',
    'SPHERE-x2':       'SPHERE×2',
    'ColShuffle':      'ColShuffle',
    'DP':              'DP',
    'MVN':             'MVN',
    'ROMM':            'ROMM' + r'$^{\dagger}$',
    'Mixup-Beta(0.4)': 'Mixup-β',
    'Synthpop':        'Synthpop',
    'CTGAN':           'CTGAN' + r'$^{\dagger}$',
    'TabDDPM':         'TabDDPM' + r'$^{\dagger}$',
    'GaussianCopula':  'GaussianCopula' + r'$^{\dagger}$',
}
METHOD_COLORS = {
    'Real':            '#2166AC',
    'SPHERE':          '#D73027',
    'SPHERE-x2':       '#A50F15',
    'ColShuffle':      '#999999',
    'DP':              '#D95F02',
    'MVN':             '#1B9E77',
    'ROMM':            '#E7298A',
    'Mixup-Beta(0.4)': '#7570B3',
    'Synthpop':        '#A65628',
    'CTGAN':           '#E6AB02',
    'TabDDPM':         '#66A61E',
    'GaussianCopula':  '#984EA3',
}

C_SHADE  = '#D6E8F7'   # target-zone shading (80–100 %)
C_REF    = '#333333'   # reference dashed line
EPS      = 1e-8
R2_MIN   = 0.05        # exclude (dataset, model) pairs with |Real R²| < this

# ── Dataset domain classification ────────────────────────────────────────────
DATASET_DOMAIN = {
    'breast-w': 'Life & Health Sci.',
    'diabetes': 'Life & Health Sci.',
    'wdbc': 'Life & Health Sci.',
    'ilpd': 'Life & Health Sci.',
    'abalone': 'Life & Health Sci.',
    'abalone-2': 'Life & Health Sci.',
    'QSAR_fish_toxicity': 'Life & Health Sci.',
    'airfoil_self_noise': 'Phys. Sci. & Eng.',
    'concrete_compressive_strength': 'Phys. Sci. & Eng.',
    'grid_stability': 'Phys. Sci. & Eng.',
    'energy_efficiency': 'Phys. Sci. & Eng.',
    'ozone-level-8hr': 'Phys. Sci. & Eng.',
    'climate-model-simulation-crashes': 'Phys. Sci. & Eng.',
    'wilt': 'Phys. Sci. & Eng.',
    'kin8nm': 'Phys. Sci. & Eng.',
    'pumadyn32nh': 'Phys. Sci. & Eng.',
    'yprop_4_1': 'Phys. Sci. & Eng.',
    'spambase': 'Computer Science',
    'pc4': 'Computer Science',
    'pc3': 'Computer Science',
    'kc2': 'Computer Science',
    'kc1': 'Computer Science',
    'pc1': 'Computer Science',
    'cpu_activity': 'Computer Science',
    'cpu_act': 'Computer Science',
    'madelon': 'Computer Science',
    'student_performance_por': 'Social & Behav. Sci.',
    'space_ga': 'Social & Behav. Sci.',
    'eye_movements': 'Social & Behav. Sci.',
    'heloc': 'Social & Behav. Sci.',
    'phoneme': 'Social & Behav. Sci.',
    'white_wine': 'Food & Sensory Sci.',
    'red_wine': 'Food & Sensory Sci.',
    'wine_quality': 'Food & Sensory Sci.',
    'geographical_origin_of_music': 'Food & Sensory Sci.',
}

DOMAIN_ORDER = [
    'Life & Health Sci.',
    'Phys. Sci. & Eng.',
    'Computer Science',
    'Social & Behav. Sci.',
    'Food & Sensory Sci.',
]

DOMAIN_COLORS = {
    'Life & Health Sci.':   '#E64B35',
    'Phys. Sci. & Eng.':   '#4DBBD5',
    'Computer Science':     '#00A087',
    'Social & Behav. Sci.': '#F39B7F',
    'Food & Sensory Sci.':  '#8491B4',
}

DOMAIN_MARKERS = {
    'Life & Health Sci.':   'o',
    'Phys. Sci. & Eng.':   's',
    'Computer Science':     'D',
    'Social & Behav. Sci.': '^',
    'Food & Sensory Sci.':  'v',
}


# ── Dataset overview panel ────────────────────────────────────────────────────
def draw_dataset_overview(ax, df_any, title='Benchmark datasets'):
    """
    Scatter plot: x = n (log), y = p (log), colour/marker = domain.
    Shows the breadth of evaluation across sample sizes, dimensionalities,
    and scientific domains.
    """
    if df_any is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=7, color='#888888')
        ax.set_title(title, loc='left', fontweight='bold', pad=4, fontsize=8)
        return

    # Get unique datasets with n, p
    info = df_any.groupby('dataset').first()[['n', 'p']].reset_index()

    rng = np.random.default_rng(7)
    for domain in DOMAIN_ORDER:
        col = DOMAIN_COLORS[domain]
        marker = DOMAIN_MARKERS[domain]
        ds_in_domain = [d for d in DATASET_DOMAIN if DATASET_DOMAIN[d] == domain]
        sub = info[info['dataset'].isin(ds_in_domain)]
        if sub.empty:
            continue
        # Small jitter to avoid overlap on log scale
        jit_x = np.exp(rng.uniform(-0.03, 0.03, len(sub)))
        jit_y = np.exp(rng.uniform(-0.03, 0.03, len(sub)))
        ax.scatter(sub['n'].values * jit_x, sub['p'].values * jit_y,
                   c=col, marker=marker, s=28, alpha=0.85, zorder=4,
                   edgecolors='white', linewidths=0.4,
                   label=f'{domain} ({len(sub)})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample size  n', fontsize=7.5)
    ax.set_ylabel('Features  p', fontsize=7.5)
    ax.set_xlim(350, 15000)
    ax.set_ylim(2, 700)
    ax.set_xticks([500, 1000, 2000, 5000, 10000])
    ax.set_xticklabels(['500', '1k', '2k', '5k', '10k'])
    ax.set_yticks([3, 5, 10, 20, 50, 100, 500])
    ax.set_yticklabels(['3', '5', '10', '20', '50', '100', '500'])
    ax.set_title(title, loc='left', fontweight='bold', pad=4, fontsize=8)
    ax.legend(fontsize=4.8, loc='upper left', borderpad=0.4,
              labelspacing=0.25, handletextpad=0.3, markerscale=0.8)

    # Annotate total
    n_ds = len(info)
    n_domains = len(set(DATASET_DOMAIN.get(d, '?') for d in info['dataset']))
    ax.text(0.98, 0.02, f'N = {n_ds} datasets · {n_domains} domains',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=6, color='#555555')


# ── Load CSVs ─────────────────────────────────────────────────────────────────
def _load_csv(name):
    path = RES / name
    if path.exists():
        return pd.read_csv(path)
    return None


def load_panels():
    """Load the five panel CSVs (b-f) and return (df_b, df_c, df_d, df_e, df_f).

    df_d and df_e are collapsed over simulations via groupby.mean().
    """
    df_b = _load_csv('panel_b_privacy.csv')
    df_c = _load_csv('panel_c_fidelity.csv')
    df_d = _load_csv('panel_d_inference.csv')
    df_e = _load_csv('panel_e_utility.csv')
    df_f = _load_csv('panel_f_timing.csv')

    # Average over simulations
    if df_d is not None:
        df_d = df_d.groupby(['dataset', 'method']).mean(numeric_only=True).reset_index()
    if df_e is not None:
        df_e = df_e.groupby(['dataset', 'method']).mean(numeric_only=True).reset_index()
    return df_b, df_c, df_d, df_e, df_f


# ── Helper: wide pivot ────────────────────────────────────────────────────────
def pivot(df, col):
    """Wide pivot: dataset × method, mean over any duplicates."""
    if df is None:
        return pd.DataFrame()
    agg = df.groupby(['dataset', 'method'])[col].mean().reset_index()
    return agg.pivot(index='dataset', columns='method', values=col)


# ══════════════════════════════════════════════════════════════════════════════
# Per-method, per-dataset score functions  (return Series indexed by dataset)
# ══════════════════════════════════════════════════════════════════════════════

def _fidelity_scores(wide, method):
    """Moment-based fidelity: clip(100 − pct_delta, 0, 100).

    pct_delta is already a percentage deviation relative to the dataset's own
    scale factor, so Real data (pct_delta = 0) scores 100% and a method loses
    1 point per 1% of deviation, floored at 0%.
    """
    if wide.empty or method not in wide.columns:
        return pd.Series(dtype=float)
    w = wide.dropna(subset=[method])
    S = w[method].values.astype(float)
    sc = np.clip(100 - S, 0, 100)
    return pd.Series(sc, index=w.index)


def _ks_scores(wide, method):
    """KS fidelity: (1 − KS) × 100 — no Real reference needed."""
    if wide.empty or method not in wide.columns:
        return pd.Series(dtype=float)
    w = wide.dropna(subset=[method])
    sc = np.clip((1 - w[method].values.astype(float)) * 100, 0, 100)
    return pd.Series(sc, index=w.index)


def _utility_scores(wide, method, higher_is_better=True, r2_filter=False):
    """Utility: clip(1 − bad_deviation/|Real|, 0,1) × 100."""
    if wide.empty or method not in wide.columns or 'Real' not in wide.columns:
        return pd.Series(dtype=float)
    w = wide.dropna(subset=['Real', method])
    R = w['Real'].values.astype(float)
    S = w[method].values.astype(float)
    if r2_filter:
        mask = R >= R2_MIN
        idx  = w.index[mask]
        R, S = R[mask], S[mask]
    else:
        idx = w.index
    bad = np.maximum(0.0, R - S) if higher_is_better else np.maximum(0.0, S - R)
    sc  = np.clip((1 - bad / (np.abs(R) + EPS)) * 100, 0, 100)
    return pd.Series(sc, index=idx)


# ── Per-method mean ± SE across datasets (used by privacy metric_labels) ───────
def _mean_se(arr):
    """(mean, SE) of a 1-D array, ignoring NaN."""
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.nan, np.nan
    return float(np.mean(a)), float(np.std(a, ddof=1) / np.sqrt(len(a)))


def method_scores_privacy(df_b):
    """
    Panel c — Privacy: Anonymeter metrics (singling-out, linkability, inference).
    (Written to code/figure1/panel_b_privacy.csv — the CSV
    letter is the original submission's, the panel letter is the published one.)

    Two-anchor normalised scoring (Real = worst = 0 %, ColShuffle = best = 100 %):
      score = clip( (S − Real) / (ColShuffle − Real), 0, 1) × 100
    higher score = closer to ColShuffle = safer.

    make_whisker_figure() only consumes the returned `has_anonymeter` flag from
    this (it builds its own per-dataset scorer inline); the mean/SE bookkeeping
    is retained verbatim from the source for fidelity.

    Returns (means, ses, metric_labels, has_anonymeter).
    """
    means, ses = {m: [] for m in METHODS_ORDER}, {m: [] for m in METHODS_ORDER}
    metric_labels = []

    has_anonymeter = False
    if df_b is not None:
        has_anonymeter = True
        anon_cols = [
            ('Singling-out\nrisk',   'singling_out'),
            ('Linkability\nrisk',    'linkability'),
            ('Inference\nrisk',      'inference'),
        ]
        anchor_methods = ['Real', 'ColShuffle']
        for lbl, col in anon_cols:
            metric_labels.append(lbl)
            wide = pivot(df_b, col)
            anchors_ok = all(c in wide.columns for c in anchor_methods)
            for method in METHODS_ORDER:
                if not anchors_ok or method not in wide.columns:
                    means[method].append(np.nan); ses[method].append(np.nan)
                    continue
                need = anchor_methods + [method]
                w = wide.dropna(subset=need)
                if w.empty:
                    means[method].append(np.nan); ses[method].append(np.nan)
                    continue
                W_anc = w['Real'].values.astype(float)        # worst (= 0)
                R_anc = w['ColShuffle'].values.astype(float)  # best  (= 100)
                S     = w[method].values.astype(float)
                denom = R_anc - W_anc
                sc    = np.where(np.abs(denom) > EPS,
                                 (S - W_anc) / denom, np.nan)
                sc    = np.clip(sc * 100, 0, 100)
                mv, sv = _mean_se(sc)
                means[method].append(mv); ses[method].append(sv)

    return means, ses, metric_labels, has_anonymeter


# ── Composite (per-dataset mean across a panel's metrics) ──────────────────────
def _composite_scores(method, score_fns):
    """
    Mean 0–100 % score across all metrics in a panel for each dataset.
    Returns a numpy array indexed by dataset (NaN where all metrics are NaN).
    """
    all_sc = []
    for fn in score_fns:
        sc = fn(method)
        all_sc.append(sc.rename('sc'))
    if not all_sc:
        return np.array([])
    df_cat = pd.concat(all_sc, axis=1)
    return df_cat.mean(axis=1, skipna=True).values


# ── Mean ± SD whisker panel (forest-plot style) ────────────────────────────────
def draw_whisker_panel(ax, arrays, labels, colors=None,
                       ylabel='Score (%)', title='', letter=None):
    """
    Mean ± SD whisker panel.
    Dot = mean, horizontal whisker = ±1 SD.  Faded dots = individual datasets.
    Clean forest-plot style.
    """
    n  = len(arrays)
    xs = np.arange(n, dtype=float)
    if colors is None:
        colors = [METHOD_COLORS.get('SPHERE', '#2166AC')] * n

    ax.axhspan(80, 100, color=C_SHADE, alpha=0.20, zorder=0)
    ax.axhline(100, color=C_REF, linestyle='--', linewidth=0.7, alpha=0.6, zorder=1)

    rng = np.random.default_rng(42)

    for xi, (arr, col) in enumerate(zip(arrays, colors)):
        c = arr[~np.isnan(arr)] if len(arr) else np.array([])
        if not len(c):
            continue
        m  = float(np.mean(c))
        sd = float(np.std(c, ddof=1)) if len(c) > 1 else 0.0

        # Faded strip (individual datasets, behind whisker)
        jitter = rng.uniform(-0.12, 0.12, len(c))
        ax.scatter(xi + jitter, c, s=4, color=col, alpha=0.18,
                   zorder=3, linewidths=0)

        # SD whisker (vertical line)
        ax.plot([xi, xi], [m - sd, m + sd], color=col,
                linewidth=2.0, alpha=0.7, zorder=5, solid_capstyle='round')

        # Mean dot
        ax.plot(xi, m, 'o', color=col, ms=6, zorder=6,
                markeredgewidth=1.0, markeredgecolor='white')

        # Mean annotation
        ax.text(xi, max(m + sd, m + 3) + 2, f'{m:.0f}',
                ha='center', va='bottom', fontsize=5.5,
                color=col, fontweight='bold')

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_ylim(-5, 120)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_title(title, loc='left', fontweight='bold', pad=4, fontsize=8)
    if letter is not None:
        _panel_label(ax, letter)


# ══════════════════════════════════════════════════════════════════════════════
# Panel a: SPHERE workflow overview banner (schematic, full width)
# ══════════════════════════════════════════════════════════════════════════════
import re as _re
import xml.etree.ElementTree as _ET
from matplotlib.path import Path as _MplPath
from matplotlib.patches import PathPatch as _PathPatch, Circle as _Circle, \
    Rectangle as _Rectangle, FancyBboxPatch as _FancyBbox
from matplotlib.lines import Line2D as _Line2D
import matplotlib.colors as _mcolors

# ── User's exact inline SVG icons (verbatim from fig-color-v3.html) ────────────
_SVG_SENSITIVE = """<svg viewBox="0 0 116 84">
<g transform="translate(0 10)">
<path d="M15 25 C15 14 37 14 37 25 Z" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<circle cx="26" cy="13" r="5" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<path d="M29 25 C29 14 51 14 51 25 Z" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<circle cx="40" cy="13" r="5" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<path d="M8 39 C8 28 30 28 30 39 Z" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<circle cx="19" cy="27" r="5" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<path d="M22 39 C22 28 44 28 44 39 Z" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<circle cx="33" cy="27" r="5" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<path d="M36 39 C36 28 58 28 58 39 Z" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<circle cx="47" cy="27" r="5" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<path d="M1 53 C1 42 23 42 23 53 Z" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<circle cx="12" cy="41" r="5" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<path d="M15 53 C15 42 37 42 37 53 Z" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<circle cx="26" cy="41" r="5" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<path d="M29 53 C29 42 51 42 51 53 Z" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<circle cx="40" cy="41" r="5" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<path d="M43 53 C43 42 65 42 65 53 Z" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
<circle cx="54" cy="41" r="5" fill="#ffffff" stroke="#1a1a1f" stroke-width="1.7"/>
</g>
<g transform="rotate(-38 96 20)">
<path d="M100 4 Q104 9 96 13 Q88 20 96 27 Q104 32 100 36" stroke="#33333a" stroke-width="1.5" fill="none" stroke-linecap="round"/>
<path d="M92 4 Q88 9 96 13 Q104 20 96 27 Q88 32 92 36" stroke="#33333a" stroke-width="1.5" fill="none" stroke-linecap="round"/>
<line x1="93" y1="8" x2="99" y2="8" stroke="#33333a" stroke-width="1.1" stroke-linecap="round"/>
<line x1="90" y1="20" x2="102" y2="20" stroke="#33333a" stroke-width="1.1" stroke-linecap="round"/>
<line x1="93" y1="32" x2="99" y2="32" stroke="#33333a" stroke-width="1.1" stroke-linecap="round"/>
</g>
<path d="M97 45 C94 41 89 42 89 46.5 C89 52 97 57 97 57 C97 57 105 52 105 46.5 C105 42 100 41 97 45 Z" stroke="#33333a" stroke-width="1.4" fill="none"/>
<path d="M90 49 L94 49 L96 45.5 L98 53 L100 48.5 L104 48.5" stroke="#33333a" stroke-width="1.2" fill="none" stroke-linejoin="round" stroke-linecap="round"/>
<circle cx="97" cy="70" r="8" fill="none" stroke="#33333a" stroke-width="1.4"/>
<text x="97" y="74.5" font-size="11" font-weight="700" fill="#33333a" text-anchor="middle">$</text>
</svg>"""

_SVG_SPHERE = """<svg viewBox="0 0 104 84" fill="none">
<line x1="33" y1="74" x2="80" y2="30" stroke="#1a1a1f" stroke-width="1" stroke-dasharray="4,3"/>
<line x1="38" y1="26" x2="74" y2="68" stroke="#1a1a1f" stroke-width="1" stroke-dasharray="4,3"/>
<circle cx="43" cy="53" r="1.9" fill="#1a1a1f"/>
<circle cx="50" cy="45" r="1.9" fill="#1a1a1f"/>
<circle cx="55" cy="56" r="1.9" fill="#1a1a1f"/>
<circle cx="60" cy="41" r="1.9" fill="#1a1a1f"/>
<circle cx="62" cy="50" r="1.9" fill="#1a1a1f"/>
<circle cx="49" cy="36" r="1.9" fill="#1a1a1f"/>
<circle cx="68" cy="45" r="1.9" fill="#1a1a1f"/>
<circle cx="57" cy="32" r="1.9" fill="#1a1a1f"/>
</svg>"""

_SVG_TWIN = """<svg viewBox="0 0 104 84">
<line x1="15" y1="68" x2="90" y2="68" stroke="#1a1a1f" stroke-width="1"/>
<line x1="34" y1="68" x2="34" y2="71" stroke="#1a1a1f" stroke-width="0.8"/>
<line x1="53" y1="68" x2="53" y2="71" stroke="#1a1a1f" stroke-width="0.8"/>
<line x1="72" y1="68" x2="72" y2="71" stroke="#1a1a1f" stroke-width="0.8"/>
<path d="M19 68 C35 68 37 20 53 20 C69 20 71 68 87 68" stroke="#c8560a" stroke-width="1.4" fill="none"/>
<path d="M19 68 C35 68 37 24 53 24 C69 24 71 68 87 68" stroke="#0072b2" stroke-width="1.4" stroke-dasharray="4.5,3" fill="none"/>
</svg>"""

_SVG_PENTAGON = """<svg viewBox="0 0 280 145" fill="none">
<polygon points="140,24 183.7,55.8 167,107.2 113,107.2 96.3,55.8" fill="none" stroke="#d0d0d4" stroke-width="0.9"/>
<polygon points="140,47 161.85,62.9 153.5,88.6 126.5,88.6 118.15,62.9" fill="none" stroke="#e0e0e4" stroke-width="0.8"/>
<line x1="140" y1="70" x2="140" y2="24" stroke="#e4e4e8" stroke-width="0.8"/>
<line x1="140" y1="70" x2="183.7" y2="55.8" stroke="#e4e4e8" stroke-width="0.8"/>
<line x1="140" y1="70" x2="167" y2="107.2" stroke="#e4e4e8" stroke-width="0.8"/>
<line x1="140" y1="70" x2="113" y2="107.2" stroke="#e4e4e8" stroke-width="0.8"/>
<line x1="140" y1="70" x2="96.3" y2="55.8" stroke="#e4e4e8" stroke-width="0.8"/>
<polygon points="140,28.6 179.4,57.2 164.3,103.5 115.7,103.5 100.6,57.2" fill="#0072b2" fill-opacity="0.14" stroke="#0072b2" stroke-width="1.6" stroke-linejoin="round"/>
<circle cx="140" cy="28.6" r="2.4" fill="#0072b2"/>
<circle cx="179.4" cy="57.2" r="2.4" fill="#0072b2"/>
<circle cx="164.3" cy="103.5" r="2.4" fill="#0072b2"/>
<circle cx="115.7" cy="103.5" r="2.4" fill="#0072b2"/>
<circle cx="100.6" cy="57.2" r="2.4" fill="#0072b2"/>
<text x="140" y="17" font-size="9.5" fill="#1a1a1f" text-anchor="middle">Privacy</text>
<text x="189" y="49" font-size="9.5" fill="#1a1a1f" text-anchor="start">Statistical</text>
<text x="189" y="60" font-size="9.5" fill="#1a1a1f" text-anchor="start">inference</text>
<text x="180" y="122" font-size="9.5" fill="#1a1a1f" text-anchor="middle">Machine-learning</text>
<text x="180" y="133" font-size="9.5" fill="#1a1a1f" text-anchor="middle">utility</text>
<text x="102" y="122" font-size="9.5" fill="#1a1a1f" text-anchor="middle">Distributional</text>
<text x="102" y="133" font-size="9.5" fill="#1a1a1f" text-anchor="middle">fidelity</text>
<text x="91" y="59" font-size="9.5" fill="#1a1a1f" text-anchor="end">Speed</text>
</svg>"""

_SVG_CAP_VALID = """<svg viewBox="0 0 34 28">
<line x1="5" y1="24" x2="5" y2="4" stroke="#a6a6ac" stroke-width="0.8"/>
<line x1="5" y1="24" x2="29" y2="24" stroke="#a6a6ac" stroke-width="0.8"/>
<line x1="6" y1="23" x2="30" y2="4" stroke="#9a9aa0" stroke-width="0.8" stroke-dasharray="3,2"/>
<circle cx="9" cy="20.5" r="1.6" fill="#33333a"/>
<circle cx="13" cy="17.5" r="1.6" fill="#33333a"/>
<circle cx="17" cy="14" r="1.6" fill="#33333a"/>
<circle cx="22" cy="10.5" r="1.6" fill="#33333a"/>
<circle cx="27" cy="6.5" r="1.6" fill="#33333a"/>
</svg>"""

_SVG_CAP_BIOBANK = """<svg viewBox="0 0 34 28" fill="none">
<ellipse cx="17" cy="7" rx="9" ry="3" stroke="#33333a" stroke-width="1.3"/>
<path d="M8 7 V21" stroke="#33333a" stroke-width="1.3"/>
<path d="M26 7 V21" stroke="#33333a" stroke-width="1.3"/>
<path d="M8 21 A9 3 0 0 0 26 21" stroke="#33333a" stroke-width="1.3" fill="none"/>
<path d="M8 14 A9 3 0 0 0 26 14" stroke="#a6a6ac" stroke-width="1" fill="none"/>
</svg>"""

_SVG_CAP_SHARE = """<svg viewBox="0 0 34 28">
<line x1="18" y1="15" x2="7" y2="7" stroke="#a6a6ac" stroke-width="0.9"/>
<line x1="18" y1="15" x2="29" y2="8" stroke="#a6a6ac" stroke-width="0.9"/>
<line x1="18" y1="15" x2="8" y2="23" stroke="#a6a6ac" stroke-width="0.9"/>
<line x1="18" y1="15" x2="28" y2="23" stroke="#a6a6ac" stroke-width="0.9"/>
<circle cx="18" cy="15" r="2.6" fill="#33333a"/>
<circle cx="7" cy="7" r="2.1" fill="none" stroke="#33333a" stroke-width="1.2"/>
<circle cx="29" cy="8" r="2.1" fill="none" stroke="#33333a" stroke-width="1.2"/>
<circle cx="8" cy="23" r="2.1" fill="none" stroke="#33333a" stroke-width="1.2"/>
<circle cx="28" cy="23" r="2.1" fill="none" stroke="#33333a" stroke-width="1.2"/>
</svg>"""

_SVG_CAP_AI = """<svg viewBox="0 0 34 28">
<rect x="9" y="9" width="16" height="13" rx="3" fill="none" stroke="#33333a" stroke-width="1.4"/>
<line x1="17" y1="9" x2="17" y2="5" stroke="#33333a" stroke-width="1.4" stroke-linecap="round"/>
<circle cx="17" cy="3.6" r="1.3" fill="#33333a"/>
<circle cx="14" cy="14.5" r="1.4" fill="#33333a"/>
<circle cx="20" cy="14.5" r="1.4" fill="#33333a"/>
<line x1="14" y1="18.5" x2="20" y2="18.5" stroke="#33333a" stroke-width="1.2" stroke-linecap="round"/>
<line x1="9" y1="14" x2="6.5" y2="14" stroke="#33333a" stroke-width="1.4" stroke-linecap="round"/>
<line x1="25" y1="14" x2="27.5" y2="14" stroke="#33333a" stroke-width="1.4" stroke-linecap="round"/>
</svg>"""

_SVG_CAP_REPRO = """<svg viewBox="0 0 34 28">
<line x1="5" y1="24" x2="31" y2="24" stroke="#a6a6ac" stroke-width="0.8"/>
<line x1="13" y1="8" x2="13" y2="21" stroke="#33333a" stroke-width="1.1"/>
<line x1="10" y1="8" x2="16" y2="8" stroke="#33333a" stroke-width="1.1"/>
<line x1="10" y1="21" x2="16" y2="21" stroke="#33333a" stroke-width="1.1"/>
<circle cx="13" cy="14.5" r="1.9" fill="#33333a"/>
<line x1="23" y1="8" x2="23" y2="21" stroke="#33333a" stroke-width="1.1"/>
<line x1="20" y1="8" x2="26" y2="8" stroke="#33333a" stroke-width="1.1"/>
<line x1="20" y1="21" x2="26" y2="21" stroke="#33333a" stroke-width="1.1"/>
<circle cx="23" cy="14.5" r="1.9" fill="#33333a"/>
</svg>"""


# ── Minimal SVG-subset renderer: ports the icons above to matplotlib vectors ───
_SVG_PATH_TOK = _re.compile(r'([MmLlHhVvCcSsQqTtAaZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)')


def _svg_fill(fill, default='black'):
    if fill is None:
        return default
    return 'none' if fill == 'none' else fill


def _svg_affine_translate(tx, ty):
    m = np.eye(3); m[0, 2] = tx; m[1, 2] = ty
    return m


def _svg_affine_rotate(deg, cx=0.0, cy=0.0):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return _svg_affine_translate(cx, cy) @ r @ _svg_affine_translate(-cx, -cy)


def _svg_parse_transform(txt):
    m = np.eye(3)
    if not txt:
        return m
    for name, arg in _re.findall(r'(translate|rotate|scale)\s*\(([^)]*)\)', txt):
        v = [float(x) for x in _re.split(r'[\s,]+', arg.strip()) if x != '']
        if name == 'translate':
            m = m @ _svg_affine_translate(v[0], v[1] if len(v) > 1 else 0.0)
        elif name == 'rotate':
            m = m @ (_svg_affine_rotate(v[0], v[1], v[2]) if len(v) >= 3
                     else _svg_affine_rotate(v[0]))
    return m


def _svg_arc_points(x0, y0, rx, ry, phi, large, sweep, x, y, n=22):
    """SVG elliptical arc -> sampled points (endpoint-to-centre parameterisation)."""
    phi = np.radians(phi); cosp, sinp = np.cos(phi), np.sin(phi)
    dx, dy = (x0 - x) / 2.0, (y0 - y) / 2.0
    x1p = cosp * dx + sinp * dy
    y1p = -sinp * dx + cosp * dy
    rx, ry = abs(rx), abs(ry)
    lam = x1p**2 / rx**2 + y1p**2 / ry**2
    if lam > 1:
        f = np.sqrt(lam); rx *= f; ry *= f
    denom = rx**2 * y1p**2 + ry**2 * x1p**2
    num = max(rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2, 0.0)
    co = np.sqrt(num / denom) if denom else 0.0
    if large == sweep:
        co = -co
    cxp = co * rx * y1p / ry
    cyp = -co * ry * x1p / rx
    cx = cosp * cxp - sinp * cyp + (x0 + x) / 2.0
    cy = sinp * cxp + cosp * cyp + (y0 + y) / 2.0

    def ang(ux, uy, vx, vy):
        d = np.hypot(ux, uy) * np.hypot(vx, vy)
        a = np.arccos(np.clip((ux * vx + uy * vy) / d, -1, 1))
        return -a if (ux * vy - uy * vx) < 0 else a

    th1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dth = ang((x1p - cxp) / rx, (y1p - cyp) / ry,
              (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dth > 0:
        dth -= 2 * np.pi
    elif sweep and dth < 0:
        dth += 2 * np.pi
    out = []
    for t in np.linspace(0, 1, n)[1:]:
        a = th1 + dth * t
        px = cosp * rx * np.cos(a) - sinp * ry * np.sin(a) + cx
        py = sinp * rx * np.cos(a) + cosp * ry * np.sin(a) + cy
        out.append((px, py))
    return out


def _svg_parse_path(d):
    toks = [m.group(1) if m.group(1) else float(m.group(2))
            for m in _SVG_PATH_TOK.finditer(d)]
    verts, codes = [], []
    i, n = 0, len(toks)
    cur = [0.0, 0.0]; start = [0.0, 0.0]; cmd = None
    while i < n:
        if isinstance(toks[i], str):
            cmd = toks[i]; i += 1
            if i >= n:
                break
        rel = cmd.islower(); C = cmd.upper()

        def take():
            nonlocal i
            v = toks[i]; i += 1
            return v

        if C == 'M':
            x, y = take(), take()
            if rel:
                x += cur[0]; y += cur[1]
            cur = [x, y]; start = [x, y]
            verts.append((x, y)); codes.append(_MplPath.MOVETO)
            cmd = 'l' if rel else 'L'
        elif C == 'L':
            x, y = take(), take()
            if rel:
                x += cur[0]; y += cur[1]
            cur = [x, y]; verts.append((x, y)); codes.append(_MplPath.LINETO)
        elif C == 'H':
            x = take() + (cur[0] if rel else 0.0)
            cur = [x, cur[1]]; verts.append(tuple(cur)); codes.append(_MplPath.LINETO)
        elif C == 'V':
            y = take() + (cur[1] if rel else 0.0)
            cur = [cur[0], y]; verts.append(tuple(cur)); codes.append(_MplPath.LINETO)
        elif C == 'C':
            p = [take() for _ in range(6)]
            if rel:
                p = [p[k] + (cur[0] if k % 2 == 0 else cur[1]) for k in range(6)]
            verts += [(p[0], p[1]), (p[2], p[3]), (p[4], p[5])]
            codes += [_MplPath.CURVE4] * 3
            cur = [p[4], p[5]]
        elif C == 'Q':
            p = [take() for _ in range(4)]
            if rel:
                p = [p[k] + (cur[0] if k % 2 == 0 else cur[1]) for k in range(4)]
            verts += [(p[0], p[1]), (p[2], p[3])]
            codes += [_MplPath.CURVE3] * 2
            cur = [p[2], p[3]]
        elif C == 'A':
            rx, ry, phi = take(), take(), take()
            large, sweep = int(take()), int(take())
            x, y = take(), take()
            if rel:
                x += cur[0]; y += cur[1]
            for px, py in _svg_arc_points(cur[0], cur[1], rx, ry, phi,
                                          large, sweep, x, y):
                verts.append((px, py)); codes.append(_MplPath.LINETO)
            cur = [x, y]
        elif C == 'Z':
            verts.append((start[0], start[1])); codes.append(_MplPath.CLOSEPOLY)
            cur = list(start)
        else:
            i += 1
    return verts, codes


def _svg_draw(ax, svg_str, ox, oy, sc, ppd, vbh, z0=5):
    """Render one inline-SVG icon into axis data coords (y flipped, origin ox/oy)."""
    def gmap(x, y):
        return ox + x * sc, oy + (vbh - y) * sc

    def walk(node, M, deffill='black'):
        for el in node:
            tag = el.tag.split('}')[-1]
            Mel = M @ _svg_parse_transform(el.get('transform', ''))
            if tag == 'g':
                walk(el, Mel, el.get('fill', deffill)); continue
            fill = el.get('fill', deffill); stroke = el.get('stroke')
            sw = float(el.get('stroke-width', '1') or 1)
            lw = sw * sc * ppd
            cap = {'round': 'round', 'square': 'projecting'}.get(
                el.get('stroke-linecap', 'butt'), 'butt')
            join = el.get('stroke-linejoin', 'miter')
            dseq = None
            da = el.get('stroke-dasharray')
            if da and da != 'none':
                dseq = [max(float(x) * sc * ppd, 0.1)
                        for x in _re.split(r'[\s,]+', da.strip()) if x != '']

            def tf(x, y):
                p = Mel @ np.array([x, y, 1.0])
                return gmap(p[0], p[1])

            if tag == 'path':
                verts, codes = _svg_parse_path(el.get('d', ''))
                pth = _MplPath([tf(x, y) for x, y in verts], codes)
                pp = _PathPatch(pth, facecolor=_svg_fill(fill),
                                edgecolor=(stroke or 'none'), lw=lw,
                                joinstyle=join, capstyle=cap, zorder=z0)
                if dseq:
                    pp.set_linestyle((0, tuple(dseq)))
                ax.add_patch(pp)
            elif tag == 'circle':
                X, Y = tf(float(el.get('cx')), float(el.get('cy')))
                c = _Circle((X, Y), float(el.get('r')) * sc,
                            facecolor=_svg_fill(fill),
                            edgecolor=(stroke or 'none'), lw=lw, zorder=z0 + 1)
                if dseq:
                    c.set_linestyle((0, tuple(dseq)))
                ax.add_patch(c)
            elif tag == 'line':
                x1, y1 = tf(float(el.get('x1')), float(el.get('y1')))
                x2, y2 = tf(float(el.get('x2')), float(el.get('y2')))
                ln = _Line2D([x1, x2], [y1, y2], color=(stroke or 'black'),
                             lw=lw, solid_capstyle=cap, dash_capstyle=cap,
                             zorder=z0)
                if dseq:
                    ln.set_dashes(dseq)
                ax.add_line(ln)
            elif tag == 'rect':
                x, y = float(el.get('x')), float(el.get('y'))
                w, h = float(el.get('width')), float(el.get('height'))
                rx = float(el.get('rx', '0') or 0)
                X, Y = tf(x, y + h)                       # lower-left after flip
                if rx > 0:
                    r = _FancyBbox((X + rx * sc, Y + rx * sc),
                                   (w - 2 * rx) * sc, (h - 2 * rx) * sc,
                                   boxstyle=f'round,pad={rx * sc},'
                                            f'rounding_size={rx * sc}',
                                   facecolor=_svg_fill(fill),
                                   edgecolor=(stroke or 'none'), lw=lw, zorder=z0)
                else:
                    r = _Rectangle((X, Y), w * sc, h * sc,
                                   facecolor=_svg_fill(fill),
                                   edgecolor=(stroke or 'none'), lw=lw, zorder=z0)
                ax.add_patch(r)
            elif tag == 'polygon':
                pts = [float(v) for v in _re.split(r'[\s,]+',
                       el.get('points', '').strip()) if v != '']
                xy = [tf(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
                verts = xy + [xy[0]]
                codes = ([_MplPath.MOVETO] + [_MplPath.LINETO] * (len(xy) - 1)
                         + [_MplPath.CLOSEPOLY])
                fo = float(el.get('fill-opacity', '1') or 1)
                if fill and fill != 'none':
                    fc = _mcolors.to_rgba(fill, fo)             # own alpha, edge stays opaque
                else:
                    fc = 'none'
                pp = _PathPatch(_MplPath(verts, codes), facecolor=fc,
                                edgecolor=(stroke or 'none'), lw=lw,
                                joinstyle=join, zorder=z0)
                if dseq:
                    pp.set_linestyle((0, tuple(dseq)))
                ax.add_patch(pp)
            elif tag == 'ellipse':
                cx, cy = float(el.get('cx')), float(el.get('cy'))
                rx, ry = float(el.get('rx')), float(el.get('ry'))
                th = np.linspace(0, 2 * np.pi, 48)
                xy = [tf(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in th]
                verts = xy + [xy[0]]
                codes = ([_MplPath.MOVETO] + [_MplPath.LINETO] * (len(xy) - 1)
                         + [_MplPath.CLOSEPOLY])
                pp = _PathPatch(_MplPath(verts, codes), facecolor=_svg_fill(fill),
                                edgecolor=(stroke or 'none'), lw=lw, zorder=z0)
                if dseq:
                    pp.set_linestyle((0, tuple(dseq)))
                ax.add_patch(pp)
            elif tag == 'text':
                X, Y = tf(float(el.get('x')), float(el.get('y')))
                ha = {'middle': 'center', 'end': 'right'}.get(
                    el.get('text-anchor', 'start'), 'left')
                fs = float(el.get('font-size', '10')) * sc * ppd
                weight = 'bold' if el.get('font-weight') in ('700', 'bold') else 'normal'
                style = 'italic' if el.get('font-style') == 'italic' else 'normal'
                rot = -np.degrees(np.arctan2(Mel[1, 0], Mel[0, 0]))
                ax.text(X, Y, el.text or '', ha=ha, va='baseline',
                        fontsize=fs, fontweight=weight, fontstyle=style,
                        color=_svg_fill(fill), rotation=rot,
                        rotation_mode='anchor', zorder=z0 + 2)

    _root = _ET.fromstring(svg_str)
    walk(_root, np.eye(3), _root.get('fill', 'black'))


def draw_pipeline_banner(ax):
    """
    Full-width schematic (Figure 1a): the SPHERE workflow
        Sensitive data -> SPHERE -> Synthetic twin -> Statistical fidelity
    plus a capability strip. Icons are the user's exact inline SVG from
    fig-color-v3.html, ported verbatim (colours, geometry, stroke widths) to
    matplotlib vector primitives -- no redesign, no rasterisation.
    """
    AR = 4.35                                     # banner data aspect (w / h)
    # the banner axes shares the panels' left/right margins (0.09/0.98), so its
    # centre sits right of the figure centre; shift the data origin left so the
    # content is optically centred on the figure (panels need the 0.09 y-label gap).
    # Content is laid out symmetric about AR/2, so XOFF = 0.0393*AR centres it.
    XOFF = 0.171
    ax.set_xlim(XOFF, AR + XOFF); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('SPHERE workflow', loc='left', fontweight='bold',
                 fontsize=8, pad=4)                 # match panels b-g titled style

    fig = ax.figure
    pos = ax.get_position()
    ppd = 72.0 * fig.get_size_inches()[0] * pos.width / AR   # points per data unit

    INK = '#000000'                               # all banner labels: black

    # optional: unify pipeline colours with the manuscript scheme
    #   real -> #D73027 (was orange #c8560a), synthetic -> #2166AC (was #0072b2)
    if os.environ.get('FIG1_UNIFY_COLOR', '1') == '1':
        _recolor = lambda s: s.replace('#0072b2', '#2166AC').replace('#c8560a', '#D73027')
    else:
        _recolor = lambda s: s

    # ── pipeline row: 3 compact nodes + 1 larger self-labelled pentagon result ──
    # Lay slots left->right with an EQUAL edge-to-edge gap so every arrow is the
    # same length. The pentagon viewBox has whitespace, so use its *visible* left
    # offset (Speed label ~vb x=72, centre 140 -> 68 units) for the gap, not wp/2.
    icon_y, h_icon = 0.84, 0.30
    title_y, sub_y = 0.575, 0.470
    h_pent = 0.46; sc_p = h_pent / 145; wp = 280 * sc_p
    y_pent = 0.76
    pent_lvis = 68 * sc_p                          # pentagon visible half-width (left)

    # (svg, vbw, vbh, height, left_visible_off, right_visible_off)
    slots = [(_SVG_SENSITIVE, 116, 84, h_icon, None, None),
             (_SVG_SPHERE,    104, 84, h_icon, None, None),
             (_SVG_TWIN,      104, 84, h_icon, None, None),
             (_SVG_PENTAGON,  280, 145, h_pent, pent_lvis, None)]
    GAP, INSET, x0 = 0.62, 0.05, 0.42
    centers, cur = [], x0
    for svg, vbw, vbh, h, lvis, rvis in slots:
        sc = h / vbh; w = vbw * sc
        cx = cur + w / 2
        loff = lvis if lvis is not None else w / 2
        roff = rvis if rvis is not None else w / 2
        yc = y_pent if svg is _SVG_PENTAGON else icon_y
        _svg_draw(ax, _recolor(svg), cx - w / 2, yc - h / 2, sc, ppd, vbh)
        centers.append((cx, loff, roff))
        cur = cx + roff + GAP                       # next slot's visible-left edge
    for (ca, _, ra), (cb, lb, _) in zip(centers[:-1], centers[1:]):
        ax.annotate('', xy=(cb - lb - INSET, icon_y), xytext=(ca + ra + INSET, icon_y),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.0,
                                    shrinkA=0, shrinkB=0))
    xs_node = [c[0] for c in centers[:3]]
    x_pent = centers[3][0]

    # node captions (node 3 wraps; pentagon caption sits below the radar)
    ax.text(xs_node[0], title_y, 'Sensitive data', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color=INK)
    ax.text(xs_node[0], sub_y, 'Real, access-restricted', ha='center', va='center',
            fontsize=5.8, color=INK)
    ax.text(xs_node[1], title_y, 'SPHERE', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color=INK)
    ax.text(xs_node[1], sub_y, 'Runs locally,\nmodel-free, no tuning',
            ha='center', va='center', fontsize=5.8, color=INK, linespacing=1.05)
    ax.text(xs_node[2], title_y - 0.005, 'SPHERE\ntransformed data', ha='center',
            va='center', fontsize=7.5, fontweight='bold', color=INK, linespacing=1.05)
    ax.text(x_pent, 0.480, 'Meets all five\nrequirements', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color=INK, linespacing=1.05)

    # ── fan-out connector: pipeline -> 5 capabilities ──
    show_icons = os.environ.get('FIG1_CAP_ICONS', '1') != '0'
    caps = [(_SVG_CAP_VALID,   '33 datasets\nvalidated',       'Five scientific domains'),
            (_SVG_CAP_AI,       'Enabled\nAI-agent analysis',   'Without access to\noriginal data'),
            (_SVG_CAP_BIOBANK,  'Demonstrated at\nbiobank scale', 'UK Biobank\ngenomics & proteomics'),
            (_SVG_CAP_SHARE,    'Open multimodal\nAD cohort',   'Stanford ADRC'),
            (_SVG_CAP_REPRO,    'Published findings\nreproduced', 'Same biological &\nclinical conclusions')]
    xs_cap = np.linspace(0.45, AR - 0.45, len(caps))
    rail_y = 0.400
    # vertex drops from the 'SPHERE transformed data' node (node 3), not the centre
    ax.plot([xs_node[2], xs_node[2]], [0.490, rail_y],
            color=INK, lw=0.8, clip_on=False)
    ax.plot([xs_cap[0], xs_cap[-1]], [rail_y, rail_y], color=INK, lw=0.8, clip_on=False)
    for xc in xs_cap:
        ax.plot([xc, xc], [rail_y, rail_y - 0.03], color=INK, lw=0.8, clip_on=False)

    # ── capability strip (icons toggled by FIG1_CAP_ICONS=0/1 -> two versions) ──
    ci_h = 0.085
    iy = 0.310
    wy, cy = (0.175, 0.050) if show_icons else (0.255, 0.090)
    for xc, (svg, word, capt) in zip(xs_cap, caps):
        if show_icons:
            sc = ci_h / 28; tw = 34 * sc
            _svg_draw(ax, svg, xc - tw / 2, iy - ci_h / 2, sc, ppd, 28)
        ax.text(xc, wy, word, ha='center', va='center', fontsize=6.2,
                fontweight='bold', color=INK, linespacing=1.0)
        ax.text(xc, cy, capt, ha='center', va='center', fontsize=5.4,
                color=INK, linespacing=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Main figure: 7-panel benchmark (schematic banner a + mean ± SD whisker panels)
# ══════════════════════════════════════════════════════════════════════════════
def make_whisker_figure(df_b, df_c, df_d, df_e, df_f):
    """
    Figure 1 main: dot (mean) + SD whisker + faded strip per method.
    Saves figures/sphere_benchmark_figure.{pdf,png}
    """
    b_fns = [
        lambda m: _fidelity_scores(pivot(df_c, 'pct_delta_cor'),  m),
        lambda m: _fidelity_scores(pivot(df_c, 'pct_delta_mean'), m),
        lambda m: _fidelity_scores(pivot(df_c, 'pct_delta_var'),  m),
        lambda m: _ks_scores(pivot(df_c, 'ks_statistic'),         m),
    ] if df_c is not None else []
    c_fns = [
        lambda m: _utility_scores(pivot(df_d, 'type1_error'), m, higher_is_better=False),
        lambda m: _utility_scores(pivot(df_d, 'power'),       m, higher_is_better=True),
        lambda m: _utility_scores(pivot(df_d, 'ci_coverage'), m, higher_is_better=True),
    ] if df_d is not None else []
    d_fns = [
        (lambda c: lambda m: _utility_scores(pivot(df_e, c), m, True, True))(col)
        for col in ('EN', 'SVM', 'RF', 'GB', 'MLP')
    ] if df_e is not None else []
    _, _, _, has_anon = method_scores_privacy(df_b)
    p_fns = []
    if df_b is not None:
        anchor_ok = all(c in pivot(df_b, 'singling_out').columns
                        for c in ['Real', 'ColShuffle'])
        if anchor_ok:
            def _anon_score_v4(col):
                def fn(m):
                    wide = pivot(df_b, col)
                    need = ['Real', 'ColShuffle', m]
                    if not all(c in wide.columns for c in need):
                        return pd.Series(dtype=float)
                    w = wide.dropna(subset=need)
                    W = w['Real'].values.astype(float)
                    R = w['ColShuffle'].values.astype(float)
                    S = w[m].values.astype(float)
                    denom = R - W
                    sc = np.where(np.abs(denom) > EPS, (S - W) / denom, np.nan)
                    return pd.Series(np.clip(sc * 100, 0, 100), index=w.index)
                return fn
            p_fns = [_anon_score_v4(c) for c in
                     ('singling_out', 'linkability', 'inference')]

    FIG_W = 183 / 25.4
    FIG_H = 247 / 25.4
    fig   = plt.figure(figsize=(FIG_W, FIG_H))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.70, 3.0],
        top=0.965, bottom=0.145, left=0.09, right=0.98, hspace=0.16)
    ax_a = fig.add_subplot(outer[0])                       # full-width banner
    gs = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=outer[1],
        hspace=0.5, wspace=0.28)
    ax_b = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[2, 0])
    ax_g = fig.add_subplot(gs[2, 1])

    draw_pipeline_banner(ax_a)
    # align panel 'a' letter to the left-column letters (b/d/f) in figure x
    fig.canvas.draw()
    pb = ax_b.get_position(); pa = ax_a.get_position()
    bx = pb.x0 + (-0.14) * pb.width                    # b's letter figure-x
    _panel_label(ax_a, 'a', dx=(bx - pa.x0) / pa.width, dy=1.0)

    draw_dataset_overview(ax_b, df_c, title='Benchmark datasets')
    _panel_label(ax_b, 'b')

    for ax, fns, letter, title in [
        (ax_c, p_fns, 'c', 'Privacy'),
        (ax_d, b_fns, 'd', 'Distributional fidelity'),
        (ax_e, c_fns, 'e', 'Utility for statistical inference'),
        (ax_f, d_fns, 'f', 'Utility for machine learning'),
    ]:
        if not fns:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7, color='#888888')
            ax.set_title(title, loc='left', fontweight='bold', pad=4, fontsize=8)
            _panel_label(ax, letter)
            continue
        arrays = [_composite_scores(m, fns) for m in METHODS_ORDER]
        colors = [METHOD_COLORS[m] for m in METHODS_ORDER]
        labels = [METHOD_LABELS_SHORT.get(m, m) for m in METHODS_ORDER]
        draw_whisker_panel(ax, arrays, labels, colors=colors,
                           ylabel='Score (%)', title=title, letter=letter)

    # Panel g: timing whisker (same style as c–f, but log-scale, relative to ColShuffle)
    if df_f is not None and 'runtime_ms' in df_f.columns:
        cs = df_f[df_f['method'] == 'ColShuffle'][['dataset', 'runtime_ms']].dropna()
        cs_map = dict(zip(cs['dataset'], cs['runtime_ms']))

        plot_methods = [m for m in METHODS_ORDER if m != 'Real']
        xs_f = np.arange(len(plot_methods))
        rng_f = np.random.default_rng(42)

        for xi, m in enumerate(plot_methods):
            col = METHOD_COLORS[m]
            sub = df_f[df_f['method'] == m][['dataset', 'runtime_ms']].dropna()
            rels = []
            for _, row in sub.iterrows():
                cs_val = cs_map.get(row['dataset'])
                if cs_val and cs_val > 0:
                    rels.append(row['runtime_ms'] / cs_val)
            vals = np.array(rels)
            if len(vals) < 1:
                ax_g.text(xi, 1, 'n/a', ha='center', va='bottom',
                          fontsize=5, color='#aaaaaa', style='italic')
                continue
            m_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

            # Faded strip
            jitter = rng_f.uniform(-0.12, 0.12, len(vals))
            ax_g.scatter(xi + jitter, vals, s=4, color=col, alpha=0.18,
                         zorder=3, linewidths=0)
            # SD whisker
            lo = max(m_val - sd_val, m_val * 0.1)  # avoid negative on log
            ax_g.plot([xi, xi], [lo, m_val + sd_val], color=col,
                      linewidth=2.0, alpha=0.7, zorder=5, solid_capstyle='round')
            # Mean dot
            ax_g.plot(xi, m_val, 'o', color=col, ms=6, zorder=6,
                      markeredgewidth=1.0, markeredgecolor='white')
            # Annotation
            lbl = f'{m_val:.0f}×' if m_val >= 10 else f'{m_val:.1f}×'
            ax_g.text(xi, (m_val + sd_val) * 1.3, lbl,
                      ha='center', va='bottom', fontsize=5.5,
                      color=col, fontweight='bold')

        ax_g.axhline(1.0, color=C_REF, linestyle='--', linewidth=0.7, alpha=0.5, zorder=1)
        ax_g.set_yscale('log')
        ax_g.set_ylabel('Runtime relative to ColShuffle', fontsize=7.5)
        ax_g.set_yticks([1, 10, 100, 1e3, 1e4, 1e5])
        ax_g.set_ylim(0.28, 3e5)
        ax_g.set_xticks(xs_f)
        ax_g.set_xticklabels([METHOD_LABELS_SHORT.get(m, m) for m in plot_methods],
                             rotation=35, ha='right')
        ax_g.set_title('Computing time', loc='left',
                       fontweight='bold', pad=4, fontsize=8)
        _panel_label(ax_g, 'g')
        ax_g.set_xlim(-0.6, len(plot_methods) - 0.4)
    else:
        ax_g.text(0.5, 0.5, 'No timing data', ha='center', va='center',
                  transform=ax_g.transAxes, fontsize=7, color='#888888')
        ax_g.set_title('Computing time', loc='left',
                       fontweight='bold', pad=4, fontsize=8)
        _panel_label(ax_g, 'g')

    # Desired left-to-right, top-to-bottom reading order = METHODS_ORDER, then the
    # target-zone key. matplotlib fills a multi-column legend COLUMN-major (down the
    # first column, then the next), so a naive handle list reads scrambled. Lay the
    # desired sequence into a row-major grid, then flatten column-major so the on-
    # screen order matches METHODS_ORDER exactly.
    desired = [Line2D([0], [0], color=METHOD_COLORS[m], linewidth=1.5,
                      marker='o', markersize=3,
                      label=METHOD_LABELS_SHORT.get(m, m))
               for m in METHODS_ORDER]
    # The ≥80 % target-zone band stays in the panels but is described in the caption,
    # not the legend (a one-line key cannot define "satisfy"; it would read arbitrary).
    ncol = 6                                     # 12 entries -> 2 clean rows (6 + 6)
    nrow = -(-len(desired) // ncol)
    grid = [desired[r * ncol:(r + 1) * ncol] for r in range(nrow)]
    handles = [grid[r][c] for c in range(ncol) for r in range(nrow)
               if c < len(grid[r])]
    fig.legend(handles=handles, loc='lower center', ncol=ncol,
               bbox_to_anchor=(0.53, 0.006),
               columnspacing=0.9, handletextpad=0.4, fontsize=6.5, borderpad=0.4)

    stem = os.environ.get('FIG1_STEM', 'sphere_benchmark_figure')
    for ext in ('pdf', 'png'):
        dpi = 600 if ext == 'png' else None
        path = os.path.join(str(FIGS), f'{stem}.{ext}')
        fig.savefig(path, format=ext, **(dict(dpi=dpi) if dpi else {}))
        print(f'Saved: {path}')
    plt.close(fig)




def main():
    df_b, df_c, df_d, df_e, df_f = load_panels()
    missing = [n for n, df in
               [('panel_b_privacy.csv', df_b),
                ('panel_c_fidelity.csv', df_c),
                ('panel_d_inference.csv', df_d),
                ('panel_e_utility.csv', df_e),
                ('panel_f_timing.csv', df_f)] if df is None]
    if missing:
        raise SystemExit(
            f"Missing panel CSVs: {missing}\n"
            "Run the panel scripts first (panel_b..f).")
    make_whisker_figure(df_b, df_c, df_d, df_e, df_f)


if __name__ == '__main__':
    main()

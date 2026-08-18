"""
Shared constants, paths and small helpers for the Table 1 pipeline.

Importing this module pins every BLAS backend to a single thread. That has to
happen before numpy is imported anywhere in the process, so `import _common`
must be the first project import in every step script — several steps fan out
over processes, and an unpinned BLAS oversubscribes the machine badly enough to
look like the method itself is slow.

The probes are deliberately NOT here. Each modality trains a different network
with a different objective and a different checkpoint rule (binary AUC for text
and ECG, Top-1 for image; AdamW for the text probe, SGD for the image linear
probe), and the published numbers are tied to those exact definitions. They live
in the step that uses them so that a reader can check one file against one row
of the table.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # <repo>/code/table1
CODE = HERE.parent                              # <repo>/code
REPO = CODE.parent                              # <repo>
DATA = REPO / "data" / "table1"                 # raw inputs + heavy caches (gitignored)
OUT = HERE                                       # aggregate results
RUNLOG = HERE / "_runlog"


# ── protocol constants ───────────────────────────────────────────────────────
# Shared by every modality. Anything modality-specific (epochs, learning rate,
# batch size, hidden width) stays in that modality's step script, because the
# three probes were tuned separately and share no defaults.
THETA = np.pi / 6           # SPHERE rotation angle, one pass
K = 1                       # Table 1 reports k=1; the k=1..5 sweep is ST19/ST20
N_SEEDS = 10                # every cell is a mean over 10 seeds

# SPHERE seed bases, per modality. These are NOT interchangeable: the privacy
# step must rotate with the same seed and the same call signature as the utility
# step, or the two halves of a Table 1 row describe different matrices.
SPHERE_SEED = {"text": 0, "image": 42, "ecg": 0}


def stat_leaf(vals):
    """mean / std / sem / per-seed record, the shape every result JSON stores."""
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "sem": float(a.std() / np.sqrt(len(a))),
        "per_seed": [float(v) for v in a],
    }


def colshuffle(Z, seed):
    """Independent per-column row permutation — the privacy floor.

    Every marginal is preserved and every row identity is destroyed, so this is
    the "maximum protection" anchor against which SPHERE is scored.
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(Z)
    for j in range(Z.shape[1]):
        out[:, j] = Z[rng.permutation(Z.shape[0]), j]
    return out


def get_device(pref=None):
    """Resolve the torch device. Default matches the published run: MPS > CUDA > CPU.

    The published Table 1 was produced on Apple silicon, so MPS is what reproduces
    it -- image real/sphere match the original run per-seed on this path. Forcing
    "cpu" is portable across machines but gives different numbers, because the
    reduction order differs; use it only for a deliberate portability check.

    step4 (text) pins CPU on purpose: it is small enough that CPU is not the
    bottleneck, and it reproduces bit-exactly there.

    Resolution order: explicit `pref` > $TABLE1_DEVICE > "auto".
    """
    import torch
    name = pref or os.environ.get("TABLE1_DEVICE") or "auto"
    if name == "auto":
        if torch.backends.mps.is_available():
            name = "mps"
        elif torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
    return torch.device(name)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {path}", flush=True)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def require(path, hint):
    """Fail loudly and specifically when an upstream step has not been run."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing input: {p}\n  {hint}")
    return p

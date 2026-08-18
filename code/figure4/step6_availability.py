#!/usr/bin/env python3
"""
Step 6 — participant x modality data availability (panel a).

Which of the nine modalities each participant carries, in the real tree and in
the released rotation, plus the consensus diagnosis the panel sorts and colours
the participants by. Real and synthetic availability are recorded separately
because the rotation is supposed to preserve the missingness pattern exactly;
the panel is the visual check that it does.

Output: one row per participant, ordered as the panel draws them (grouped by
diagnosis), with a 0/1 column per modality per arm.

    python3 step6_availability.py \\
        --real-root  data/adrc/real \\
        --sphere-dir results/figure4/sphere_xk/x2 \\
        --out-csv    results/figure4/panel_a_availability.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adrc_common import DIAG_ORDER, DIAG_SHORT, FIG_MOD_FILE, FIG_MOD_ORDER, ID_COL


def build_availability(real_root: Path, sphere_dir: Path) -> pd.DataFrame:
    dem_sphere = pd.read_csv(sphere_dir / FIG_MOD_FILE["demographics"],
                             low_memory=False, dtype={ID_COL: str})
    diag_map = {str(r[ID_COL]): str(r.get("diagnosis_consensus", ""))
                for _, r in dem_sphere.iterrows()}
    all_pids = sorted(dem_sphere[ID_COL].astype(str).tolist())
    pid_diag = {p: DIAG_SHORT.get(diag_map.get(p, ""), "Other") for p in all_pids}
    pids = sorted(all_pids, key=lambda p: DIAG_ORDER.index(pid_diag.get(p, "Other")))

    def build_avail(root):
        mat = np.zeros((len(pids), len(FIG_MOD_ORDER)), dtype=np.float32)
        pid_idx = {p: i for i, p in enumerate(pids)}
        for j, mod in enumerate(FIG_MOD_ORDER):
            path = root / FIG_MOD_FILE[mod]
            if not path.exists():
                continue
            df = pd.read_csv(path, usecols=[ID_COL], low_memory=False, dtype={ID_COL: str})
            for p in df[ID_COL].astype(str):
                if p in pid_idx:
                    mat[pid_idx[p], j] = 1.0
        return mat

    out = pd.DataFrame({ID_COL: pids, "diagnosis": [pid_diag[p] for p in pids]})
    for arm, root in (("real", real_root), ("sphere", sphere_dir)):
        mat = build_avail(root)
        for j, mod in enumerate(FIG_MOD_ORDER):
            out[f"{arm}_{mod}"] = mat[:, j].astype(int)
    return out


def main():
    ap = argparse.ArgumentParser(description="Participant x modality availability for panel a")
    ap.add_argument("--real-root", type=Path, required=True)
    ap.add_argument("--sphere-dir", type=Path, required=True,
                    help="Per-modality CSVs of the released rotation")
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    df = build_availability(args.real_root, args.sphere_dir)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    n_mod = [f"{m}: real {int(df['real_' + m].sum())} / sphere {int(df['sphere_' + m].sum())}"
             for m in FIG_MOD_ORDER]
    print(f"[write] {len(df)} participants -> {args.out_csv}")
    print("  " + "\n  ".join(n_mod))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Figure 1 panel b — empirical privacy (Anonymeter).

Runs singling-out, linkability, and inference attacks against each
synthesiser's output. Lower risk = more private.

Anonymeter requires Python <3.13 and a separate venv, so the attacks
run in a subprocess (_anonymeter_worker.py). Set the interpreter via:

    ANONYMETER_PYTHON=/path/to/venv/bin/python python3 code/figure1/panel_b_privacy.py

Requires anonymeter==1.0.0 (pinned for reproducibility).

Usage:
    python3 code/figure1/panel_b_privacy.py [--methods ...] [--skip-deep] [--workers N]
"""
import os
import shutil
import subprocess

import pandas as pd

import _common as C

WORKER = C.HERE / "_anonymeter_worker.py"
TIMEOUT_S = 7200
INTERPRETER_CANDIDATES = ("python3.11", "python3.12")   # 3.11 first: anonymeter 1.0.0


def find_interpreter():
    """The interpreter that has anonymeter installed. None if we cannot find one."""
    env = os.environ.get("ANONYMETER_PYTHON")
    if env:
        if not shutil.which(env) and not os.path.exists(env):
            raise SystemExit(f"ANONYMETER_PYTHON={env} is not executable")
        return env
    for cand in INTERPRETER_CANDIDATES:
        found = shutil.which(cand)
        if found:
            return found
    return None


def compute(methods, interpreter):
    """Run the attacks in `interpreter`, returning the resulting DataFrame."""
    tmp_csv = C.RES / "panel_b_privacy_tmp.csv"
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv.unlink(missing_ok=True)          # drop any stale temp from a prior crash

    # 'Real' is the ori-vs-ori baseline and is added by the worker itself.
    methods_arg = ",".join(m for m in methods if m != "Real")
    try:
        result = subprocess.run(
            [interpreter, str(WORKER), str(C.REAL_DIR), str(C.SYNTH_FULL),
             str(tmp_csv), methods_arg],
            timeout=TIMEOUT_S)
        if result.returncode != 0:
            raise SystemExit(f"anonymeter worker exited with code {result.returncode}")
        if not tmp_csv.exists():
            raise SystemExit("anonymeter worker produced no output")
        return pd.read_csv(tmp_csv)
    finally:
        tmp_csv.unlink(missing_ok=True)


def main():
    args = C.build_parser("Figure 1 panel c — Anonymeter privacy risk").parse_args()

    interpreter = find_interpreter()
    if interpreter is None:
        raise SystemExit(
            "No interpreter with anonymeter found.\n"
            f"Tried: {', '.join(INTERPRETER_CANDIDATES)}.\n"
            "Install anonymeter==1.0.0 in a separate venv and point this script "
            "at it:\n    ANONYMETER_PYTHON=/path/to/venv/bin/python python3 "
            f"{__file__}")

    # setup() is called for its side effect: filling the synthetic_full cache the
    # attacks read. The dataset arrays themselves stay in this process.
    _, methods = C.setup(args, stages={"full"},
                         description="Figure 1 panel c — empirical privacy (Anonymeter)")
    print(f"\n> Panel c (privacy) — attacks run under {interpreter}")
    df = compute(methods, interpreter)
    # The worker adds 'Real'; merge on whatever it actually produced.
    C.save_csv(C.RES / "panel_b_privacy.csv", df,
               sorted(set(df["method"].unique())))


if __name__ == "__main__":
    main()

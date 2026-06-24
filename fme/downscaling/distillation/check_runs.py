# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""Inspect distillation runs on Weights & Biases.

Helper for the MoE distillation effort (see MOE_DISTILLATION_STATUS.md).  Pure
wandb — does NOT import fastgen, so it runs in the plain ``fme`` conda env.

Usage:
    # List the most recent runs in the project (id | state | step | name):
    conda run -n fme python -m fme.downscaling.distillation.check_runs --list

    # Compare validation/training metrics across one or more runs:
    conda run -n fme python -m fme.downscaling.distillation.check_runs \
        syz25njv r9lerxok z5usj8so

The default metric set mirrors the comparison used to diagnose the flat-CRPS
problem (per-variable CRPS, spectral MAE, precip tail).  Override with
``--keys k1 k2 ...``.
"""

from __future__ import annotations

import argparse

import wandb

DEFAULT_PROJECT = "ai2cm/fastgen"

DEFAULT_KEYS = [
    "train/total_loss",
    "train/f_distill_loss",
    "train/gan_loss_gen",
    "val/crps_PRMSL",
    "val/crps_PRATEsfc",
    "val/crps_eastward_wind_at_ten_meters",
    "val/crps_northward_wind_at_ten_meters",
    "val/spec_mae_mean",
    "val/spec_mae_hi_PRATEsfc",
    "val/tail_99.99_PRATEsfc",
]


def list_runs(project: str, limit: int) -> None:
    api = wandb.Api()
    runs = list(api.runs(project, order="-created_at")[:limit])
    print(f"{'id':12s} {'state':9s} {'step':>7s}  name")
    for r in runs:
        step = r.summary.get("_step")
        print(f"{r.id:12s} {r.state:9s} {str(step):>7s}  {r.name}")


def compare_runs(project: str, run_ids: list[str], keys: list[str]) -> None:
    api = wandb.Api()
    for rid in run_ids:
        r = api.run(f"{project}/{rid}")
        runtime_min = round(r.summary.get("_runtime", 0) / 60, 1)
        print("=" * 76)
        print(
            f"{r.name}\n  {rid} | {r.state} | step {r.summary.get('_step')} "
            f"| runtime {runtime_min} min"
        )
        hist = r.history(keys=keys, samples=4000, pandas=True)
        for key in keys:
            if key in hist.columns:
                series = hist[key].dropna()
                if len(series):
                    argmin_frac = series.values.argmin() / max(len(series) - 1, 1)
                    print(
                        f"  {key:42s} first={series.iloc[0]:.4g}  "
                        f"best={series.min():.4g}@{argmin_frac:.0%}  "
                        f"last={series.iloc[-1]:.4g}  n={len(series)}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_ids", nargs="*", help="W&B run ids to compare (e.g. syz25njv r9lerxok)"
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="entity/project")
    parser.add_argument(
        "--list", action="store_true", help="list recent runs instead of comparing"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="number of runs to list (with --list)"
    )
    parser.add_argument(
        "--keys", nargs="+", default=DEFAULT_KEYS, help="metric keys to report"
    )
    args = parser.parse_args()

    if args.list or not args.run_ids:
        list_runs(args.project, args.limit)
        if not args.run_ids:
            return
    if args.run_ids:
        compare_runs(args.project, args.run_ids, args.keys)


if __name__ == "__main__":
    main()

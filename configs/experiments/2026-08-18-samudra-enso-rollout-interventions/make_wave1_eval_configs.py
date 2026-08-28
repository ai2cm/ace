#!/usr/bin/env python3
"""Generate coupled-evaluator configs to score wave-1 arms on the scouting ICs.

One config per (arm, year): free-running coupled rollout from the arm's
fine-tuned checkpoint over the same 12 monthly initial conditions per year the
diagnosis and the prescribe probe used (years 0233/0246/0250, the three most
ENSO-active of the verification window), writing monthly-mean SST plus the
ocean interior (zos + four thermocline levels). Scoring then reuses the
diagnosis report's canonical Nino3.4 machinery, so arm curves drop directly
onto the existing free-run baseline and prescribe-probe results.

The loader/aggregator blocks come from the arm's own training config's
inference section (already valid against main's schema); only the ICs, rollout
length and writer change.

Usage: python make_wave1_eval_configs.py --arms wint5 noohc
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

# The same 12 monthly ICs per year the AR-SST/prescribe evals used: the last
# 5-day step of the prior December, then approximately-monthly 5-day-grid steps.
IC_PATTERN = [
    ("{p}-12-29", 12),
    ("{y}-02-02", 0),
    ("{y}-02-27", 0),
    ("{y}-03-29", 0),
    ("{y}-04-28", 0),
    ("{y}-06-02", 0),
    ("{y}-07-02", 0),
    ("{y}-08-01", 0),
    ("{y}-08-31", 0),
    ("{y}-09-30", 0),
    ("{y}-10-30", 0),
    ("{y}-11-29", 0),
]
YEARS = [233, 246, 250]
OCEAN_WRITE_NAMES = ["sst", "zos", "thetao_2", "thetao_4", "thetao_6", "thetao_8"]


def ic_times(year: int) -> list[str]:
    out = []
    for pat, _ in IC_PATTERN:
        d = pat.format(y=f"{year:04d}", p=f"{year - 1:04d}")
        out.append(f"{d}T12:00:00")
    return out


def make_eval_config(arm: str, year: int) -> dict:
    with open(HERE / "wave1_configs" / f"{arm}.yaml") as f:
        train = yaml.safe_load(f)
    inf = copy.deepcopy(train["inference"])
    inf["loader"]["start_indices"] = {"times": ic_times(year)}
    inf["n_coupled_steps"] = 146
    inf["coupled_steps_in_memory"] = 1
    return {
        "experiment_dir": "/results",
        "checkpoint_path": "/ckpt.tar",
        "n_coupled_steps": 146,
        "coupled_steps_in_memory": 1,
        "loader": inf["loader"],
        "aggregator": inf["aggregator"],
        "data_writer": {
            "ocean": {
                "save_prediction_files": False,
                "save_monthly_files": True,
                "names": OCEAN_WRITE_NAMES,
            },
            "atmosphere": {
                "save_prediction_files": False,
                "save_monthly_files": False,
            },
        },
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": True,
            "log_to_file": True,
            "project": "ace-samudra-coupled-cm4",
            "entity": "ai2cm",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "wave1_eval_configs")
    args = ap.parse_args()
    for arm in args.arms:
        d = args.out_dir / arm
        d.mkdir(parents=True, exist_ok=True)
        for y in YEARS:
            p = d / f"yr{y:04d}.yaml"
            p.write_text(yaml.safe_dump(make_eval_config(arm, y), sort_keys=False))
            print(f"wrote {p}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate per-year coupled AR SST Nino eval configs.

Job layout mirrors ``exp/samudrace-enso-skill`` /
``2026-02-17-samudrace-enso-eval`` (one year per gantry job, 12 monthly ICs,
``n_coupled_steps: 146``), but the year window matches our coupled FT
held-out split:

* train: 0256-01-03 → 0349-01-01 (``coupled-finetune-atmos.yaml``)
* validation / OOS: 0251-01-03 → 0255-12-29

So defaults are years 0251–0255 (not the template's 0244–0254).

IC times are the closest 5-day ocean samples to each calendar-month start on
the CM4 1pctCO2 ocean zarr (stable noleap alignment used by the reference
enso-eval configs). Atmosphere paths match our coupled FT checkpoint
(sea_ice); the nino-leads zarr is included because FT ocean ``out_names``
still list ``nino34_lead_*``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# Coupled FT validation window (inclusive calendar years for monthly ICs).
YEAR_START = 251
YEAR_END = 255


def monthly_ic_times(year: int) -> list[str]:
    """12 monthly IC timestamps for ``year`` on the CM4 1pctCO2 5-day ocean grid.

    Matches the closest-to-month-start selection in
    ``make_year_configs-1pctc02.py`` on ``exp/samudrace-enso-skill`` for this
    zarr (noleap, 5-day stride starting 0211-01-03).
    """
    return [
        f"{year - 1:04d}-12-29T12:00:00",  # ~Jan 1
        f"{year:04d}-02-02T12:00:00",  # ~Feb 1
        f"{year:04d}-02-27T12:00:00",  # ~Mar 1
        f"{year:04d}-03-29T12:00:00",  # ~Apr 1
        f"{year:04d}-04-28T12:00:00",  # ~May 1
        f"{year:04d}-06-02T12:00:00",  # ~Jun 1
        f"{year:04d}-07-02T12:00:00",  # ~Jul 1
        f"{year:04d}-08-01T12:00:00",  # ~Aug 1
        f"{year:04d}-08-31T12:00:00",  # ~Sep 1
        f"{year:04d}-09-30T12:00:00",  # ~Oct 1
        f"{year:04d}-10-30T12:00:00",  # ~Nov 1
        f"{year:04d}-11-29T12:00:00",  # ~Dec 1
    ]


def make_config(times: list[str]) -> dict:
    return {
        "experiment_dir": "/results",
        "n_coupled_steps": 146,
        "coupled_steps_in_memory": 1,
        "checkpoint_path": "/ckpt.tar",
        "aggregator": {
            "log_histograms": False,
            "log_zonal_mean_images": False,
            "log_global_mean_time_series": False,
            "log_global_mean_norm_time_series": False,
        },
        "data_writer": {
            "ocean": {
                "save_prediction_files": False,
                "save_monthly_files": True,
                "names": ["sst"],
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
        "loader": {
            "num_data_workers": 1,
            "dataset": {
                "ocean": {
                    "merge": [
                        {
                            "data_path": "/climate-default",
                            "file_pattern": (
                                "2025-10-21-cm4-1pctCO2-140yr-no-smoothing-"
                                "coupled-ocean.zarr"
                            ),
                            "engine": "zarr",
                        },
                        {
                            "data_path": "/climate-default",
                            "file_pattern": (
                                "2025-10-16-cm4-1pctCO2-140yr-ocean-no-smoothing.zarr"
                            ),
                            "engine": "zarr",
                        },
                        {
                            "data_path": "/climate-default",
                            "file_pattern": (
                                "2026-07-14-cm4-1pctCO2-140yr-ocean-no-smoothing-"
                                "nino-leads.zarr"
                            ),
                            "engine": "zarr",
                        },
                    ]
                },
                "atmosphere": {
                    "merge": [
                        {
                            "data_path": "/climate-default",
                            "file_pattern": (
                                "2025-10-21-cm4-1pctCO2-140yr-no-smoothing-"
                                "coupled-sea_ice.zarr"
                            ),
                            "engine": "zarr",
                        },
                        {
                            "data_path": "/climate-default",
                            "file_pattern": (
                                "2025-06-18-CM4-1pctCO2-atmosphere-land-1deg-"
                                "8layer-140yr.zarr"
                            ),
                            "engine": "zarr",
                        },
                    ]
                },
            },
            "start_indices": {"times": times},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "ar_sst_year_configs",
    )
    parser.add_argument("--year-start", type=int, default=YEAR_START)
    parser.add_argument("--year-end", type=int, default=YEAR_END)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Drop stale years from a previous window (e.g. 0244–0250).
    for stale in args.output_dir.glob("evaluator-config-1pct-ar-sst-yr*.yaml"):
        stale.unlink()

    for year in range(args.year_start, args.year_end + 1):
        times = monthly_ic_times(year)
        config = make_config(times)
        out = args.output_dir / f"evaluator-config-1pct-ar-sst-yr{year:04d}.yaml"
        header = (
            f"# Autoregressive SST Nino eval for year {year:04d}.\n"
            "# OOS window for coupled FT (train starts 0256): validation years\n"
            "# 0251–0255. Job layout matches exp/samudrace-enso-skill\n"
            "# 2026-02-17-samudrace-enso-eval (12 monthly ICs, 146 ocean steps).\n"
            "# Checkpoint: coupled FT 01KY3DATM3CAEA479JQZQDPT9W (/ckpt.tar).\n"
            "# Atmosphere data matches coupled FT (sea_ice), not interpolate_sst.\n"
        )
        with open(out, "w") as file:
            file.write(header)
            yaml.dump(config, file, sort_keys=False, default_flow_style=False)
        print(f"Wrote {out} ({len(times)} ICs, first={times[0]}, last={times[-1]})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate per-year coupled AR SST Nino eval configs.

Job layout mirrors ``exp/samudrace-enso-skill`` /
``2026-02-17-samudrace-enso-eval`` (one year per gantry job, 12 monthly ICs,
``n_coupled_steps: 146``).

CM4 1pctCO2 zarr spans model years **0211–0350**. Samudra + coupled FT use:

* **train:** 0256-01-03 → 0349-01-01
* **validation (early-stop metric):** 0251-01-03 → 0255-12-29
* **extended OOS (never train/val):** 0211–0250

Presets (``--oos-window``):

* ``val`` — 0251–0255 (official FT validation; default)
* ``extended20`` — 0231–0250 (20 calendar years, strictly unseen)
* ``extended40`` — 0211–0250 (full pre-train window)

IC times are the closest 5-day ocean samples to each calendar-month start on
the CM4 1pctCO2 ocean zarr. Atmosphere paths match our coupled FT checkpoint
(sea_ice); the nino-leads zarr is included because FT ocean ``out_names``
still list ``nino34_lead_*``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# OOS year-window presets (inclusive calendar years, 12 monthly ICs each).
OOS_WINDOWS: dict[str, tuple[int, int, str]] = {
    "val": (
        251,
        255,
        "official coupled FT validation (early-stop metric; train starts 0256)",
    ),
    "extended20": (
        231,
        250,
        "20y strictly OOS (never in Samudra or coupled FT train/val)",
    ),
    "extended40": (
        211,
        250,
        "40y pre-train OOS (0211–0250; zarr starts at 0211)",
    ),
}
YEAR_START = OOS_WINDOWS["val"][0]
YEAR_END = OOS_WINDOWS["val"][1]


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


def _resolve_year_range(
    oos_window: str | None, year_start: int | None, year_end: int | None
) -> tuple[int, int, str]:
    if oos_window is not None:
        if oos_window not in OOS_WINDOWS:
            choices = ", ".join(sorted(OOS_WINDOWS))
            raise ValueError(f"Unknown --oos-window {oos_window!r}; choose: {choices}")
        start, end, note = OOS_WINDOWS[oos_window]
        return start, end, note
    start = YEAR_START if year_start is None else year_start
    end = YEAR_END if year_end is None else year_end
    return start, end, f"custom years {start:04d}–{end:04d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "ar_sst_year_configs",
    )
    parser.add_argument(
        "--oos-window",
        choices=sorted(OOS_WINDOWS),
        default=None,
        help="Preset OOS year range (overrides --year-start/--year-end).",
    )
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    args = parser.parse_args()

    year_start, year_end, window_note = _resolve_year_range(
        args.oos_window, args.year_start, args.year_end
    )
    n_years = year_end - year_start + 1
    print(f"Generating {n_years} year configs ({window_note})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Drop stale years outside the requested window.
    for stale in args.output_dir.glob("evaluator-config-1pct-ar-sst-yr*.yaml"):
        year_token = stale.stem.split("yr")[-1]
        year = int(year_token)
        if year < year_start or year > year_end:
            stale.unlink()

    for year in range(year_start, year_end + 1):
        times = monthly_ic_times(year)
        config = make_config(times)
        out = args.output_dir / f"evaluator-config-1pct-ar-sst-yr{year:04d}.yaml"
        header = (
            f"# Autoregressive SST Nino eval for year {year:04d}.\n"
            f"# OOS window: {window_note}.\n"
            "# 12 monthly ICs, 146 ocean steps (~2y rollout). Layout:\n"
            "# exp/samudrace-enso-skill/2026-02-17-samudrace-enso-eval.\n"
            "# Checkpoint: coupled FT 01KY3DATM3CAEA479JQZQDPT9W (/ckpt.tar).\n"
        )
        with open(out, "w") as file:
            file.write(header)
            yaml.dump(config, file, sort_keys=False, default_flow_style=False)
        print(f"Wrote {out} ({len(times)} ICs, first={times[0]}, last={times[-1]})")


if __name__ == "__main__":
    main()

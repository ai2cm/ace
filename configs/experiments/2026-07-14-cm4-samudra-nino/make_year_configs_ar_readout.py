#!/usr/bin/env python3
"""Generate per-year coupled AR *readout* eval configs.

Same job layout as ``make_year_configs_ar_sst.py`` (one year per gantry job,
12 monthly ICs, ``n_coupled_steps: 146``), but the data writer saves
native-cadence prediction files instead of monthly means:

* ocean (5-day): ``sst`` + ``nino34_lead_01..12`` — the MLP readout's forecast
  at every AR step, for the "readout skill vs AR step" diagnostic (where in
  the rollout does the coupled state lose ENSO information?).
* atmosphere (6-hr): surface wind stress components — for the coupled-rollout
  Bjerknes/bridge audit (wind-stress response to SST anomalies vs CM4 truth).

~24 GB per year job (atmosphere dominates). No monthly files: monthly means
are reproducible offline from the native output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from make_year_configs_ar_sst import (
    OOS_WINDOWS,
    _resolve_year_range,
    make_config,
    monthly_ic_times,
)

OCEAN_NAMES = ["sst"] + [f"nino34_lead_{k:02d}" for k in range(1, 13)]
ATMOS_NAMES = ["eastward_surface_wind_stress", "northward_surface_wind_stress"]
# Surface energy-flux components exchanged across the coupled interface. Saved
# as monthly means only: the flux *response* to SST (damping / Bjerknes) is a
# monthly-scale regression, while the interface-noise question is answered by
# the natively-saved wind stress. Native 6-hr fluxes would be ~8.5 GB per
# variable per year.
ATMOS_FLUX_NAMES = [
    "LHTFLsfc",
    "SHTFLsfc",
    "DSWRFsfc",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFsfc",
]


def make_readout_config(times: list[str], with_fluxes: bool = False) -> dict:
    config = make_config(times)
    config["data_writer"] = {
        "ocean": {
            "save_prediction_files": True,
            "save_monthly_files": False,
            "names": OCEAN_NAMES,
        },
        "atmosphere": {
            "save_prediction_files": not with_fluxes,
            "save_monthly_files": with_fluxes,
            "names": ATMOS_NAMES + (ATMOS_FLUX_NAMES if with_fluxes else []),
        },
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "ar_readout_year_configs",
    )
    parser.add_argument(
        "--oos-window",
        choices=sorted(OOS_WINDOWS),
        default=None,
        help="Preset OOS year range (overrides --year-start/--year-end).",
    )
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    parser.add_argument(
        "--with-fluxes",
        action="store_true",
        help=(
            "Save monthly-mean surface energy fluxes (plus stress) instead of "
            "native-cadence stress, for the coupled flux-response audit."
        ),
    )
    args = parser.parse_args()

    year_start, year_end, window_note = _resolve_year_range(
        args.oos_window, args.year_start, args.year_end
    )
    print(f"Generating {year_end - year_start + 1} year configs ({window_note})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("evaluator-config-1pct-ar-readout-yr*.yaml"):
        year = int(stale.stem.split("yr")[-1])
        if year < year_start or year > year_end:
            stale.unlink()

    for year in range(year_start, year_end + 1):
        times = monthly_ic_times(year)
        config = make_readout_config(times, with_fluxes=args.with_fluxes)
        out = args.output_dir / f"evaluator-config-1pct-ar-readout-yr{year:04d}.yaml"
        header = (
            f"# Coupled AR readout eval for year {year:04d}.\n"
            f"# OOS window: {window_note}.\n"
            "# 12 monthly ICs, 146 ocean steps. Native-cadence writer: ocean\n"
            "# sst + nino34_lead_01..12 (5-day), atmosphere wind stress (6-hr).\n"
            "# Checkpoint: coupled FT 01KY3DATM3CAEA479JQZQDPT9W (/ckpt.tar).\n"
        )
        with open(out, "w") as file:
            file.write(header)
            yaml.dump(config, file, sort_keys=False, default_flow_style=False)
        print(f"Wrote {out} ({len(times)} ICs, first={times[0]}, last={times[-1]})")


if __name__ == "__main__":
    main()

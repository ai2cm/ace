"""Generate coupled ERA5+UFS *inference* configs (one per year).

These configs are consumed by ``fme.coupled.inference`` and run the model
forward freely from a set of initial conditions, using prescribed forcing,
WITHOUT comparing against reference target data. For evaluation against target
data, see ``make_evaluator_config.py``.

The key structural difference from the evaluator config is that inference uses
separate ``initial_condition`` (IC paths + start indices) and ``forcing_loader``
(forcing datasets) sections instead of a single ``loader`` section.
"""

import os
from datetime import datetime

import cftime
import numpy as np
import xarray as xr
import yaml

# ==============================
# USER SETTINGS
# ==============================

ZARR_PATH = "gs://vcm-ml-intermediate/2026-06-03-ufs-replay-ocean-1deg-19level-5day-1994-2023.zarr/"
YEAR_START = 2022
YEAR_END = 2023  # inclusive
OUTPUT_DIR = (
    "./configs/ufs-era5-fully-coupled-v0"  # os.path.dirname(os.path.abspath(__file__))
)

# Mount point and file patterns for the datasets (as mounted in the run env).
DATA_PATH = "/climate-default"
OCEAN_FILE = "2026-06-03-ufs-replay-ocean-1deg-19level-5day-1994-2023.zarr"
ATMOSPHERE_FILE = "2026-03-19-era5-1deg-8layer-1940-2025.zarr"

# Number of times to repeat the ocean forcing dataset in time. The ocean is
# prognostic, but its static forcing fields (e.g. land_fraction,
# sea_surface_fraction, deptho) are read from this dataset. Repeating extends
# the dataset's time axis so the run window can start near the end of the
# ocean record without exceeding the available timesteps.
OCEAN_FORCING_N_REPEATS = 2

# Samudra next_step_forcing_names, using ACE atmosphere output names
# (wind stress renamed from *_surface_wind_stress to *_surface_stress).
SAMUDRA_ATMOSPHERE_FLUX_NAMES = [
    "DLWRFsfc",
    "DSWRFsfc",
    "ULWRFsfc",
    "USWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "PRATEsfc",
    "eastward_surface_stress",
    "northward_surface_stress",
    "total_frozen_precipitation_rate",
]

# ==============================
# HELPERS
# ==============================


def format_time(t):
    """
    Format datetime, cftime, or numpy datetime64 to:
    '1994-01-03T12:00:00'.
    """
    if isinstance(t, np.datetime64):
        t = datetime.utcfromtimestamp(t.astype("datetime64[s]").astype(int))
    return (
        f"{t.year:04d}-"
        f"{t.month:02d}-"
        f"{t.day:02d}T"
        f"{t.hour:02d}:"
        f"{t.minute:02d}:"
        f"{t.second:02d}"
    )


def get_month_start_targets(year):
    return [(year, month, 1) for month in range(1, 13)]


def find_closest_times(ds, year):
    time = ds["time"]

    # Load once for speed
    time_vals = time.values
    is_cftime = isinstance(time_vals[0], cftime.datetime)

    selected_times = []

    for year_, month, day in get_month_start_targets(year):
        # Build ideal timestamp matching calendar type
        if is_cftime:
            ideal = type(time_vals[0])(year_, month, day)
        else:
            ideal = np.datetime64(datetime(year_, month, day))

        # Compute absolute difference
        idx = int(np.argmin(np.abs(time.values - ideal)))

        closest = time_vals[idx]
        selected_times.append(format_time(closest))

    return selected_times


def make_config(times):
    ocean_path = f"{DATA_PATH}/{OCEAN_FILE}"
    atmosphere_path = f"{DATA_PATH}/{ATMOSPHERE_FILE}"
    return {
        "experiment_dir": "/results",
        "n_coupled_steps": 147,  # 30 years with 5-day ocean steps
        "coupled_steps_in_memory": 1,
        "checkpoint_path": "/ckpt.tar",
        "n_ensemble_per_ic": 4,
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": True,
            "log_to_file": True,
            "project": "ace-samudra-ufs",
            "entity": "ai2cm",
        },
        "initial_condition": {
            "ocean": {
                "path": ocean_path,
                "engine": "zarr",
            },
            "atmosphere": {
                "path": atmosphere_path,
                "engine": "zarr",
            },
            "start_indices": {"times": times},
        },
        # The ocean is prognostic, but its static forcing fields are read from
        # the ocean forcing dataset, so it is included here. n_repeats extends
        # the ocean time axis past the inference window (see OCEAN_FORCING_N_REPEATS).
        "forcing_loader": {
            "num_data_workers": 4,
            "atmosphere": {
                "dataset": {
                    "data_path": DATA_PATH,
                    "file_pattern": ATMOSPHERE_FILE,
                    "engine": "zarr",
                },
            },
            "ocean": {
                "dataset": {
                    "data_path": DATA_PATH,
                    "file_pattern": OCEAN_FILE,
                    "engine": "zarr",
                    "n_repeats": OCEAN_FORCING_N_REPEATS,
                },
            },
        },
        "aggregator": {
            "log_global_mean_time_series": True,
        },
        "data_writer": {
            "ocean": {
                "save_prediction_files": False,
                "save_monthly_files": True,
                "names": [
                    "sst",
                    "zos",
                    "thetao_0",
                    "vo_0",
                    "uo_0",
                ],
            },
            "atmosphere": {
                "save_prediction_files": False,
                "save_monthly_files": True,
                "names": SAMUDRA_ATMOSPHERE_FLUX_NAMES,
            },
        },
    }


# ==============================
# MAIN
# ==============================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Opening dataset...")
    ds = xr.open_zarr(ZARR_PATH)

    for year in range(YEAR_START, YEAR_END + 1):
        print(f"Processing year {year}...")

        times = find_closest_times(ds, year)

        config = make_config(times)

        filename = os.path.join(
            OUTPUT_DIR,
            f"inference-config-coupled-ERA5-UFS-v0-out-of-sample-yr{year:04d}.yaml",
        )

        with open(filename, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        print(f"  Wrote {filename}")


if __name__ == "__main__":
    main()

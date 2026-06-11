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
YEAR_START = 2010
YEAR_END = 2020  # inclusive
OUTPUT_DIR = (
    "./configs/ufs-era5-fully-coupled-v0"  # os.path.dirname(os.path.abspath(__file__))
)

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
    return {
        "experiment_dir": "/results",
        "n_coupled_steps": 147,  # 30 years with 5-day ocean steps
        "coupled_steps_in_memory": 1,
        "checkpoint_path": "/ckpt.tar",
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": True,
            "log_to_file": True,
            "project": "ace-samudra-ufs",
            "entity": "ai2cm",
        },
        "loader": {
            "num_data_workers": 4,
            "dataset": {
                "ocean": {
                    "data_path": "/climate-default",
                    "file_pattern": (
                        "2026-06-03-ufs-replay-ocean-1deg-19level-5day-1994-2023.zarr"
                    ),
                    "engine": "zarr",
                },
                "atmosphere": {
                    "data_path": "/climate-default",
                    "file_pattern": "2026-03-19-era5-1deg-8layer-1940-2025.zarr",
                    "engine": "zarr",
                },
            },
            "start_indices": {"times": times},
        },
        "aggregator": {
            "log_zonal_mean_images": True,
            "log_histograms": True,
        },
        "data_writer": {
            "ocean": {
                "save_prediction_files": False,
                "save_monthly_files": False,
            },
            "atmosphere": {
                "save_prediction_files": False,
                "save_monthly_files": False,
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
            f"evaluator-config-coupled-ERA5-UFS-v0-yr{year:04d}.yaml",
        )

        with open(filename, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        print(f"  Wrote {filename}")


if __name__ == "__main__":
    main()

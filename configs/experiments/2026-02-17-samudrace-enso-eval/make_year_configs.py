import os
from datetime import datetime

import cftime
import numpy as np
import xarray as xr
import yaml

# ==============================
# USER SETTINGS
# ==============================

ZARR_PATH = "gs://vcm-ml-intermediate/2025-04-16-cm4-piControl-ocean-200yr-dataset.zarr"
YEAR_START = 311
YEAR_END = 320  # inclusive
OUTPUT_DIR = "./configs"

# ==============================
# HELPERS
# ==============================


def format_time(t):
    """
    Format datetime or cftime object to:
    '0311-05-31T00:00:00'.
    """
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
        diff = np.abs(time - ideal)
        idx = diff.argmin().item()

        closest = time_vals[idx]
        selected_times.append(format_time(closest))

    return selected_times


def make_config(year, times):
    return {
        "experiment_dir": "/results",
        "n_coupled_steps": 146,
        "coupled_steps_in_memory": 1,
        "checkpoint_path": "/ckpt.tar",
        "aggregator": {"log_histograms": True},
        "atmosphere_stepper_override": {
            "prescribed_prognostic_names": [
                "DLWRFsfc",
                "DSWRFsfc",
                "ULWRFsfc",
                "USWRFsfc",
                "LHTFLsfc",
                "SHTFLsfc",
                "PRATEsfc",
                "eastward_surface_wind_stress",
                "northward_surface_wind_stress",
            ],
        },
        "data_writer": {
            "ocean": {
                "save_prediction_files": False,
                "save_monthly_files": True,
                "names": [
                    "sst",
                    "zos",
                    "thetao_0",
                    "thetao_1",
                    "thetao_2",
                    "thetao_3",
                    "thetao_4",
                    "thetao_5",
                    "thetao_6",
                    "vo_0",
                    "vo_1",
                    "vo_2",
                    "vo_3",
                    "vo_4",
                    "vo_5",
                    "vo_6",
                    "uo_0",
                    "uo_1",
                    "uo_2",
                    "uo_3",
                    "uo_4",
                    "uo_5",
                    "uo_6",
                ],
            },
            "atmosphere": {
                "save_prediction_files": False,
                "save_monthly_files": True,
                "names": [
                    "eastward_surface_wind_stress",
                    "northward_surface_wind_stress",
                ],
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
                                "2025-06-03-cm4-piControl-200yr-coupled-ocean" ".zarr"
                            ),
                            "engine": "zarr",
                        },
                        {
                            "data_path": "/climate-default",
                            "file_pattern": (
                                "2025-04-16-cm4-piControl-ocean-200yr-dataset" ".zarr"
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
                                "2025-06-03-cm4-piControl-200yr-coupled-"
                                "interpolate_sst-atmosphere.zarr"
                            ),
                            "engine": "zarr",
                        },
                        {
                            "data_path": "/climate-default",
                            "file_pattern": (
                                "2025-03-21-CM4-piControl-atmosphere-land-"
                                "1deg-8layer-200yr.zarr"
                            ),
                            "engine": "zarr",
                        },
                    ]
                },
            },
            "start_indices": {"times": times},
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

        config = make_config(year, times)

        filename = os.path.join(
            OUTPUT_DIR,
            f"evaluator-config-override-atmosphere-forcing-winds-only-yr{year:04d}.yaml",
        )

        with open(filename, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        print(f"  Wrote {filename}")


if __name__ == "__main__":
    main()

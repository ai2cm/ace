# mypy: ignore-errors
# ruff: noqa
# ruff: noqa: E501
"""Evaluator configs for the 2002-2010 clean-window S2S hindcast of the GLO12 coupled SamudrACE checkpoint: one config per
year, 12 starts (the first 5-day GLORYS12 state of each month, day 1-5) x 8 noise members, 18 coupled steps (90 days),
daily atmosphere output (MJO: OLR + hybrid-level winds + PRESsfc; weeks 3-6: TMP2m, PRATEsfc; 10 m winds and surface
temperature for the coupling diagnostics) and 5-day ocean output (SST, upper 372 m temperature, ice, zos).

    python make_hindcast_configs.py   # writes coupled-evaluator-config-glorys12-era5-hindcast-<year>.yaml
"""

import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
OCEAN_ZARR = "2026-10-27-glorys12-ocean-4deg-19level-5day-2001-2011-cm4vars.zarr"
ERA5_ZARR = "2026-10-27-era5-4deg-8layer-daily-glorys12-coupled-2001-2011.zarr"
N_MEMBERS = 8
N_STEPS = 18  # 90 days
# first GLORYS12 5-day state (1993-01-01 + 5 k days) in each month
import pandas as pd

grid = pd.date_range("1993-01-01", "2011-12-31", freq="5D")
starts = {
    y: [grid[(grid.year == y) & (grid.month == m)][0] for m in range(1, 13)]
    for y in range(2002, 2011)
}
assert all(d.day <= 5 for ds in starts.values() for d in ds)

ATMOS_NAMES = [
    "ULWRFtoa",
    "PRESsfc",
    "TMP2m",
    "PRATEsfc",
    "surface_temperature",
    "UGRD10m",
    "VGRD10m",
    "eastward_surface_stress",
] + [f"eastward_wind_{k}" for k in range(8)]
OCEAN_NAMES = (
    ["sst"]
    + [f"thetao_{k}" for k in range(9)]
    + ["so_0", "zos", "ocean_sea_ice_fraction", "sea_ice_volume"]
)

for y, ds in starts.items():
    cfg = {
        "experiment_dir": "/results",
        "checkpoint_path": "/ckpt.tar",
        "n_coupled_steps": N_STEPS,
        "coupled_steps_in_memory": 6,
        "loader": {
            "num_data_workers": 2,
            "dataset": {
                "ocean": {
                    "data_path": "/glorys",
                    "file_pattern": OCEAN_ZARR,
                    "engine": "zarr",
                },
                "atmosphere": {
                    "data_path": "/era5glorys",
                    "file_pattern": ERA5_ZARR,
                    "engine": "zarr",
                },
            },
            "start_indices": {
                "times": [
                    d.strftime("%Y-%m-%dT00:00:00")
                    for d in ds
                    for _ in range(N_MEMBERS)
                ]
            },
        },
        "aggregator": {"log_zonal_mean_images": False, "log_histograms": False},
        "data_writer": {
            "atmosphere": {
                "names": ATMOS_NAMES,
                "save_prediction_files": True,
                "save_monthly_files": False,
            },
            "ocean": {
                "names": OCEAN_NAMES,
                "save_prediction_files": True,
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
    p = HERE / f"coupled-evaluator-config-glorys12-era5-hindcast-{y}.yaml"
    header = (
        f"# 2002-2010 clean-window S2S hindcast of the GLO12 coupled SamudrACE checkpoint, year {y}: 12 monthly starts (first\n"
        f"# GLORYS12 5-day state of each month) x {N_MEMBERS} noise members, {N_STEPS} five-day steps (90 d), daily atmosphere + 5-day ocean\n"
        "# prediction files for MJO (RMM), SST CRPS, weeks 3-6 T2m/precip. GLORYS12 reanalysis ICs (GLO12 starts 2022). Troy 2026-10-27.\n"
    )
    p.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
    print("wrote", p.name, [d.strftime("%m-%d") for d in ds][:3], "...")

# mypy: ignore-errors
# ruff: noqa
"""Same 2002-2010 S2S hindcast design for the UFS-trained coupled SamudrACE (spb-ufsstats, results 01M5DXQST7WCYTXYXKMEF3KQ13)
started from UFS replay states (its own holdout decade, in-distribution ICs): first UFS 5-day state (18Z) of each month x 8
members x 18 steps, same prediction-file variables. Writes coupled-evaluator-config-ufs-era5-hindcast-<year>.yaml."""

import pathlib
import pandas as pd
import yaml

HERE = pathlib.Path(__file__).parent
N_MEMBERS, N_STEPS = 8, 18
grid = pd.date_range("1994-01-05T18:00", "2011-12-31T18:00", freq="5D")
starts = {
    y: [grid[(grid.year == y) & (grid.month == m)][0] for m in range(1, 13)]
    for y in range(2002, 2011)
}
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
                    "data_path": "/ufs4deg",
                    "file_pattern": "2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr",
                    "engine": "zarr",
                },
                "atmosphere": {
                    "data_path": "/era5",
                    "file_pattern": "2026-10-14-era5-4deg-8layer-daily-ufs-coupled-1994-2023.zarr",
                    "engine": "zarr",
                },
            },
            "start_indices": {
                "times": [
                    d.strftime("%Y-%m-%dT%H:%M:%S")
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
    p = HERE / f"coupled-evaluator-config-ufs-era5-hindcast-{y}.yaml"
    p.write_text(
        f"# UFS-trained coupled SamudrACE (spb-ufsstats) on its own 2002-2010 holdout from UFS replay ICs, year {y}: 12 monthly starts x {N_MEMBERS} members x {N_STEPS} five-day steps; same outputs as the GLORYS12-IC hindcast. Troy 2026-10-27.\n"
        + yaml.safe_dump(cfg, sort_keys=False)
    )
    print("wrote", p.name, [d.strftime("%m-%dT%H") for d in ds][:2])

#!/usr/bin/env python
"""240-initial-condition coupled forecast configs at 4 degrees: piControl years
0241-0260 (unseen: training 156-236, validation 236-241), 12 monthly ICs per year,
146 coupled steps (24 months), one config per year. Two checkpoint forms:
  two    : ocean-only checkpoint at /ocean_ckpt.tar + atmosphere at /atmos_ckpt.tar
  single : one coupled checkpoint at /ckpt.tar
Mirrors the 1-degree protocol in
2026-08-18-samudra-enso-rollout-interventions/wave1_eval_configs/zeroshot240.
IC dates are the first 5-day step on or after each month start (steps fall on
day-of-year 1+5k; January uses day 6 to stay clear of the first step of the year).
"""

import pathlib

import yaml

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "eval" / "zs240"
YEARS = range(241, 261)
IC_DATES = [
    "01-06",
    "02-05",
    "03-02",
    "04-01",
    "05-01",
    "06-05",
    "07-05",
    "08-04",
    "09-03",
    "10-03",
    "11-02",
    "12-02",
]
TWO_CKPT = {
    "ocean": {"timedelta": "5D", "path": "/ocean_ckpt.tar"},
    "atmosphere": {"timedelta": "6h", "path": "/atmos_ckpt.tar"},
    "sst_name": "sst",
    "ocean_fraction_prediction": {
        "sea_ice_fraction_name": "ocean_sea_ice_fraction",
        "land_fraction_name": "land_fraction",
        "sea_ice_fraction_name_in_atmosphere": "sea_ice_fraction",
    },
}
DATASET = {
    "ocean": {
        "merge": [
            {
                "data_path": "/climate-default",
                "file_pattern": "2026-07-22-cm4-picontrol-4deg-coupled-ocean.zarr",
                "engine": "zarr",
            },
            {
                "data_path": "/climate-default",
                "file_pattern": "2026-07-15-om4-picontrol-4deg-ocean-5daily.zarr",
                "engine": "zarr",
            },
        ]
    },
    "atmosphere": {
        "merge": [
            {
                "data_path": "/climate-default",
                "file_pattern": "2026-07-22-cm4-picontrol-4deg-coupled-atmosphere.zarr",
                "engine": "zarr",
            },
            {
                "data_path": "/climate-default",
                "file_pattern": (
                    "2026-06-19-CM4-piControl-atmosphere-land-4deg-8layer-200yr.zarr"
                ),
                "engine": "zarr",
            },
        ]
    },
}


def config(year: int, form: str) -> dict:
    return {
        "experiment_dir": "/results",
        "checkpoint_path": "/ckpt.tar" if form == "single" else TWO_CKPT,
        "n_coupled_steps": 146,
        "coupled_steps_in_memory": 1,
        "loader": {
            "num_data_workers": 4,
            "dataset": DATASET,
            "start_indices": {"times": [f"{year:04d}-{d}T00:00:00" for d in IC_DATES]},
        },
        "aggregator": {"log_zonal_mean_images": False, "log_histograms": False},
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
                    "thetao_6",
                    "thetao_8",
                ],
            },
            "atmosphere": {"save_prediction_files": False, "save_monthly_files": False},
        },
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": True,
            "log_to_file": True,
            "project": "ace-samudra-coupled-cm4",
            "entity": "ai2cm",
        },
    }


OUT.mkdir(parents=True, exist_ok=True)
for y in YEARS:
    for form in ("two", "single"):
        (OUT / f"yr{y:04d}-{form}.yaml").write_text(
            yaml.safe_dump(config(y, form), sort_keys=False)
        )
print(f"wrote {2 * len(YEARS)} configs to {OUT}")

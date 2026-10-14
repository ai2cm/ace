#!/usr/bin/env python
# ruff: noqa: E501
"""Compose the coupled fine-tune config for a UFS 4deg ocean + the ERA5 4deg daily ACE2S atmosphere.

Team protocol (the 4deg CM4 coupled fine-tunes): ocean trained with the ensemble loss (CRPS 0.9 + energy 0.1, two
members), atmosphere frozen with zero loss weight, 4 coupled steps per sample, lr 7e-5 with a linear warm-up, EMA.
UFS specifics: train 1994-2001 + 2012-2021 (the 2002-2011 decade is never touched), validation 2022-2023, inline
10-yr coupled rollouts (730 steps) from 2012-01/2012-07/2013-01/2013-07, ocean 5-day / atmosphere daily (5 inner
steps), stress rename + sign flip. The atmosphere's normalization files are its own on weka; the ocean's are the CM4
stats plus the unaccounted_heating stats at /ocean_stats.

    python make_coupled_ft_config.py <ocean arm yaml in ../2026-10-08-ufs-learned-heating> <out name>
"""

import copy
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
STRIP = (
    "loss",
    "optimize_last_step_only",
    "n_ensemble",
    "parameter_init",
    "train_n_forward_steps",
)
UFS = {
    "data_path": "/ufs4deg",
    "file_pattern": "2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr",
    "engine": "zarr",
}
ERA5 = {
    "data_path": "/era5",
    "file_pattern": "2026-10-14-era5-4deg-8layer-daily-ufs-coupled-1994-2023.zarr",
    "engine": "zarr",
}
ATMOS_STEPPER = HERE / "atmos-era5-4deg-daily-stepper.yaml"


def pair(start, stop=None):
    sub = {"start_time": start} | ({"stop_time": stop} if stop else {})
    return {
        "ocean": UFS | {"subset": dict(sub)},
        "atmosphere": ERA5 | {"subset": dict(sub)},
    }


def main(ocean_arm: str, out_name: str) -> None:
    ocean_cfg = yaml.safe_load(
        (
            HERE.parent / "2026-10-08-ufs-learned-heating" / f"{ocean_arm}.yaml"
        ).read_text()
    )
    ocean_st = {k: v for k, v in ocean_cfg["stepper"].items() if k not in STRIP}
    atmos_st = yaml.safe_load(ATMOS_STEPPER.read_text())["stepper"]
    cfg = {
        "experiment_dir": "/results",
        "save_checkpoint": True,
        "validate_using_ema": True,
        "evaluate_before_training": True,
        "ema": {"decay": 0.999},
        "max_epochs": 12,
        "inference": [
            {
                "name": "inference_10yr",
                "weight": 1.0,
                "n_coupled_steps": 730,
                "coupled_steps_in_memory": 2,
                "loader": {
                    "num_data_workers": 2,
                    "start_indices": {
                        "times": [
                            "2012-01-01T18:00:00",
                            "2012-07-04T18:00:00",
                            "2013-01-05T18:00:00",
                            "2013-07-04T18:00:00",
                        ]
                    },
                    "dataset": {"ocean": dict(UFS), "atmosphere": dict(ERA5)},
                },
                "aggregator": {"log_zonal_mean_images": False, "log_histograms": False},
            }
        ],
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": True,
            "log_to_file": True,
            "project": "ace-samudra-coupled-cm4",
            "entity": "ai2cm",
            "metrics_log_dir": "/results/metrics",
        },
        "train_loader": {
            "batch_size": 8,
            "num_data_workers": 2,
            "prefetch_factor": 2,
            "dataset": {
                "concat": [
                    pair("1994-01-05T18:00:00", "2001-12-31T18:00:00"),
                    pair("2012-01-01T18:00:00", "2021-12-31T18:00:00"),
                ]
            },
        },
        "validation": [
            {
                "name": "val_2022_2023",
                "evaluate_all_steps": False,
                "weight": 1.0,
                "loader": {
                    "batch_size": 8,
                    "num_data_workers": 2,
                    "prefetch_factor": 2,
                    "dataset": pair("2022-01-03T18:00:00"),
                },
            }
        ],
        "optimization": {
            "use_gradient_accumulation": True,
            "enable_automatic_mixed_precision": False,
            "lr": 7e-05,
            "max_grad_norm": 1.0,
            "optimizer_type": "AdamW",
            "kwargs": {"fused": True, "weight_decay": 0.01},
            # ~160 iterations per epoch on 18 training years: 300 warm-up iterations ~ 2 epochs (the CM4 template's 1460 would exceed the run)
            "scheduler": {
                "type": "LinearLR",
                "step_each_iteration": True,
                "kwargs": {"start_factor": 0.01, "end_factor": 1, "total_iters": 300},
            },
        },
        "stepper_training": {
            "n_coupled_steps": 4,
            "n_ensemble": 2,
            "optimize_single_component_per_batch": False,
            "ocean": {
                "loss": {
                    "type": "EnsembleLoss",
                    "kwargs": {"crps_weight": 0.9, "energy_score_weight": 0.1},
                },
                "loss_weight": 1.0,
                "n_steps": 4,
                "optimize_last_step_only": False,
                "parameter_init": {"weights_path": "/ocean_ckpt.tar"},
            },
            "atmosphere": {
                "loss": {
                    "type": "EnsembleLoss",
                    "kwargs": {"crps_weight": 0.9, "energy_score_weight": 0.1},
                    "weights": {"h500": 5},
                },
                "loss_weight": 0.0,
                "n_steps": 0,
                "parameter_init": {
                    "weights_path": "/atmos_ckpt.tar",
                    "parameters": [{"frozen": {"include": ["*"]}}],
                },
            },
        },
        "stepper": {
            "sst_name": "sst",
            "ocean_fraction_prediction": {
                "land_fraction_name": "land_fraction",
                "sea_ice_fraction_name": "ocean_sea_ice_fraction",
                "sea_ice_fraction_name_in_atmosphere": "sea_ice_fraction",
            },
            "atmosphere_output_rename": {
                "eastward_surface_wind_stress": "eastward_surface_stress",
                "northward_surface_wind_stress": "northward_surface_stress",
            },
            "atmosphere_output_scale": {
                "eastward_surface_wind_stress": -1.0,
                "northward_surface_wind_stress": -1.0,
            },
            "ocean": {"timedelta": "5D", "stepper": copy.deepcopy(ocean_st)},
            "atmosphere": {"timedelta": "1D", "stepper": copy.deepcopy(atmos_st)},
        },
    }
    hdr = f"# {out_name}: coupled fine-tune of the UFS 4deg ocean '{ocean_arm}' (ocean checkpoint at /ocean_ckpt.tar) with the frozen ERA5 4deg\n# daily ACE2S atmosphere (/atmos_ckpt.tar). Generated by make_coupled_ft_config.py; see its docstring for the protocol.\n"
    (HERE / f"{out_name}.yaml").write_text(
        hdr + yaml.safe_dump(cfg, sort_keys=False, width=120)
    )
    print("wrote", HERE / f"{out_name}.yaml")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

#!/usr/bin/env python
# ruff: noqa: E501
"""fme.ace.train config: multi-step fine-tune of the CM4-line ACE2.5 atmosphere on ERA5 (Troy 2026-11-13: "launch Spencer's CM4
ACE2.5-like ckpt and FT on ERA5, start with just multistep fine-tuning, use the older 0-20 step schedule for ACE2S").

Why: under SSP2-4.5 the 6-hourly ERA5 ACE2S checkpoint we couple with turns over past ~460 ppm CO2 (its stratosphere flips to
warming and reflected SW grows +23 W/m2) while the CM4-line ACE2.5 (trained on CM4 1pct / piControl / random-CO2 runs, so it has
seen 280-1200 ppm) extrapolates like CMIP6. Fine-tuning ACE2.5 on ERA5 aims at an ERA5-mean-state atmosphere that keeps that
CO2 response.

Recipe = the summer-2026 ERA5 ACE2S multi-step fine-tune (Beaker 01KSN76D3GQ7MVP058Y2Z2TGKE, wandb ai2cm/ace u9ek57d9):
sampled forward-step schedule {1: 0.6, 2: 0.2, 4: 0.1, 12: 0.05, 20: 0.05}, loss on the last step only, EnsembleLoss
(CRPS 0.9 / energy 0.1, 2 members, no per-variable weights), AdamW lr 1e-4 wd 0.01, batch 8, EMA 0.999, train on ERA5
1940-1995 + 2011-2019 + 2021-2025 (1996-2010 and 2020 held out), validation 1996-1997, 5-yr inline inference from 8 ICs in 1996.
Stepper = ACE2.5's own stepper block (inputs/ace2p5-cm4-ft-rs1-train-config.yaml, Beaker 01M6DWYW0RWBN0X33ABS98T0AH) with its
CM4 normalization (mount /atmos_stats = 01KXNT0RA6VX2YTZ8WJ936Q5RS:coupled_atmosphere) and WITHOUT derived_forcings, so DSWRFtoa
is read from the ERA5 store (the CM4 orbital insolation lookup fails on a standard calendar). Weights initialised from the
ACE2.5 best_inference checkpoint (the one run under SSP2-4.5). The ERA5 store lacks ACE2.5's carbon_dioxide (ppmv), lake_fraction
and CM4-convention wind stresses: they come from the extra store built by build_ace2p5_era5_extra.py, merged in (mount /extra).

    python make_ace2p5_era5_ft_configs.py   -> ssp/ace2p5-era5-multistep-ft.yaml
"""

import copy
import pathlib
from typing import Any

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ERA5 = "2026-03-19-era5-1deg-8layer-1940-2025.zarr"
EXTRA = "ace2p5_era5_extra.zarr"


def ds(subsets: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """The ERA5 store merged with the ACE2.5 extra store; several time subsets = merge of two concats (the only nesting fme
    allows: merge: [concat | xarray, ...]).
    """
    era5 = {"data_path": "/climate-default", "file_pattern": ERA5, "engine": "zarr"}
    extra = {"data_path": "/extra", "file_pattern": EXTRA, "engine": "zarr"}
    if not subsets:
        return {"merge": [era5, extra]}
    if len(subsets) == 1:
        return {
            "merge": [
                {**era5, "subset": dict(subsets[0])},
                {**extra, "subset": dict(subsets[0])},
            ]
        }
    return {
        "merge": [
            {"concat": [{**era5, "subset": dict(sb)} for sb in subsets]},
            {"concat": [{**extra, "subset": dict(sb)} for sb in subsets]},
        ]
    }


src = yaml.safe_load((HERE / "inputs/ace2p5-cm4-ft-rs1-train-config.yaml").read_text())
stepper: dict[str, Any] = copy.deepcopy(src["stepper"])
stepper.pop(
    "derived_forcings"
)  # read DSWRFtoa from ERA5 instead of the CM4 insolation lookup
sc = stepper["step"]["config"]
assert (
    "DSWRFtoa" in sc["in_names"]
    and "carbon_dioxide" in sc["in_names"]
    and "lake_fraction" in sc["in_names"]
)
assert (
    sc["normalization"]["network"]["global_means_path"] == "/atmos_stats/centering.nc"
)

cfg: dict[str, Any] = {
    "experiment_dir": "/results",
    "save_checkpoint": True,
    "validate_using_ema": True,
    "evaluate_before_training": True,  # epoch-0 scores of the un-fine-tuned ACE2.5 on ERA5
    "max_epochs": 20,
    "ema": {"decay": 0.999},
    "seed": 0,
    "logging": {
        "log_to_screen": True,
        "log_to_wandb": True,
        "log_to_file": True,
        "project": "ace-samudra-coupled-cm4",
        "entity": "ai2cm",
    },
    "train_loader": {
        "batch_size": 8,
        "num_data_workers": 8,
        "prefetch_factor": 2,
        "dataset": ds(
            [
                {"stop_time": "1995-12-31"},
                {"start_time": "2011-01-01", "stop_time": "2019-12-31"},
                {"start_time": "2021-01-01"},
            ]
        ),
    },
    "validation": [
        {
            "name": "val",
            "evaluate_all_steps": False,
            "weight": 1.0,
            "loader": {
                "batch_size": 16,  # 64 OOMs on 4 H100s in the pre-training validation pass (2 members, up to 20 steps)
                "num_data_workers": 8,
                "prefetch_factor": 2,
                "dataset": ds(
                    [{"start_time": "1996-01-01", "stop_time": "1997-12-31"}]
                ),
            },
        }
    ],
    "inference": [
        {
            "name": "inference",
            "weight": 1.0,
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 40,
            "loader": {
                "start_indices": {
                    "times": [
                        "1996-01-01T00:00:00",
                        "1996-02-15T00:00:00",
                        "1996-04-01T00:00:00",
                        "1996-05-15T00:00:00",
                        "1996-07-01T00:00:00",
                        "1996-08-15T00:00:00",
                        "1996-10-01T00:00:00",
                        "1996-11-15T00:00:00",
                    ]
                },
                "dataset": ds(),
                "num_data_workers": 8,
            },
            "aggregator": {"log_histograms": False},
        }
    ],
    "optimization": {
        "use_gradient_accumulation": True,
        "enable_automatic_mixed_precision": False,
        "lr": 0.0001,
        "optimizer_type": "AdamW",
        "kwargs": {"fused": True, "weight_decay": 0.01},
    },
    "stepper_training": {
        "n_ensemble": 2,
        "n_forward_steps": {
            "outcomes": [
                {"steps": 1, "probability": 0.6},
                {"steps": 2, "probability": 0.2},
                {"steps": 4, "probability": 0.1},
                {"steps": 12, "probability": 0.05},
                {"steps": 20, "probability": 0.05},
            ]
        },
        "optimize_last_step_only": True,
        "parameter_init": {"weights_path": "/ckpt.tar"},
        "loss": {
            "type": "EnsembleLoss",
            "kwargs": {"crps_weight": 0.9, "energy_score_weight": 0.1},
        },
    },
    "stepper": stepper,
}
h = "# Multi-step fine-tune of the CM4-line ACE2.5 atmosphere on ERA5 1deg 6-hourly, summer-2026 ACE2S recipe (sampled 1-20 step schedule).\n# Generated by make_ace2p5_era5_ft_configs.py; launch with launch-1deg-era5-ace2p5-ft.sh (mounts /ckpt.tar, /atmos_stats, /extra).\n"
(HERE / "ssp/ace2p5-era5-multistep-ft.yaml").write_text(
    h + yaml.safe_dump(cfg, sort_keys=False, width=120)
)
print("wrote ssp/ace2p5-era5-multistep-ft.yaml")

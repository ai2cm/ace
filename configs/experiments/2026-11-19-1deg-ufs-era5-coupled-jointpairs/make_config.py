#!/usr/bin/env python
# ruff: noqa: E501
"""1deg UFS + ERA5 coupled fine-tune with the one-stage joint recipe of the CM4 run
1deg-hybridsstdiag-randco2ft-x4-pclr-replay-jointpairs (configs/experiments/cm4_1pct_46to125_piC_156to235/coupled_randco2ft_x4).

Data, stepper, validation and inline inference come from base-two-stage-alternating.yaml (the alternating stage of the
two-stage UFS + ERA5 recipe with the ACE2.5-CM4-ERA5-6HR atmosphere and the corrector-kept UFS ocean, -1.7 W/m2). The
training recipe is replaced by X4's:
  - one stage from the uncoupled checkpoints (UFS ocean-only fine-tune, ERA5 atmosphere fine-tune), no ocean-only stage;
  - per-component optimizers: ocean warmup -> constant -> cosine, atmosphere cosine 0 -> peak at mid-run -> 0;
  - 4 coupled steps, ocean loss on every step, 2-member ensemble loss;
  - each coupled batch optimizes the ocean or one joint pair [[1,20],[2,40],[3,60],[4,80]] (weights 1:1), so the
    atmosphere gets coupled gradient only through the joint pairs;
  - half of each epoch's batches are uncoupled atmosphere batches (weight 2).
What differs from X4: the uncoupled atmosphere batches are ERA5 (the atmosphere's own fine-tuning data, minus the
2002-2011 hindcast years and the 2022-2023 coupled validation years), not the CM4 random-CO2 runs; 16 epochs of 330
batches instead of 4 of 2920, with the schedule milestones at the same fractions of the run; no inline atmosphere-only
inference.

    python make_config.py
"""

import copy
import pathlib
from typing import Any

import yaml

HERE = pathlib.Path(__file__).resolve().parent
EPOCHS = 16
BATCHES_PER_EPOCH = 330  # 165 coupled + 165 uncoupled atmosphere (share 0.5)
TOTAL = EPOCHS * BATCHES_PER_EPOCH
LR = 7.0e-05
SCHEDULE = {
    "outcomes": [
        {"steps": 1, "probability": 0.6},
        {"steps": 2, "probability": 0.2},
        {"steps": 4, "probability": 0.1},
        {"steps": 12, "probability": 0.05},
        {"steps": 20, "probability": 0.05},
    ]
}
ENSEMBLE_LOSS: dict[str, Any] = {
    "type": "EnsembleLoss",
    "kwargs": {"crps_weight": 0.9, "energy_score_weight": 0.1},
}
ERA5 = {
    "data_path": "/climate-default",
    "file_pattern": "2026-03-19-era5-1deg-8layer-1940-2025.zarr",
    "engine": "zarr",
    "overwrite": {"multiply_scalar": {"PRMSL": 0.01}},
}
EXTRA = {
    "data_path": "/climate-default",
    "file_pattern": "2026-11-17-era5-1deg-ace2p5-extra-1940-2025.zarr",
    "engine": "zarr",
}
# the ERA5 fine-tune's training years, without 2011 (hindcast years) and 2022 on (coupled validation)
PERIODS: list[dict[str, str]] = [
    {"stop_time": "1995-12-31"},
    {"start_time": "2012-01-01", "stop_time": "2019-12-31"},
    {"start_time": "2021-01-01", "stop_time": "2021-12-31"},
]


def optimizer(lr: float, scheduler: dict[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_type": "AdamW",
        "kwargs": {"fused": True, "weight_decay": 0.01},
        "enable_automatic_mixed_precision": False,
        "use_gradient_accumulation": True,
        "max_grad_norm": 1.0,
        "lr": lr,
        "scheduler": scheduler,
    }


base: dict[str, Any] = yaml.safe_load(
    (HERE / "base-two-stage-alternating.yaml").read_text()
)
cfg = copy.deepcopy(base)
cfg["max_epochs"] = EPOCHS
cfg["optimization"] = {
    "atmosphere": optimizer(
        0.0,
        {
            "type": "CosineAnnealingLR",
            "step_each_iteration": True,
            "kwargs": {"T_max": TOTAL // 2, "eta_min": LR},
        },
    ),
    "ocean": optimizer(
        LR,
        {
            "schedulers": [
                {
                    "type": "LinearLR",
                    "step_each_iteration": True,
                    "kwargs": {
                        "start_factor": 0.01,
                        "end_factor": 1.0,
                        "total_iters": TOTAL // 4,
                    },
                },
                {
                    "type": "ConstantLR",
                    "step_each_iteration": True,
                    "kwargs": {"factor": 1.0, "total_iters": 0},
                },
                {
                    "type": "CosineAnnealingLR",
                    "step_each_iteration": True,
                    "kwargs": {"T_max": TOTAL // 2, "eta_min": 0.0},
                },
            ],
            "milestones": [TOTAL // 4, TOTAL // 2],
        },
    ),
}
old = base["stepper_training"]
cfg["stepper_training"] = {
    "n_coupled_steps": 4,
    "n_ensemble": 2,
    "ocean": {
        "parameter_init": {"weights_path": "/ocean_ckpt.tar"},
        "n_steps": 4,
        "optimize_last_step_only": False,
        "loss_weight": 1.0,
        "loss": ENSEMBLE_LOSS,
    },
    "atmosphere": old["atmosphere"],
    "joint_pairs": [[1, 20], [2, 40], [3, 60], [4, 80]],
    "objective_per_batch": {"ocean": 1.0, "joint": 1.0, "uncoupled_atmosphere": 2.0},
    "uncoupled_atmosphere": {
        "loss": {**ENSEMBLE_LOSS, "weights": {"h500": 5}},
        "n_steps": SCHEDULE,
        "optimize_last_step_only": True,
    },
}
cfg["train_loader_uncoupled_atmosphere"] = {
    "batch_size": 8,
    "num_data_workers": 4,
    "prefetch_factor": 2,
    "time_buffer": 10,
    "dataset": {
        "merge": [
            {"concat": [{**ERA5, "subset": p} for p in PERIODS]},
            {"concat": [{**EXTRA, "subset": p} for p in PERIODS]},
        ]
    },
}
cfg["checkpoint_every_n_batches"] = 100
header = (
    "# 1deg UFS + ERA5 coupled fine-tune, one-stage joint recipe (CM4 X4 template): per-component schedules, ocean /\n"
    "# joint-pair objectives 1:1, half of each epoch uncoupled ERA5 atmosphere batches. Generated by make_config.py.\n"
    "# Launch: ./launch.sh\n"
)
(HERE / "ufs1deg-era5-ace2p5cm4-jointpairs.yaml").write_text(
    header + yaml.safe_dump(cfg, sort_keys=False, width=120)
)
resume = copy.deepcopy(cfg)
resume["resume_results"] = {"existing_dir": "/existing-results", "resume_wandb": True}
(HERE / "ufs1deg-era5-ace2p5cm4-jointpairs-resume.yaml").write_text(
    header + yaml.safe_dump(resume, sort_keys=False, width=120)
)
print("wrote configs")

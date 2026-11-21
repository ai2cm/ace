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


def write(name: str, header: str, config: dict[str, Any]) -> None:
    (HERE / f"{name}.yaml").write_text(
        header + yaml.safe_dump(config, sort_keys=False, width=120)
    )
    resume = copy.deepcopy(config)
    resume["resume_results"] = {
        "existing_dir": "/existing-results",
        "resume_wandb": True,
    }
    (HERE / f"{name}-resume.yaml").write_text(
        header + yaml.safe_dump(resume, sort_keys=False, width=120)
    )


header = (
    "# 1deg UFS + ERA5 coupled fine-tune, one-stage joint recipe (CM4 X4 template): per-component schedules, ocean /\n"
    "# joint-pair objectives 1:1, half of each epoch uncoupled ERA5 atmosphere batches. Generated by make_config.py.\n"
    "# Launch: ./launch.sh\n"
)
write("ufs1deg-era5-ace2p5cm4-jointpairs", header, cfg)

# Random-CO2 variant (Troy 2026-11-20): the uncoupled atmosphere batches are the CM4 random-CO2 runs, as in X4, so the
# atmosphere keeps seeing CO2 varied independently of the climate state. Fixed-SST runs show that the ERA5 fine-tune
# lost the surface-flux response to CO2 that the CM4-only atmosphere has (net heat into the surface 400 -> 550 ppm:
# +2.0 W/m2 CM4-only, 0.0 after the ERA5 fine-tune, -0.8 after the two-stage coupled fine-tune). The stores are the
# ones in X4's train_loader_uncoupled_atmosphere: the SST-blended coupled atmosphere store first, then the full
# atmosphere-land store (carbon_dioxide in ppmv, PRMSL in hPa and DSWRFtoa are in the store, CM4 normalization).
RANDCO2_START = "0153-01-01T06:00:00"
RANDCO2_RUNS = [f"random-CO2-{x}xCO2-ic_000{i}" for x in (1, 2, 4) for i in (1, 2)]


def randco2(directory: str, suffix: str) -> dict[str, Any]:
    return {
        "concat": [
            {
                "data_path": f"/climate-default/{directory}",
                "file_pattern": f"{run}{suffix}.zarr",
                "engine": "zarr",
                "subset": {"start_time": RANDCO2_START},
            }
            for run in RANDCO2_RUNS
        ]
    }


rcfg = copy.deepcopy(cfg)
rcfg["train_loader_uncoupled_atmosphere"]["dataset"] = {
    "merge": [
        randco2("2026-10-11-CM4-like-AM4-random-CO2-coupled", "-atmosphere"),
        randco2("2026-10-11-CM4-like-AM4-random-CO2", ""),
    ]
}
rheader = (
    "# 1deg UFS + ERA5 coupled fine-tune, one-stage joint recipe (CM4 X4 template) with the uncoupled atmosphere batches\n"
    "# drawn from the CM4 random-CO2 runs instead of ERA5. Generated by make_config.py.\n"
    "# Launch: VARIANT=randco2 ./launch.sh\n"
)
write("ufs1deg-era5-ace2p5cm4-jointpairs-randco2", rheader, rcfg)
# X6 variants (Troy 2026-11-21): X4 with the CM4 X6 run's learning-rate schedules and nothing else changed. Both
# components follow repeated rise-and-fall cycles under a decaying envelope (ocean 3 cycles then a tail, atmosphere
# 2), as chains of linear segments. The atmosphere's base rate is 7e-7 (X4 peaks at 7e-5), so it moves far less.
# The segment boundaries are X6's, rescaled from its 11680 batches to this run's length.
X6 = yaml.safe_load((HERE / "x6-schedules-cm4.yaml").read_text())


def x6_optimization(component: str) -> dict[str, Any]:
    source = X6[component]["scheduler"]
    scale = TOTAL / X6["total_batches"]
    milestones: list[int] = []
    for m in source["milestones"]:
        milestones.append(
            max(round(m * scale), (milestones[-1] if milestones else 0) + 1)
        )
    edges = [0] + milestones
    schedulers = copy.deepcopy(source["schedulers"])
    for i, sched in enumerate(schedulers):
        if i < len(milestones):
            sched["kwargs"]["total_iters"] = edges[i + 1] - edges[i]
        else:
            sched["kwargs"]["total_iters"] = max(TOTAL - 1 - edges[-1], 1)
    out = copy.deepcopy(cfg["optimization"][component])
    out["lr"] = X6[component]["lr"]
    out["scheduler"] = {"schedulers": schedulers, "milestones": milestones}
    return out


for base, base_name, launch in (
    (cfg, "ufs1deg-era5-ace2p5cm4-jointpairs-x6", "VARIANT=x6"),
    (rcfg, "ufs1deg-era5-ace2p5cm4-jointpairs-x6-randco2", "VARIANT=x6-randco2"),
):
    xcfg = copy.deepcopy(base)
    xcfg["optimization"] = {k: x6_optimization(k) for k in ("atmosphere", "ocean")}
    xheader = (
        "# 1deg UFS + ERA5 coupled fine-tune, one-stage joint recipe with the CM4 X6 learning-rate schedules (X4 otherwise).\n"
        "# Generated by make_config.py.\n"
        f"# Launch: {launch} ./launch.sh\n"
    )
    write(base_name, xheader, xcfg)
# CM4 X6 initialization (Troy 2026-11-21): the X6 protocol on UFS + ERA5 starting from both components of the CM4 X6
# coupled fine-tune (its EMA epoch 3, the best inline epoch) instead of the ERA5-fine-tuned atmosphere and the UFS
# ocean-only fine-tune. Its atmosphere is the CM4 network without grouped filters or the reduced spectral ratio.
icfg = copy.deepcopy(
    yaml.safe_load(
        (HERE / "ufs1deg-era5-ace2p5cm4-jointpairs-x6-randco2.yaml").read_text()
    )
)
builder = icfg["stepper"]["atmosphere"]["stepper"]["step"]["config"]["builder"][
    "config"
]
for key in ("filter_num_groups", "spectral_ratio"):
    del builder[key]
iheader = (
    "# 1deg UFS + ERA5 coupled fine-tune, X6 protocol with random-CO2 uncoupled batches, both components initialized\n"
    "# from the CM4 X6 coupled fine-tune. Generated by make_config.py.\n"
    "# Launch: VARIANT=x6-randco2-cm4x6init ATMOS_DS=<lifted atmosphere> ATMOS_CKPT=atmos_ckpt.tar "
    "OCEAN_DS=<lifted ocean> OCEAN_CKPT=ocean_ckpt.tar ./launch.sh\n"
)
write("ufs1deg-era5-ace2p5cm4-jointpairs-x6-randco2-cm4x6init", iheader, icfg)
print("wrote configs")

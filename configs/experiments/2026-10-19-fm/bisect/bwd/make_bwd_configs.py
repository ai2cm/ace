"""Write the backward-bisection configs (v2.2 -> v2.1, one change reverted per arm).

Goal: find which v2.1 -> v2.2 change makes the full-resolution skip path of
nc-swin-v2.2 a ~unit-gain integrator for the grid-scale (-1)^(i+j) mode.

Every arm starts from the 2026-10-19 v2.2 s0-maskoff run config and shares
these short-run settings (B0 is the reference with nothing reverted):

- EMA checkpoints kept every epoch (ema_ckpt_NNNN.tar) so arms compare at
  near-matched optimizer steps. checkpoint_every_n_batches only overwrites
  ckpt.tar, so it cannot keep step-indexed checkpoints. Epochs keep their
  native length (no sample_with_replacement: its RandomSampler draws from the
  global torch RNG, which is seeded identically on every rank, so ranks would
  likely train on duplicate samples). An epoch is 25,317 steps for every arm
  except B3, whose v2.1 data gives 7,434-step epochs.
- ~100k steps: max_epochs 4 (101,268 steps), or 14 for B3 (104,076 steps).
  The reference v2.2 s3-maskoff run reached Nyquist gain 0.94 by epoch 3
  (75,951 steps) and v2.1 sits at 0.58-0.67.
- LR warmup in optimizer steps (step_each_iteration): linear 0.01 -> 1 over
  50,634 steps (= v2.2's 2 epochs x 25,317 steps), then constant. v2.1's 8
  epochs x 7,434 = 59,472 steps, so the warmup is nearly the same in steps
  for both versions, and is held fixed across arms (B3's short epochs would
  otherwise shorten it).
- No inline inference; validation as in the base config.

Arms:
  B0  reference (v2.2 maskoff, seed 0)
  B1  num_levels 1
  B2  embed_dim 128 (num_heads [4, 8, 8, 4] kept: head dim 32 as in v2.1)
  B3  v2.1 training data (ERA5 1940-1993 + 1995-2013, C96 AMIP 1979-2008,
      ramped, SOM) with v2.1's labels (amip/ramped/som/era5) and v2.1's
      normalization stats. global_mean_co2 stays an input: v2.1 had it too.
  B4  include_channel_mask_inputs false

Usage: python make_bwd_configs.py
"""

import copy
import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
BASE = (
    HERE.parent.parent
    / "run_configs"
    / "ace-train-config-4deg-nc-swin-v2.2-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-s0-maskoff.yaml"  # noqa: E501
)
PREFIX = "ace-train-config-4deg-bisect-bwd-"

MAX_EPOCHS = 4
MAX_EPOCHS_B3 = 14
WARMUP_STEPS = 50634

V21_NORM = "/climate-default/alexeyy/norm_ablation_0/fm"
C96_AMIP = (
    "/climate-default/2026-01-28-vertically-resolved-c96-4deg-daily-"
    "shield-amip-ensemble-dataset"
)
C96_RAMPED = (
    "/climate-default/2026-06-08-vertically-resolved-c96-shield-ramped-"
    "climSST-random-CO2-ensemble-fme-dataset-4deg-daily"
)
C96_SOM = (
    "/climate-default/2026-06-08-vertically-resolved-4deg-daily-c96-shield-"
    "som-ensemble-fme-dataset"
)
ERA5 = "2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr"


def _ds(path, pattern, start, stop, label):
    return {
        "data_path": path,
        "file_pattern": pattern,
        "engine": "zarr",
        "subset": {"start_time": start, "stop_time": stop},
        "labels": [label],
    }


# Training data of wandb run ai2cm/FM x9azik2e (ace2-fm-nc-swin-v2.1-fm-a2-cond).
V21_TRAIN = [
    _ds(C96_AMIP, "ic_0001.zarr", "1979-01-01", "2008-12-31", "amip"),
    *[
        _ds(
            C96_RAMPED,
            f"ramped-sst-{co2}xCO2-random-perturbation{ic}.zarr",
            "2020-01-01",
            "2025-01-01",
            "ramped",
        )
        for co2 in (1, 2, 4)
        for ic in ("", "-ic_0002")
    ],
    *[
        _ds(C96_SOM, f"{co2}xCO2-ic_0001.zarr", "2031-01-02", "2041-01-01", "som")
        for co2 in (1, 2, 4)
    ],
    _ds("/climate-default", ERA5, "1940-01-03", "1993-12-31", "era5"),
    _ds("/climate-default", ERA5, "1995-01-01", "2013-12-31", "era5"),
]


def common(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["max_epochs"] = MAX_EPOCHS
    cfg["inference"] = []
    cfg["ema_checkpoint_save_epochs"] = {"start": 1, "step": 1}
    cfg["optimization"]["scheduler"] = {
        "schedulers": [
            {
                "type": "LinearLR",
                "kwargs": {
                    "start_factor": 0.01,
                    "end_factor": 1.0,
                    "total_iters": WARMUP_STEPS,
                },
                "step_each_iteration": True,
            },
            {
                "type": "ConstantLR",
                "kwargs": {"factor": 1.0, "total_iters": 10_000_000},
                "step_each_iteration": True,
            },
        ],
        "milestones": [WARMUP_STEPS],
    }
    return cfg


def step_cfg(cfg: dict) -> dict:
    return cfg["stepper"]["step"]["config"]


def b1(cfg):
    step_cfg(cfg)["builder"]["config"]["num_levels"] = 1


def b2(cfg):
    step_cfg(cfg)["builder"]["config"]["embed_dim"] = 128


def b3(cfg):
    cfg["max_epochs"] = MAX_EPOCHS_B3
    cfg["train_loader"]["dataset"]["concat"] = copy.deepcopy(V21_TRAIN)
    for entry in cfg["validation"]["loader"]["dataset"]["concat"]:
        if entry["labels"] == ["c96"]:
            entry["labels"] = ["amip"]
    norm = step_cfg(cfg)["normalization"]
    norm["network"] = {
        "global_means_path": f"{V21_NORM}/centering.nc",
        "global_stds_path": f"{V21_NORM}/scaling-full-field.nc",
    }
    norm["residual"] = {
        "global_means_path": f"{V21_NORM}/centering.nc",
        "global_stds_path": f"{V21_NORM}/scaling-residual.nc",
    }

    def group(name):
        return {
            "global_means_path": f"{V21_NORM}/groups/{name}/centering.nc",
            "global_stds_path": f"{V21_NORM}/groups/{name}/scaling-full-field.nc",
        }

    norm["grouped"]["groups"] = {
        "amip": group("c96"),
        "ramped": group("c96"),
        "som": group("c96"),
        "era5": group("era5"),
    }
    norm["grouped"]["default_label"] = "era5"


def b4(cfg):
    step_cfg(cfg)["include_channel_mask_inputs"] = False


ARMS = {
    "B0-ref": None,
    "B1-levels1": b1,
    "B2-embed128": b2,
    "B3-v21data": b3,
    "B4-nomaskin": b4,
}


def main() -> None:
    base = yaml.safe_load(BASE.read_text())
    header = (
        f"# Generated by make_bwd_configs.py from {BASE.name}.\n"
        "# Backward bisection arm; see make_bwd_configs.py for the shared\n"
        "# short-run settings.\n"
    )
    for arm, fn in ARMS.items():
        cfg = common(base)
        if fn is not None:
            fn(cfg)
        out = HERE / f"{PREFIX}{arm}.yaml"
        out.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
        print("wrote", out.name)


if __name__ == "__main__":
    main()

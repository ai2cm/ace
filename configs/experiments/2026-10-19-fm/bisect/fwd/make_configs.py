"""Generate the forward-bisection configs (nc-swin-v2.1 -> v2.2 checker gain).

Forward arms start from the clean nc-swin-v2.1 FM model (wandb ai2cm/FM run
x9azik2e, `ace2-fm-nc-swin-v2.1-fm-a2-cond`) and add one v2.2 change per arm:

- F0: the v2.1 config (data, labels, normalization, architecture) on this
  branch's code. Tests the code changes since v2.1 (fc41006).
- F1: F0 with v2.2's training data, labels (c96/era5/cm4/xshield),
  normalization stats (2026-10-22-fm-all-datasets-v2, grouped) and
  validation data. global_mean_co2 was already an input in v2.1, so there
  is no separate co2 change. Mask-input channels are not part of the data
  setup and stay off (include_channel_mask_inputs is false in every arm).
- F2: F1 with num_levels 2.
- F3: F2 with embed_dim 256 (= v2.2 base minus the LR schedule and the
  mask-input channels).

Shared by all arms, to compare at matched optimizer steps:

- label_sampling draws each label in proportion to its number of samples
  (the same mix as plain shuffling) but caps an epoch at SAMPLES_PER_EPOCH
  time-buffer windows: 8888 windows / batch 8 * 9 batches per window =
  9999 steps per epoch on any GPU count. (sample_with_replacement would
  draw the same indices on every rank, as all ranks share one seed.)
- An EMA checkpoint is kept every epoch (~10k steps), up to 10 epochs.
- LR: linear warmup from 0.01x over 6 epochs (~60k steps, v2.1's 8 epochs
  of 7.4k steps; v2.2's was 2 epochs of 25k = 50k steps), then constant.
  The same schedule in steps for every arm, so it is not a factor here.
- No inline inference; validation as in the arm's source config.

Usage: python make_configs.py  (writes the yaml files next to this script)
"""

import copy
import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent.parent
V22_BASE = (
    EXP
    / "base_configs"
    / "ace-train-config-4deg-nc-swin-v2.2-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield.yaml"
)
SAMPLES_PER_EPOCH = 8888
MAX_EPOCHS = 10
WARMUP_EPOCHS = 6

C96_AMIP = (
    "/climate-default/2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-"
    "ensemble-dataset"
)
C96_RAMPED = (
    "/climate-default/2026-06-08-vertically-resolved-c96-shield-ramped-climSST-"
    "random-CO2-ensemble-fme-dataset-4deg-daily"
)
C96_SOM = (
    "/climate-default/2026-06-08-vertically-resolved-4deg-daily-c96-shield-som-"
    "ensemble-fme-dataset"
)
ERA5 = "2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr"
V21_NORM = "/climate-default/alexeyy/norm_ablation_0/fm"

# v2.1 in_names order (same set as v2.2, which moves global_mean_co2 forward)
V21_IN_NAMES = (
    ["land_fraction", "ocean_fraction", "sea_ice_fraction", "DSWRFtoa", "HGTsfc"]
    + ["PRESsfc", "surface_temperature", "TMP2m", "Q2m", "UGRD10m", "VGRD10m"]
    + [f"air_temperature_{i}" for i in range(8)]
    + [f"specific_total_water_{i}" for i in range(8)]
    + [f"eastward_wind_{i}" for i in range(8)]
    + [f"northward_wind_{i}" for i in range(8)]
    + ["global_mean_co2"]
)

# Samples (days) per label, for label_sampling weights proportional to size.
V21_LABEL_DAYS = {"amip": 10958, "ramped": 10968, "som": 10959, "era5": 26662}
V22_LABEL_DAYS = {"c96": 32520, "era5": 24836, "cm4": 141353, "xshield": 4020}


def _entry(path, pattern, start, stop, label):
    return {
        "data_path": path,
        "file_pattern": pattern,
        "engine": "zarr",
        "subset": {"start_time": start, "stop_time": stop},
        "labels": [label],
    }


def v21_train_concat():
    out = [_entry(C96_AMIP, "ic_0001.zarr", "1979-01-01", "2008-12-31", "amip")]
    for co2 in ("1xCO2", "2xCO2", "4xCO2"):
        for suffix in ("", "-ic_0002"):
            out.append(
                _entry(
                    C96_RAMPED,
                    f"ramped-sst-{co2}-random-perturbation{suffix}.zarr",
                    "2020-01-01",
                    "2025-01-01",
                    "ramped",
                )
            )
    for co2 in ("1xCO2", "2xCO2", "4xCO2"):
        out.append(
            _entry(C96_SOM, f"{co2}-ic_0001.zarr", "2031-01-02", "2041-01-01", "som")
        )
    out.append(_entry("/climate-default", ERA5, "1940-01-03", "1993-12-31", "era5"))
    out.append(_entry("/climate-default", ERA5, "1995-01-01", "2013-12-31", "era5"))
    return out


def v21_validation_concat():
    return [
        _entry("/climate-default", ERA5, "1994-01-01", "1994-12-31", "era5"),
        _entry("/climate-default", ERA5, "2014-01-01", "2014-12-31", "era5"),
    ]


def v21_normalization(old):
    def stats(sub, std_file):
        return {
            "global_means_path": f"{V21_NORM}/{sub}centering.nc",
            "global_stds_path": f"{V21_NORM}/{sub}{std_file}",
        }

    c96 = stats("groups/c96/", "scaling-full-field.nc")
    return {
        "network": stats("", "scaling-full-field.nc"),
        "residual": stats("", "scaling-residual.nc"),
        "grouped": {
            # v2.1 mapped labels amip/ramped/som to one c96 group; the current
            # schema keys groups by label, so each gets the c96 stats.
            "groups": {
                "amip": dict(c96),
                "ramped": dict(c96),
                "som": dict(c96),
                "era5": stats("groups/era5/", "scaling-full-field.nc"),
            },
            "default_label": "era5",
            "pinned_variables": old["grouped"]["pinned_variables"],
        },
    }


def common(cfg, arm):
    cfg["seed"] = 0
    cfg["max_epochs"] = MAX_EPOCHS
    cfg["inference"] = []
    cfg["checkpoint_every_n_batches"] = 2500
    cfg["ema_checkpoint_save_epochs"] = {"start": 1, "step": 1}
    cfg["optimization"]["scheduler"] = {
        "schedulers": [
            {
                "type": "LinearLR",
                "kwargs": {
                    "start_factor": 0.01,
                    "end_factor": 1.0,
                    "total_iters": WARMUP_EPOCHS,
                },
                "step_each_iteration": False,
            },
            {
                "type": "ConstantLR",
                "kwargs": {"factor": 1.0, "total_iters": MAX_EPOCHS},
                "step_each_iteration": False,
            },
        ],
        "milestones": [WARMUP_EPOCHS],
    }
    sc = cfg["stepper"]["step"]["config"]
    sc["in_names"] = list(V21_IN_NAMES)
    sc.pop("include_channel_mask_inputs", None)
    return cfg


def label_sampling(days):
    return {"weights": dict(days), "samples_per_epoch": SAMPLES_PER_EPOCH}


def build():
    base = yaml.safe_load(V22_BASE.read_text())
    arms = {}

    # F1: v2.2 data/norm/validation, v2.1 architecture
    f1 = common(copy.deepcopy(base), "F1")
    f1["train_loader"]["label_sampling"] = label_sampling(V22_LABEL_DAYS)
    b = f1["stepper"]["step"]["config"]["builder"]["config"]
    b["embed_dim"] = 128
    b["num_levels"] = 1

    # F0: F1 with v2.1 data/labels/norm/validation
    f0 = copy.deepcopy(f1)
    f0["train_loader"]["dataset"]["concat"] = v21_train_concat()
    f0["train_loader"]["label_sampling"] = label_sampling(V21_LABEL_DAYS)
    f0["validation"]["loader"]["dataset"]["concat"] = v21_validation_concat()
    sc0 = f0["stepper"]["step"]["config"]
    sc0["normalization"] = v21_normalization(sc0["normalization"])

    f2 = copy.deepcopy(f1)
    f2["stepper"]["step"]["config"]["builder"]["config"]["num_levels"] = 2

    f3 = copy.deepcopy(f2)
    f3["stepper"]["step"]["config"]["builder"]["config"]["embed_dim"] = 256

    arms["F0-v21"] = f0
    arms["F1-v22data"] = f1
    arms["F2-levels2"] = f2
    arms["F3-embed256"] = f3
    return arms


HEADER = (
    "# Generated by make_configs.py (forward bisection, nc-swin-v2.1 -> v2.2).\n"
    "# Arm {arm}; see make_configs.py for the per-arm changes.\n"
)


def main():
    for arm, cfg in build().items():
        path = HERE / f"ace-train-config-4deg-nc-swin-bisect-fwd-{arm}.yaml"
        path.write_text(
            HEADER.format(arm=arm) + yaml.safe_dump(cfg, sort_keys=False, width=88)
        )
        print(path.name)


if __name__ == "__main__":
    main()

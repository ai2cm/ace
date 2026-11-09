"""Generate the 2026-11-09 FM v2 run configs (wandb group ace2-fm-2026-11-09).

18 runs: 2 architectures x 3 data arms x 3 seeds, all with input masking off.
Each run config is derived from a base config in base_configs/ (the "parent"),
so everything not listed below is copied from the parent verbatim: loss,
corrector, normalization groups and pinned variables (stats are replaced, see
below), global-mean removal, optimizer and weight decay, AMP, batch size,
checkpoint_every_n_batches, and for swin float32_matmul_precision and
max_consecutive_non_finite_losses.

Architectures (tag directly after `ace-train-config-4deg-`), sized to about the
same parameter count:
- nc-sfno-v2: NoiseConditionedSFNO, embed_dim 1024 (parent 512), 52.6M params.
- nc-swin-v2.2-v2: NoiseConditionedSwinTransformer, embed_dim 224 (parent 256),
  50.9M params.

Data arms (label sampling weights are equal across the arm's labels):
- `fm-era5`: parent nc-*-fm-era5-only, the balanced base config restricted to
  ERA5: its ERA5 training datasets (1940-01-03..1993-12-31, 1995..2008),
  ERA5 validation and ERA5 inline inference only, and ERA5-only normalization
  (no grouped normalizer; network and residual stats from the era5 group of
  the arm's own stats). Label weights {era5: 1}.
- `fm-balanced`: parent nc-*-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-balanced,
  its data and year splits plus the extra c96 members below (era5, c96, cm4,
  xshield at 0.25 each).
- `fm-e3sm-balanced`: parent
  nc-*-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-e3sm-balanced (the ablation data +
  E3SMv3 piControl, with the e3sm normalization group) plus the same extra c96
  members, so its training data is exactly fm-balanced's plus E3SM. era5, c96,
  cm4, xshield, e3sm at 0.2 each.

Extra c96 members (fm-balanced and fm-e3sm-balanced): the c96 SHiELD AMIP
ic_0002 member (same year split as ic_0001) and c96 SOM members ic_0002..ic_0005
at 1x/2x/4xCO2, except 1xCO2-ic_0005, which is the held-out member that the
paper SOM control / abrupt-4xCO2 evaluations initialize from
(generate_inference_configs.SOM_CONTROL_FILE).

Every arm draws SAMPLES_PER_EPOCH = 2758 samples per epoch (label_sampling
samples_per_epoch). With time_buffer 8 a sample is a window yielding 9
batches, and 2758 is the number of windows fme makes from the fm-era5 training
data without label sampling (the concatenated ERA5 splits, 24818 one-step or
24814 three-step samples, strided by 9), so all arms see as many samples per
epoch as fm-era5 would by default.

Changes from the parents, identical for every arm:
- Normalization stats pooled over exactly the arm's training data (the
  datasets listed above, including the extra c96 members), one stats root per
  arm under /climate-default/alexeyy/norm_ablation_0/2026-11-09-fm-v2-{era5,
  balanced,e3sm-balanced} (scripts/data_process/configs/fm-v2-*-stats.yaml);
  the e3sm group is groups/e3sm-filled (fill_e3sm_group_stats.py).
- Input masking off with presence channels: `include_channel_mask_inputs:
  true` and no `input_dropout` (the maskoff arm of create_fm_jobs.py).
- max_epochs 50; EMA decay 0.9999 with validate_using_ema; EMA checkpoints
  every epoch (`ema_checkpoint_save_epochs: {start: 1, step: 1}`, 1..50).
- Step curriculum: 1 forward step for epochs 1-40, 3 for epochs 41-50, with
  the loss on every step (optimize_last_step_only false) and gradients
  detached between steps (use_gradient_accumulation true, from the parent).
  Validation evaluates only the steps the train stepper would for the epoch.
- LR 1e-4 constant for all 50 epochs. Swin keeps the parent's 2-epoch linear
  warmup (0.01 -> 1); sfno has no scheduler.
- `sht_roundtrip_variables: [HGTsfc]` on every training dataset labelled cm4
  (CM4 piControl, CM4 1pctCO2, CM4-like AM4 random-CO2) or xshield. Their
  PRESsfc was spherical-harmonic roundtripped upstream but HGTsfc was not, so
  grid-scale PRESsfc does not follow HGTsfc; roundtripping HGTsfc restores it.
- Validation is a list of named entries, each with `evaluate_all_steps: false`:
  `val`, ERA5 1994 + 2014 (weight 1, the only term of the best_ckpt
  selection), and, where the parent validates on C96 (fm-balanced and
  fm-e3sm-balanced), `val_c96`, C96 AMIP ic_0001 1994 + 2014 (weight 0, logged
  only).
- Inline inference: aimip_checkpoint_era5 every epoch (weight 1, the only term
  of the best_inference_ckpt selection) and, on epochs 10, 20, ..., 50
  (`epochs: {start: 9, step: 10}`), 10year_oosample_varying_co2_era5,
  long_46year_varying_co2_era5 and two 5-day ensemble weather forecasts,
  weather_1950_era5 and weather_2020_era5 (all weight 0). These are the
  parents' weather_{1950_insample,2020_oosample}_ensemble_varying_co2_era5,
  renamed; the parents' constant_co2 weather variants are dropped (CO2 barely
  changes over 5 days).

The configs are written by a YAML round trip, so the parents' comments are
not carried over; the header of each file points back to its parent.

Usage:
    python create_fm_v2_jobs.py
"""

import copy
import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
EXP_DIR = HERE.parent  # experiment root (configs, records); scripts live in scripts/
BASE_CONFIGS_DIR = EXP_DIR / "base_configs"
RUN_CONFIGS_DIR = EXP_DIR / "run_configs"
CONFIG_PREFIX = "ace-train-config-4deg-"

SEEDS = (0, 1, 2)
MAX_EPOCHS = 50
SINGLE_STEP_EPOCHS = 40
MULTI_STEP_N_FORWARD_STEPS = 3
SAMPLES_PER_EPOCH = 2758
LR = 1e-4

# arch tag -> (parent arch tag, parent embed_dim, embed_dim)
ARCHS = {
    "nc-sfno-v2": ("nc-sfno", 512, 1024),
    "nc-swin-v2.2-v2": ("nc-swin-v2.2", 256, 224),
}
# arm name (in the config name) -> parent data stem
ARMS = {
    "fm-era5": "fm-era5-only",
    "fm-balanced": "fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-balanced",
    "fm-e3sm-balanced": "fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-e3sm-balanced",
}
ERA5_ONLY_ARM = "fm-era5"

# Normalization stats: each arm's stats are pooled over exactly that arm's
# training data (scripts/data_process/configs/fm-v2-{arm}-stats.yaml), replacing
# the parent's stats root. The e3sm group is read from groups/e3sm-filled
# (fill_e3sm_group_stats.py), as in the parent.
NORM_ROOT = "/climate-default/alexeyy/norm_ablation_0"
PARENT_STATS_ROOTS = {
    "fm-era5": f"{NORM_ROOT}/2026-10-22-fm-all-datasets-v2",
    "fm-balanced": f"{NORM_ROOT}/2026-10-22-fm-all-datasets-v2",
    "fm-e3sm-balanced": f"{NORM_ROOT}/2026-10-26-fm-all-datasets-v3",
}
STATS_ROOTS = {
    "fm-era5": f"{NORM_ROOT}/2026-11-09-fm-v2-era5",
    "fm-balanced": f"{NORM_ROOT}/2026-11-09-fm-v2-balanced",
    "fm-e3sm-balanced": f"{NORM_ROOT}/2026-11-09-fm-v2-e3sm-balanced",
}

SELECTION_INFERENCE = "aimip_checkpoint_era5"
HEAVY_INFERENCE = (
    "10year_oosample_varying_co2_era5",
    "long_46year_varying_co2_era5",
)
# 5-day ensemble weather forecasts (4 members x 8 ICs), on the same epochs.
# parent entry name -> name in the run config (varying CO2 only).
WEATHER_INFERENCE = {
    "weather_1950_insample_ensemble_varying_co2_era5": "weather_1950_era5",
    "weather_2020_oosample_ensemble_varying_co2_era5": "weather_2020_era5",
}
# Inline inference slices index the epochs 1..max_epochs, so index 9 is epoch 10.
HEAVY_INFERENCE_EPOCHS = {"start": 9, "step": 10}
# EMA checkpoint slices index range(max_epochs + 1), so index 1 is epoch 1.
EMA_CHECKPOINT_SAVE_EPOCHS = {"start": 1, "step": 1}

ROUNDTRIP_LABELS = ("cm4", "xshield")
ROUNDTRIP_VARIABLES = ["HGTsfc"]

C96_AMIP_IC2 = {
    "data_path": (
        "/climate-default/"
        "2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-dataset"
    ),
    "file_pattern": "ic_0002.zarr",
}
C96_AMIP_TRAIN_SPLITS = (
    ("1979-01-01", "1993-12-31"),
    ("1995-01-01", "2008-12-31"),
)
SOM_DATA_PATH = (
    "/climate-default/"
    "2026-06-08-vertically-resolved-4deg-daily-c96-shield-som-ensemble-fme-dataset"
)
SOM_SUBSET = ("2031-01-02", "2041-01-01")
# Held out: the paper SOM evaluations start from this member.
SOM_HELD_OUT = "1xCO2-ic_0005.zarr"
SOM_NEW_MEMBERS = tuple(
    f"{co2}xCO2-ic_{ic:04d}.zarr"
    for co2 in (1, 2, 4)
    for ic in (2, 3, 4, 5)
    if f"{co2}xCO2-ic_{ic:04d}.zarr" != SOM_HELD_OUT
)


def parent_path(arch: str, arm: str) -> pathlib.Path:
    parent_arch = ARCHS[arch][0]
    return BASE_CONFIGS_DIR / f"{CONFIG_PREFIX}{parent_arch}-{ARMS[arm]}.yaml"


def run_config_name(arch: str, arm: str, seed: int) -> str:
    return f"{CONFIG_PREFIX}{arch}-{arm}-s{seed}-maskoff.yaml"


def _zarr_entry(data_path, file_pattern, start, stop, label="c96") -> dict:
    return {
        "data_path": data_path,
        "file_pattern": file_pattern,
        "engine": "zarr",
        "subset": {"start_time": start, "stop_time": stop},
        "labels": [label],
    }


def extra_c96_entries() -> list[dict]:
    entries = [
        _zarr_entry(C96_AMIP_IC2["data_path"], C96_AMIP_IC2["file_pattern"], a, b)
        for a, b in C96_AMIP_TRAIN_SPLITS
    ]
    entries += [_zarr_entry(SOM_DATA_PATH, f, *SOM_SUBSET) for f in SOM_NEW_MEMBERS]
    return entries


def _labels(concat: list[dict]) -> list[str]:
    labels = []
    for dataset in concat:
        (label,) = dataset["labels"]
        if label not in labels:
            labels.append(label)
    return labels


def make_train_concat(parent_concat: list[dict], arm: str) -> list[dict]:
    concat = copy.deepcopy(parent_concat)
    if arm == ERA5_ONLY_ARM:
        assert _labels(concat) == ["era5"] and len(concat) == 2
    else:
        existing = {(d["data_path"], d["file_pattern"]) for d in concat}
        for entry in extra_c96_entries():
            assert (entry["data_path"], entry["file_pattern"]) not in existing
        # Insert after the last c96 entry so the c96 members stay together.
        last_c96 = max(i for i, d in enumerate(concat) if d["labels"] == ["c96"])
        concat[last_c96 + 1 : last_c96 + 1] = extra_c96_entries()

    n_roundtripped = 0
    for dataset in concat:
        assert "sht_roundtrip_variables" not in dataset
        if set(dataset["labels"]) & set(ROUNDTRIP_LABELS):
            dataset["sht_roundtrip_variables"] = list(ROUNDTRIP_VARIABLES)
            n_roundtripped += 1
    assert (n_roundtripped > 0) == (arm != ERA5_ONLY_ARM)
    return concat


def make_validation(parent_validation: dict, arm: str) -> list[dict]:
    assert list(parent_validation) == ["loader"]
    concat = parent_validation["loader"]["dataset"]["concat"]
    labels = _labels(concat)
    assert labels == (["era5"] if arm == ERA5_ONLY_ARM else ["era5", "c96"])

    def entry(name: str, label: str, weight: float) -> dict:
        loader = copy.deepcopy(parent_validation["loader"])
        loader["dataset"]["concat"] = [
            d for d in loader["dataset"]["concat"] if d["labels"] == [label]
        ]
        assert len(loader["dataset"]["concat"]) == 2
        return {
            "name": name,
            "weight": weight,
            "evaluate_all_steps": False,
            "loader": loader,
        }

    entries = [entry("val", "era5", 1.0)]
    if "c96" in labels:
        entries.append(entry("val_c96", "c96", 0.0))
    return entries


def make_inference(parent_inference: list[dict]) -> list[dict]:
    by_name = {entry["name"]: entry for entry in parent_inference}
    for name, entry in by_name.items():
        expected_weight = 1.0 if name == SELECTION_INFERENCE else 0.0
        assert entry["weight"] == expected_weight, name
    selection = copy.deepcopy(by_name[SELECTION_INFERENCE])
    assert selection["epochs"] == {"start": 0, "step": 1}
    inference = [selection]
    renames = {name: name for name in HEAVY_INFERENCE} | WEATHER_INFERENCE
    for name, new_name in renames.items():
        entry = copy.deepcopy(by_name[name])
        assert entry["epochs"] == {"start": 0, "step": 10}, name
        entry["name"] = new_name
        entry["epochs"] = dict(HEAVY_INFERENCE_EPOCHS)
        inference.append(entry)
    return inference


def make_scheduler(parent_scheduler: dict | None) -> dict | None:
    """The parent's 2-epoch linear warmup (swin only), then LR constant."""
    if parent_scheduler is None:
        return None
    warmup = copy.deepcopy(parent_scheduler["schedulers"][0])
    assert warmup["type"] == "LinearLR"
    assert warmup["kwargs"] == {
        "start_factor": 0.01,
        "end_factor": 1.0,
        "total_iters": 2,
    }
    n_warmup = warmup["kwargs"]["total_iters"]
    constant = {
        "type": "ConstantLR",
        "kwargs": {"factor": 1.0, "total_iters": MAX_EPOCHS - n_warmup},
        "step_each_iteration": False,
    }
    return {"schedulers": [warmup, constant], "milestones": [n_warmup]}


def replace_stats_root(normalization: dict, arm: str) -> dict:
    """Point every stats path of the parent's normalization at the arm's root."""
    old_root, new_root = PARENT_STATS_ROOTS[arm] + "/", STATS_ROOTS[arm] + "/"
    n_replaced = 0

    def replace(node):
        nonlocal n_replaced
        if isinstance(node, dict):
            return {key: replace(value) for key, value in node.items()}
        if isinstance(node, list):
            return [replace(value) for value in node]
        if isinstance(node, str) and node.startswith(NORM_ROOT):
            assert node.startswith(old_root), node
            n_replaced += 1
            return new_root + node[len(old_root) :]
        return node

    normalization = replace(normalization)
    assert n_replaced > 0
    return normalization


def make_config(arch: str, arm: str, seed: int) -> dict:
    parent = yaml.safe_load(parent_path(arch, arm).read_text())
    cfg = copy.deepcopy(parent)
    _, parent_embed_dim, embed_dim = ARCHS[arch]

    cfg["seed"] = seed
    cfg["max_epochs"] = MAX_EPOCHS
    cfg["ema"] = {"decay": 0.9999}
    cfg["validate_using_ema"] = True
    cfg["ema_checkpoint_save_epochs"] = dict(EMA_CHECKPOINT_SAVE_EPOCHS)
    cfg["inference"] = make_inference(parent["inference"])
    cfg["validation"] = make_validation(parent["validation"], arm)

    train_loader = cfg["train_loader"]
    concat = make_train_concat(parent["train_loader"]["dataset"]["concat"], arm)
    train_loader["dataset"]["concat"] = concat
    assert train_loader["time_buffer"] == 8
    labels = _labels(concat)
    assert set(labels) == set(parent["train_loader"]["label_sampling"]["weights"])
    train_loader["label_sampling"] = {
        "weights": {label: round(1 / len(labels), 6) for label in labels},
        "samples_per_epoch": SAMPLES_PER_EPOCH,
    }

    training = cfg["stepper_training"]
    assert training["n_forward_steps"] == 1
    training["n_forward_steps"] = {
        "start_value": 1,
        "milestones": [
            {"epoch": SINGLE_STEP_EPOCHS + 1, "value": MULTI_STEP_N_FORWARD_STEPS}
        ],
    }
    training["optimize_last_step_only"] = False
    assert "pushforward_steps" not in training
    assert "parameter_init" not in training

    optimization = cfg["optimization"]
    assert optimization["lr"] == LR
    assert optimization["use_gradient_accumulation"] is True
    scheduler = make_scheduler(optimization.get("scheduler"))
    if scheduler is None:
        optimization.pop("scheduler", None)
    else:
        optimization["scheduler"] = scheduler

    step_config = cfg["stepper"]["step"]["config"]
    assert "include_channel_mask_inputs" not in step_config
    assert "input_dropout" not in step_config
    step_config["include_channel_mask_inputs"] = True
    normalization = replace_stats_root(step_config["normalization"], arm)
    step_config["normalization"] = normalization
    assert ("grouped" in normalization) == (arm != ERA5_ONLY_ARM)
    builder_config = step_config["builder"]["config"]
    assert builder_config["embed_dim"] == parent_embed_dim
    builder_config["embed_dim"] = embed_dim
    return cfg


def header(arch: str, arm: str) -> str:
    parent = parent_path(arch, arm).name
    lines = [
        "# Generated by create_fm_v2_jobs.py; do not edit by hand.",
        f"# Parent: base_configs/{parent}",
        "# See create_fm_v2_jobs.py for every change from the parent.",
    ]
    return "\n".join(lines) + "\n"


def dump(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=1000)


def main() -> None:
    for arch in ARCHS:
        for arm in ARMS:
            for seed in SEEDS:
                path = RUN_CONFIGS_DIR / run_config_name(arch, arm, seed)
                path.write_text(header(arch, arm) + dump(make_config(arch, arm, seed)))
                print(f"Wrote {path.relative_to(EXP_DIR)}")


if __name__ == "__main__":
    main()

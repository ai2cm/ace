"""Generate the 2026-11-18 FM v3 per-grid-cell normalization runs (wandb group
ace2-fm-sfno-v3-2026-11-18).

18 runs ace2-fm-nc-sfno-v3-fm-e3sm-balanced-{c1,c2,c3}-{scalarloss,cellloss}-
s{0,1,2}-maskoff: 3 network normalizations x 2 loss normalizations x 3 seeds,
each derived from the v2 `fm-e3sm-balanced` maskoff SFNO run config of the same
seed (create_fm_v2_jobs.py, run_configs/ace-train-config-4deg-nc-sfno-v2-
fm-e3sm-balanced-s{seed}-maskoff.yaml; the "parent"). Everything not listed
below is copied from the parent verbatim: model (NoiseConditionedSFNO
embed_dim 1024), 4 GPUs at global batch 8, training data and label sampling
(0.2 per label, 2758 samples per epoch), HGTsfc SHT roundtrip on cm4/xshield,
masking off with channel-mask inputs, EMA 0.9999 with EMA checkpoints every
epoch, EnsembleLoss (crps 0.9, energy 0.1), shared global-mean removal,
pinned variables, multi-step settings (loss on every step, gradients
detached), validation (`val` ERA5 weight 1 for best_ckpt, `val_c96` weight 0)
and inline inference (aimip_checkpoint_era5 every epoch, weight 1, for
best_inference_ckpt; ERA5 10yr/46yr and the two weather ensembles after
epochs 10, 20, ..., weight 0).

Changes from the parent:

1. Training recipe (ext80's schedule mechanics, create_fm_v2_followup_jobs.py,
   shifted to a 120-epoch run from scratch):
   - max_epochs 120 (parent 50), from scratch (no parameter_init).
   - 1 forward step for epochs 1-114, 3 for epochs 115-120 (parent: 3 from
     epoch 41).
   - LR 1e-4 constant with no warmup (as the parent) for epochs 1-114, then
     the 1-sqrt cooldown to 0 over epochs 115-120: ext80's 16-piece
     SequentialLR stepped every iteration, with ITERATIONS_PER_EPOCH = 3105
     iterations per epoch (each of the 4 ranks takes ceil(2758 / 4) = 690
     windows of time_buffer + 1 = 9 batches at 2 samples per rank: 690 * 9 /
     2 = 3105, the value measured in wandb for the v2 runs). The constant
     piece is 114 * 3105 = 353970 iterations and the cooldown 6 * 3105 =
     18630, ending at 120 * 3105 = 372600.
   - Inline inference after epochs 10, 20, ..., 120 for the heavy entries (the
     parent's `epochs: {start: 9, step: 10}`, unchanged).

2. Normalization. STATS_DIR (per-grid-cell stats; --stats-dir) holds pooled
   centering.nc, scaling-full-field.nc, scaling-residual.nc and
   groups/{c96,era5,cm4,xshield,e3sm}/ with the same files; HGTsfc,
   land_fraction, ocean_fraction, sea_ice_fraction and global_mean_co2 are
   scalar entries in them (equal to the v2 values). V2_DIR is the parent's
   scalar stats root.
   - Network, `c1`: pooled per-cell (STATS_DIR centering.nc +
     scaling-full-field.nc, per_grid_cell true, per_grid_cell_min_std_fraction
     0.01 = the default), no grouped block (the parent has one).
   - Network, `c2`: the same pooled block plus `grouped` with per-cell groups
     c96/era5/cm4/xshield/e3sm from STATS_DIR/groups/<label>/ (centering.nc +
     scaling-full-field.nc, per_grid_cell true), default_label era5 and the
     parent's pinned_variables. The e3sm group is groups/e3sm (the parent used
     groups/e3sm-filled, which adds TMP850 and h500 that the E3SM store
     lacks; STATS_DIR's e3sm group must provide them, see
     validate_fm_v3_configs.py --stats-dir-local).
   - Loss, `scalarloss`: `loss` = V2_DIR centering.nc + scaling-full-field.nc
     and `residual` = V2_DIR centering.nc + scaling-residual.nc, both scalar.
     The parent has no `loss` block, so its loss normalizer is its `network`
     block (the same V2_DIR files) overridden by its `residual` block for the
     prognostic names; these arms state that explicitly (a per-cell network
     requires an explicit loss block) and reproduce the v2 loss exactly.
   - Loss, `cellloss`: `loss` = STATS_DIR centering.nc + scaling-full-field.nc
     and `residual` = STATS_DIR centering.nc + scaling-residual.nc, both
     per_grid_cell true with per_grid_cell_min_std_fraction 0.1: prognostics
     are scored in per-cell tendency-std units, diagnostics in per-cell
     full-field units; pooled, not grouped, in both c1 and c2.

   - Network, `c3`: per-cell normalization grouped per config. The pooled
     block as c1/c2, plus `grouped` with one per-cell group per C3 label from
     STATS_DIR_C3/groups/<label>/ (--stats-dir-c3; its timestep-pooled root
     files are not used), default_label era5, the parent's pinned_variables.
     Labels drive both grouping and conditioning, so the datasets are
     relabelled (loss blocks as for c1/c2):
       c96 -> amip    (2026-01-28-...-c96-...-shield-amip-ensemble-dataset:
                       ic_0001, ic_0002 x 2 year splits; 4 entries)
              ramped  (2026-06-08-...-ramped-climSST-random-CO2-...: 6)
              som     (2026-06-08-...-c96-shield-som-ensemble-...: 14)
       cm4 -> cm4-picontrol (2025-03-21-CM4-piControl-...: 1)
              cm4-1pctco2   (2025-06-18-CM4-1pctCO2-...: 1)
              am4-randco2   (2026-10-11-CM4-like-AM4-random-CO2-...: 9)
       era5, xshield, e3sm unchanged (2, 1, 1 entries).
     The assignment is by data_path and matches the C3 stats config
     (scripts/data_process/configs/fm-v3-e3sm-balanced-percell-c3-stats.yaml)
     entry for entry (validate_fm_v3_configs.py checks). val_c96 (C96 AMIP
     ic_0001) is labelled amip; the inline inference datasets are ERA5.
     label_sampling keeps each source's 0.2 and splits c96 and cm4 across
     their labels in proportion to training timesteps (C3_TIMESTEPS: amip
     21186, ramped 10968, som 51142; cm4-picontrol 73000, cm4-1pctco2 51100,
     am4-randco2 17253): amip 0.050869, ramped 0.026335, som 0.122796,
     cm4-picontrol 0.103288, cm4-1pctco2 0.072301, am4-randco2 0.024411.
     Within a label the sampler draws uniformly over its samples (time_buffer
     windows, one per 9 timesteps of the concatenated dataset), so c1/c2 draw
     each sub-source in proportion to its windows only in expectation (the
     split varies per epoch); c3 fixes it at the timestep proportions, which
     differ from the window proportions by the few windows lost at entry
     boundaries.

3. Seeds 0, 1, 2 (submit_fm_v3_jobs.py: s0 at Beaker priority high, s1/s2
   normal).

Output directory: the configs keep the parent's `experiment_dir: /results`,
but submit_fm_v3_jobs.py overrides it at job start with
`--override experiment_dir=$AGENT_OUTPUT_DIR`, the agent proxy's per-experiment
folder under /climate-default/agents/alexeyy/, which survives Beaker's
automatic resume of a preempted job (header line EXPERIMENT_DIR).

Each file starts with a `# KEY: value` header (create_anneal_jobs.read_header)
naming the job, its wandb group, its parent and the climate-default folders it
needs mounted (WEKA_MOUNT; the gantry shim mounts every /climate-default/<dir>
named in the config text, and the ERA5 store is otherwise named only as
`data_path: /climate-default` + `file_pattern`).

Smoke test (always written): ...-c2-cellloss-s0-maskoff-smoke.yaml, the
c2-cellloss s0 config with these changes only, so a short job exercises the
per-cell network, grouped and loss normalizers, a 1-step and a 3-step epoch
with the LR cooldown, validation, inline inference and checkpointing:
- max_epochs 2: epoch 1 is 1-step at LR 1e-4, epoch 2 is 3-step with the
  1-sqrt cooldown to 0 (same scheduler construction, 288 iterations per
  epoch).
- samples_per_epoch 256 (288 iterations per epoch: 64 windows per rank x 9 /
  2), the smallest that fills each rank's time_buffer_pool_size 32 pool.
- log_train_every_n_batches 20.
- Validation is as in the real runs (full years 1994, 2014): with
  time_buffer 16 and 8 samples per rank per batch, shorter subsets leave the
  validation loaders with no batches (the first smoke job failed so).
- Inline inference every epoch: aimip_checkpoint_era5 (weight 1) with 100
  forward steps (1825 in the real runs) and weather_2020_era5 (weight 0); the
  10yr/46yr and weather_1950 entries are dropped.
- wandb group ace2-fm-sfno-v3-2026-11-18-smoke.

Usage:
    python create_fm_v3_jobs.py [--stats-dir DIR]
"""

import argparse
import copy
import datetime
import math
import pathlib
from dataclasses import dataclass

import yaml
from create_fm_v2_followup_jobs import (
    BATCH_SIZE,
    ITERATIONS_PER_EPOCH,
    PARENT_LR,
    TIME_BUFFER,
    check_iterations_per_epoch,
    lr_scheduler,
    set_steps,
)
from create_fm_v2_jobs import (
    CONFIG_PREFIX,
    MULTI_STEP_N_FORWARD_STEPS,
    RUN_CONFIGS_DIR,
    SAMPLES_PER_EPOCH,
    SEEDS,
    STATS_ROOTS,
    run_config_name,
)

EXP_DIR = RUN_CONFIGS_DIR.parent
JOB_PREFIX = "ace2-fm-"
WANDB_GROUP = "ace2-fm-sfno-v3-2026-11-18"
SMOKE_WANDB_GROUP = f"{WANDB_GROUP}-smoke"

# Per-grid-cell stats root (see the module docstring): 51 per-cell variables
# on the 45 x 90 grid (dims grid_yt/grid_xt) and the 5 scalars, written by the
# agent job agent-alexeyy-fm-v3-percell-stats-75w7; groups/e3sm holds
# placeholder TMP850/h500 constants (E3SM lacks them).
STATS_DIR = (
    "/climate-default/agents/alexeyy/agent-alexeyy-fm-v3-percell-stats-75w7/"
    "2026-11-18-fm-v3-e3sm-balanced-percell"
)
PARENT_ARCH = "nc-sfno-v2"
ARCH = "nc-sfno-v3"
ARM = "fm-e3sm-balanced"
V2_DIR = STATS_ROOTS[ARM]
NETWORKS = ("c1", "c2", "c3")
LOSSES = ("scalarloss", "cellloss")
GROUP_LABELS = ("c96", "era5", "cm4", "xshield", "e3sm")
PARENT_GROUP_DIRS = {label: label for label in GROUP_LABELS} | {"e3sm": "e3sm-filled"}
DEFAULT_LABEL = "era5"
NETWORK_MIN_STD_FRACTION = 0.01  # NormalizationConfig's default

# C3: per-config groups (see the module docstring). Only STATS_DIR_C3/groups/
# is used; its root files are timestep-pooled and are not read.
STATS_DIR_C3 = (
    "/climate-default/agents/alexeyy/agent-alexeyy-fm-v3-percell-c3-stats-legy/"
    "2026-11-18-fm-v3-e3sm-balanced-percell-c3"
)
# Parent label -> {last data_path component: C3 label}; other labels keep
# their name.
C3_SPLITS = {
    "c96": {
        "2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-dataset": (
            "amip"
        ),
        "2026-06-08-vertically-resolved-c96-shield-ramped-climSST-random-CO2-"
        "ensemble-fme-dataset-4deg-daily": "ramped",
        "2026-06-08-vertically-resolved-4deg-daily-c96-shield-som-ensemble-"
        "fme-dataset": "som",
    },
    "cm4": {
        "2025-03-21-CM4-piControl-atmosphere-land-4deg-8layer-200yr-daily": (
            "cm4-picontrol"
        ),
        "2025-06-18-CM4-1pctCO2-atmosphere-land-4deg-8layer-140yr-daily": (
            "cm4-1pctco2"
        ),
        "2026-10-11-CM4-like-AM4-random-CO2-4deg-daily": "am4-randco2",
    },
}
C3_LABELS = (
    "era5",
    "amip",
    "ramped",
    "som",
    "cm4-picontrol",
    "cm4-1pctco2",
    "am4-randco2",
    "xshield",
    "e3sm",
)
# Training entries per C3 label, as in the C3 stats config
# (scripts/data_process/configs/fm-v3-e3sm-balanced-percell-c3-stats.yaml).
C3_ENTRIES = {
    "era5": 2,
    "amip": 4,
    "ramped": 6,
    "som": 14,
    "cm4-picontrol": 1,
    "cm4-1pctco2": 1,
    "am4-randco2": 9,
    "xshield": 1,
    "e3sm": 1,
}
# Training timesteps per C3 label (inclusive subset windows, whole stores
# otherwise), counted by the C3 stats job; the subset-windowed ones are
# re-counted from the configs by c3_window_timesteps.
C3_TIMESTEPS = {
    "era5": 24836,
    "amip": 21186,
    "ramped": 10968,
    "som": 51142,
    "cm4-picontrol": 73000,
    "cm4-1pctco2": 51100,
    "am4-randco2": 17253,
    "xshield": 4020,
    "e3sm": 38690,
}
LOSS_MIN_STD_FRACTION = 0.1

N_GPUS = 4
MAX_EPOCHS = 120
COOLDOWN_EPOCHS = 6  # epochs 115-120: 3 forward steps + 1-sqrt LR cooldown
MULTI_STEP_FROM = MAX_EPOCHS - COOLDOWN_EPOCHS + 1

SMOKE_NETWORK = "c2"
SMOKE_LOSS = "cellloss"
SMOKE_SEED = 0
SMOKE_MAX_EPOCHS = 2
SMOKE_COOLDOWN_EPOCHS = 1
SMOKE_SAMPLES_PER_EPOCH = 256
SMOKE_LOG_TRAIN_EVERY_N_BATCHES = 20
SELECTION_INFERENCE = "aimip_checkpoint_era5"
SMOKE_SELECTION_N_FORWARD_STEPS = 100
SMOKE_WEATHER_INFERENCE = "weather_2020_era5"
EVERY_EPOCH = {"start": 0, "step": 1}

# Top-level climate-default folders named only through data_path
# /climate-default + file_pattern, which the gantry shim does not see.
ERA5_STORE = "/climate-default/2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr"


def iterations_per_epoch(samples_per_epoch: int) -> int:
    """Optimizer steps per epoch: each rank takes ceil(samples / N_GPUS)
    time_buffer windows of TIME_BUFFER + 1 batches of BATCH_SIZE / N_GPUS.
    """
    windows_per_rank = math.ceil(samples_per_epoch / N_GPUS)
    batches = windows_per_rank * (TIME_BUFFER + 1)
    per_rank_batch = BATCH_SIZE // N_GPUS
    assert batches % per_rank_batch == 0
    return batches // per_rank_batch


assert iterations_per_epoch(SAMPLES_PER_EPOCH) == ITERATIONS_PER_EPOCH


@dataclass(frozen=True)
class Run:
    network: str
    loss: str
    seed: int
    smoke: bool = False

    @property
    def stem(self) -> str:
        suffix = "-smoke" if self.smoke else ""
        return f"{ARCH}-{ARM}-{self.network}-{self.loss}-s{self.seed}-maskoff{suffix}"

    @property
    def config_name(self) -> str:
        return f"{CONFIG_PREFIX}{self.stem}.yaml"

    @property
    def job_name(self) -> str:
        # Same as submit_fm_jobs.config_to_job_name(self.config_name).
        return f"{JOB_PREFIX}{self.stem}"

    @property
    def path(self) -> pathlib.Path:
        return RUN_CONFIGS_DIR / self.config_name

    @property
    def group(self) -> str:
        return SMOKE_WANDB_GROUP if self.smoke else WANDB_GROUP

    @property
    def parent_config(self) -> pathlib.Path:
        return RUN_CONFIGS_DIR / run_config_name(PARENT_ARCH, ARM, self.seed, False)


def runs() -> list[Run]:
    return [
        Run(network, loss, seed)
        for network in NETWORKS
        for loss in LOSSES
        for seed in SEEDS
    ]


def smoke_run() -> Run:
    """The main smoke test (c2-cellloss)."""
    return Run(SMOKE_NETWORK, SMOKE_LOSS, SMOKE_SEED, smoke=True)


def smoke_runs() -> list[Run]:
    """The main smoke test and the C3 one (c3-cellloss)."""
    return [smoke_run(), Run("c3", SMOKE_LOSS, SMOKE_SEED, smoke=True)]


# --- normalization ---------------------------------------------------------------


def _stats(root: str, scaling: str, **extra) -> dict:
    return {
        "global_means_path": f"{root}/centering.nc",
        "global_stds_path": f"{root}/{scaling}.nc",
        **extra,
    }


def _per_cell(root: str, scaling: str, min_std_fraction: float) -> dict:
    return _stats(
        root,
        scaling,
        per_grid_cell=True,
        per_grid_cell_min_std_fraction=min_std_fraction,
    )


def check_parent_normalization(normalization: dict) -> None:
    """The parent's normalization is the scalar V2_DIR stats with the default
    loss (its network block, overridden by its residual block).
    """
    assert normalization["network"] == _stats(V2_DIR, "scaling-full-field")
    assert normalization["residual"] == _stats(V2_DIR, "scaling-residual")
    assert "loss" not in normalization
    grouped = normalization["grouped"]
    assert set(grouped) == {"groups", "default_label", "pinned_variables"}
    assert list(grouped["groups"]) == list(GROUP_LABELS)
    for label, group in grouped["groups"].items():
        root = f"{V2_DIR}/groups/{PARENT_GROUP_DIRS[label]}"
        assert group == _stats(root, "scaling-full-field"), label
    assert grouped["default_label"] == DEFAULT_LABEL


def make_normalization(
    parent_normalization: dict,
    network: str,
    loss: str,
    stats_dir: str,
    stats_dir_c3: str = STATS_DIR_C3,
) -> dict:
    check_parent_normalization(parent_normalization)
    normalization: dict = {
        "network": _per_cell(stats_dir, "scaling-full-field", NETWORK_MIN_STD_FRACTION)
    }
    if loss == "scalarloss":
        # Exactly the parent's loss normalizer: loss defaulted to its network
        # block, with its residual block for the prognostic names.
        normalization["loss"] = copy.deepcopy(parent_normalization["network"])
        normalization["residual"] = copy.deepcopy(parent_normalization["residual"])
    elif loss == "cellloss":
        normalization["loss"] = _per_cell(
            stats_dir, "scaling-full-field", LOSS_MIN_STD_FRACTION
        )
        normalization["residual"] = _per_cell(
            stats_dir, "scaling-residual", LOSS_MIN_STD_FRACTION
        )
    else:
        raise ValueError(loss)
    if network in ("c2", "c3"):
        parent_grouped = parent_normalization["grouped"]
        groups_root, labels = (
            (stats_dir, GROUP_LABELS) if network == "c2" else (stats_dir_c3, C3_LABELS)
        )
        normalization["grouped"] = {
            "groups": {
                label: _per_cell(
                    f"{groups_root}/groups/{label}",
                    "scaling-full-field",
                    NETWORK_MIN_STD_FRACTION,
                )
                for label in labels
            },
            "default_label": DEFAULT_LABEL,
            "pinned_variables": list(parent_grouped["pinned_variables"]),
        }
    elif network != "c1":
        raise ValueError(network)
    return normalization


# --- configs ---------------------------------------------------------------------


def _c3_label(dataset: dict) -> str:
    (label,) = dataset["labels"]
    if label not in C3_SPLITS:
        return label
    return C3_SPLITS[label][dataset["data_path"].rstrip("/").split("/")[-1]]


def _relabel(concat: list[dict]) -> None:
    for dataset in concat:
        dataset["labels"] = [_c3_label(dataset)]


def _inclusive_days(subset: dict) -> int:
    start = datetime.date.fromisoformat(subset["start_time"])
    stop = datetime.date.fromisoformat(subset["stop_time"])
    return (stop - start).days + 1


def c3_window_timesteps(concat: list[dict]) -> dict[str, int]:
    """Daily timesteps per C3 label of the subset-windowed entries (labels
    whose entries all have a subset).
    """
    totals: dict[str, int] = {}
    windowed: dict[str, bool] = {}
    for dataset in concat:
        label = _c3_label(dataset)
        subset = dataset.get("subset")
        windowed[label] = windowed.get(label, True) and subset is not None
        if subset is not None:
            totals[label] = totals.get(label, 0) + _inclusive_days(subset)
    return {label: n for label, n in totals.items() if windowed[label]}


def c3_weights(parent_weights: dict[str, float]) -> dict[str, float]:
    """Each parent label's weight, split across its C3 labels in proportion
    to their training timesteps.
    """
    weights = {}
    for parent_label, weight in parent_weights.items():
        children = list(
            C3_SPLITS.get(parent_label, {parent_label: parent_label}).values()
        )
        total = sum(C3_TIMESTEPS[c] for c in children)
        for child in children:
            weights[child] = round(weight * C3_TIMESTEPS[child] / total, 6)
    return {label: weights[label] for label in C3_LABELS}


def make_c3_data(cfg: dict) -> None:
    """Relabel the training and validation datasets per config (C3) and split
    the label sampling weights.
    """
    loader = cfg["train_loader"]
    concat = loader["dataset"]["concat"]
    _relabel(concat)
    counts: dict[str, int] = {}
    for dataset in concat:
        (label,) = dataset["labels"]
        counts[label] = counts.get(label, 0) + 1
    assert counts == C3_ENTRIES, counts
    for label, n in c3_window_timesteps(concat).items():
        assert n == C3_TIMESTEPS[label], (label, n)
    sampling = loader["label_sampling"]
    assert sampling["weights"] == {label: 0.2 for label in GROUP_LABELS}
    sampling["weights"] = c3_weights(sampling["weights"])
    for entry in cfg["validation"]:
        _relabel(entry["loader"]["dataset"]["concat"])
    for entry in cfg["inference"]:
        assert entry["loader"]["dataset"]["labels"] == ["era5"], entry["name"]


def make_config(run: Run, stats_dir: str, stats_dir_c3: str = STATS_DIR_C3) -> dict:
    parent = yaml.safe_load(run.parent_config.read_text())
    cfg = copy.deepcopy(parent)
    check_iterations_per_epoch(cfg)
    assert parent["seed"] == run.seed
    assert parent["max_epochs"] == 50
    assert parent["ema"] == {"decay": 0.9999}
    assert parent["ema_checkpoint_save_epochs"] == {"start": 1, "step": 1}
    assert parent["logging"]["project"] == "FM"
    cfg["max_epochs"] = MAX_EPOCHS

    training = cfg["stepper_training"]
    assert "parameter_init" not in training
    assert training["loss"] == {
        "type": "EnsembleLoss",
        "kwargs": {"crps_weight": 0.9, "energy_score_weight": 0.1},
    }
    set_steps(cfg, MULTI_STEP_FROM)
    assert training["n_forward_steps"]["milestones"] == [
        {"epoch": MULTI_STEP_FROM, "value": MULTI_STEP_N_FORWARD_STEPS}
    ]

    optimization = cfg["optimization"]
    assert optimization["lr"] == PARENT_LR
    assert optimization["use_gradient_accumulation"] is True
    assert "scheduler" not in optimization
    optimization["scheduler"] = lr_scheduler("none", MAX_EPOCHS, COOLDOWN_EPOCHS)

    step_config = cfg["stepper"]["step"]["config"]
    assert step_config["builder"]["type"] == "NoiseConditionedSFNO"
    assert step_config["builder"]["config"]["embed_dim"] == 1024
    assert step_config["include_channel_mask_inputs"] is True
    assert "input_dropout" not in step_config
    assert step_config["global_mean_removal"] == {
        "kind": "shared",
        "append_as_input": True,
    }
    step_config["normalization"] = make_normalization(
        parent["stepper"]["step"]["config"]["normalization"],
        run.network,
        run.loss,
        stats_dir,
        stats_dir_c3,
    )
    if run.network == "c3":
        make_c3_data(cfg)
    if run.smoke:
        _make_smoke(cfg)
    return cfg


def _smoke_inference(parent_inference: list[dict]) -> list[dict]:
    by_name = {entry["name"]: entry for entry in parent_inference}
    selection = copy.deepcopy(by_name[SELECTION_INFERENCE])
    assert selection["weight"] == 1.0
    assert selection["epochs"] == EVERY_EPOCH
    assert selection["n_forward_steps"] > SMOKE_SELECTION_N_FORWARD_STEPS
    selection["n_forward_steps"] = SMOKE_SELECTION_N_FORWARD_STEPS
    weather = copy.deepcopy(by_name[SMOKE_WEATHER_INFERENCE])
    assert weather["weight"] == 0.0
    weather["epochs"] = dict(EVERY_EPOCH)
    return [selection, weather]


def _make_smoke(cfg: dict) -> None:
    cfg["max_epochs"] = SMOKE_MAX_EPOCHS
    assert "log_train_every_n_batches" not in cfg
    cfg["log_train_every_n_batches"] = SMOKE_LOG_TRAIN_EVERY_N_BATCHES
    loader = cfg["train_loader"]
    assert loader["time_buffer_pool_size"] * BATCH_SIZE == SMOKE_SAMPLES_PER_EPOCH
    loader["label_sampling"]["samples_per_epoch"] = SMOKE_SAMPLES_PER_EPOCH
    set_steps_from = SMOKE_MAX_EPOCHS - SMOKE_COOLDOWN_EPOCHS + 1
    cfg["stepper_training"]["n_forward_steps"]["milestones"][0]["epoch"] = (
        set_steps_from
    )
    cfg["optimization"]["scheduler"] = lr_scheduler(
        "none",
        SMOKE_MAX_EPOCHS,
        SMOKE_COOLDOWN_EPOCHS,
        iterations_per_epoch=iterations_per_epoch(SMOKE_SAMPLES_PER_EPOCH),
    )
    cfg["inference"] = _smoke_inference(cfg["inference"])


def describe(run: Run) -> str:
    network = {
        "c1": "pooled per-cell network normalization",
        "c2": "pooled + grouped per-cell network normalization",
        "c3": "pooled + per-config grouped per-cell network normalization "
        "(9 relabelled datasets)",
    }[run.network]
    loss = {
        "scalarloss": "v2 scalar loss normalization",
        "cellloss": "pooled per-cell loss normalization (floor 0.1)",
    }[run.loss]
    if run.smoke:
        return (
            f"ACE FM {ARCH} smoke test: {SMOKE_MAX_EPOCHS} short epochs (1-step, "
            f"then 3-step + cooldown), {network}, {loss}"
        )
    return (
        f"ACE FM {ARCH}: {MAX_EPOCHS} epochs from scratch, LR {PARENT_LR:g} "
        f"constant, 3-step + 1-sqrt cooldown to 0 over epochs "
        f"{MULTI_STEP_FROM}-{MAX_EPOCHS}, {network}, {loss}, masking off"
    )


def header(run: Run) -> str:
    lines = [
        "Generated by create_fm_v3_jobs.py; do not edit by hand.",
        f"JOB_NAME: {run.job_name}",
        f"WANDB_GROUP: {run.group}",
        f"PARENT_CONFIG: run_configs/{run.parent_config.name}",
        f"WEKA_MOUNT: {ERA5_STORE}",
        "EXPERIMENT_DIR: $AGENT_OUTPUT_DIR (set by submit_fm_v3_jobs.py)",
        f"DESCRIPTION: {describe(run)}",
        "See create_fm_v3_jobs.py for every change from the parent.",
    ]
    return "".join(f"# {line}\n" for line in lines)


def write_config(run: Run, stats_dir: str, stats_dir_c3: str) -> None:
    body = yaml.safe_dump(
        make_config(run, stats_dir, stats_dir_c3),
        sort_keys=False,
        default_flow_style=False,
        width=1000,
    )
    run.path.write_text(header(run) + body)
    print(f"Wrote {run.path.relative_to(EXP_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stats-dir",
        default=STATS_DIR,
        help=f"Per-grid-cell stats root (default: {STATS_DIR}).",
    )
    parser.add_argument(
        "--stats-dir-c3",
        default=STATS_DIR_C3,
        help=f"C3 per-config stats root, groups/ only (default: {STATS_DIR_C3}).",
    )
    args = parser.parse_args()
    stats_dir = args.stats_dir.rstrip("/")
    stats_dir_c3 = args.stats_dir_c3.rstrip("/")
    for path in (stats_dir, stats_dir_c3):
        if not path.startswith("/climate-default/"):
            parser.error(f"{path} is not a /climate-default/ path")
    for run in runs() + smoke_runs():
        write_config(run, stats_dir, stats_dir_c3)


if __name__ == "__main__":
    main()

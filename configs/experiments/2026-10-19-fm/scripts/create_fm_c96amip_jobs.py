"""Generate the 2026-11-18 FM C96 AMIP-only runs and their ERA5-only
`era5-1979` baseline (wandb group ace2-fm-c96amip-2026-11-18).

6 runs ace2-fm-{nc-sfno-v2,nc-swin-v2.2-v2}-fm-c96amip-s{0,1,2}-maskoff:
2 architectures x 3 seeds, trained on C96 SHiELD AMIP data only. Each is
derived from the v2 `fm-balanced` maskoff run config of the same architecture
and seed (create_fm_v2_jobs.py, run_configs/ace-train-config-4deg-
{arch}-fm-balanced-s{seed}-maskoff.yaml; the "parent"), the v2 arm whose
training data includes the C96 AMIP ic_0001 entries and that validates on
C96. Everything not listed below is copied from the parent verbatim: model
(nc-sfno-v2: NoiseConditionedSFNO embed_dim 1024; nc-swin-v2.2-v2:
NoiseConditionedSwinTransformer embed_dim 256, the v2 size: the 224 that
create_fm_v2_jobs.py tried crashed in the first compiled backward),
`conditional: true`, 4 GPUs at global batch 8, time_buffer 8, EMA 0.9999
with validate_using_ema and EMA checkpoints every epoch, EnsembleLoss (crps
0.9, energy 0.1), masking off with channel-mask inputs, shared global-mean
removal, in/out variables, multi-step settings (loss on every step,
gradients detached), the optimizer (FusedAdam, weight decay 0.01, LR 1e-4),
the validation loaders (1994 + 2014, full years) and the inline inference
entries and their aggregators.

Changes from the parent:

1. Training data: only the parent's two C96 AMIP ic_0001 entries
   (/climate-default/2026-01-28-vertically-resolved-c96-4deg-daily-shield-
   amip-ensemble-dataset, ic_0001.zarr, 1979-01-01..1993-12-31 and
   1995-01-01..2008-12-31, label c96; 5479 + 5114 daily steps). label_sampling
   weights {c96: 1.0}, samples_per_epoch 2758 (as v2), so ITERATIONS_PER_EPOCH
   stays 3105 (asserted). The ~1176 time_buffer windows of these splits are
   drawn ~2.3 times per epoch (the sampler cycles through fresh shuffles).
   The model's label encoding is built from the training labels, so it knows
   only `c96`.

2. Normalization: scalar stats pooled over exactly that training data
   (STATS_DIRS[fm-c96amip], written by get_pooled_stats.py with
   scripts/data_process/configs/fm-c96amip-stats.yaml and checked by
   verify_c96amip_stats.py in the same job): network = its centering.nc
   + scaling-full-field.nc, residual = its centering.nc +
   scaling-residual.nc (so the loss normalizer is the network block with the
   residual block for the prognostic names, as in v2). No grouped block (the
   parent has one, with groups c96/era5/cm4/xshield).

3. Recipe (as the FM v3 runs, create_fm_v3_jobs.py), from scratch:
   - max_epochs 120 (parent 50).
   - 1 forward step for epochs 1-114, 3 for epochs 115-120 (parent: 3 from
     epoch 41).
   - LR 1e-4 constant for epochs 1-114, then the 1-sqrt cooldown to 0 over
     epochs 115-120: create_fm_v2_followup_jobs.lr_scheduler's SequentialLR
     stepped every iteration, 3105 iterations per epoch (constant piece and
     16 cooldown pieces; 120 * 3105 = 372600 iterations). SFNO: no warmup (the
     parent has no scheduler). Swin: the parent's 2-epoch warmup kept, as
     per-iteration ConstantLR pieces of one epoch each at factors 0.01 and
     0.505 (exactly the parent's per-epoch LinearLR 0.01 -> 1 over 2 epochs),
     then constant.

4. Validation (best_ckpt selection): `val_c96` (C96 AMIP ic_0001, 1994 +
   2014) weight 1 (parent 0); `val` (ERA5, 1994 + 2014) weight 0 (parent 1),
   logged only.

5. Inline inference (best_inference_ckpt selection): a new
   `aimip_checkpoint_c96`, the parent's aimip_checkpoint_era5 (8 ICs from
   2009-01-01 to 2009-11-15, 1825 daily steps each, every epoch, same
   aggregator) on the C96 AMIP ic_0001 store, label c96, weight 1. The ERA5
   entries are kept for comparison at weight 0: aimip_checkpoint_era5 (every
   epoch; parent weight 1), and 10year/46yr and the two weather ensembles
   after epochs 10, 20, ..., 120 (the parent's `epochs: {start: 9, step: 10}`,
   unchanged). The C96 store is daily and gap-free from 1979 through 2014
   (verify_c96amip_stats.py), covering the inference (last run ends
   2014-11-14) and validation windows; it holds every in/out variable
   (training reads them) and its own forcings (the inference loader reads
   forcing from the same store, as for ERA5).

6. Labels of the ERA5 evaluation data: `c96` (parent `era5`) in `val` and
   every ERA5 inline inference entry. The model knows only `c96`; fme drops
   labels missing from the model's encoding (BatchLabels.
   conform_to_encoding, with a warning per batch), which would feed ERA5 an
   all-zero label vector the model never saw in training. Labelled `c96`,
   ERA5 is evaluated as the C96-trained model would be run on it.

7. Seeds 0, 1, 2 (submit_fm_c96amip_jobs.py: s0 at Beaker priority high,
   s1/s2 normal).

ERA5-only baseline, arm `fm-era5-1979` (same wandb group): 6 runs
ace2-fm-{nc-sfno-v2,nc-swin-v2.2-v2}-fm-era5-1979-s{0,1,2}-maskoff, matched to
the C96 AMIP-only runs in data amount (the same year splits, so the same
5479 + 5114 daily steps) and identical to them (same parent, model, recipe,
2758 samples per epoch, conditional with a single label, scalar
normalization without grouping) except:
- Training data: the parent's two ERA5 entries (data_path /climate-default,
  file_pattern 2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr, no renames
  or derived variables) with the C96 splits: 1979-01-01..1993-12-31 (parent
  1940-01-03..1993-12-31) and 1995-01-01..2008-12-31. Label `era5`,
  label_sampling weights {era5: 1.0}.
- Normalization: scalar stats pooled over exactly that ERA5 data
  (STATS_DIRS[fm-era5-1979],
  scripts/data_process/configs/fm-era5-1979-stats.yaml, checked in the same
  job by verify_c96amip_stats.py --group era5).
- Selection as v2 fm-era5: `val` (ERA5) weight 1 and `val_c96` weight 0;
  aimip_checkpoint_era5 weight 1 and aimip_checkpoint_c96 weight 0.
- Every validation and inline inference dataset (C96 included) labelled
  `era5`, the model's only label (as in item 6, mirrored).
No GPU smoke test for this arm (it differs from the smoke-tested C96 arm
only in data, stats and weights); validate_fm_c96amip_configs.py builds and
trains its s0 configs on CPU.

Output directory: the configs keep the parent's `experiment_dir: /results`,
but submit_fm_c96amip_jobs.py overrides it at job start with
`--override experiment_dir=$AGENT_OUTPUT_DIR`, the agent proxy's
per-experiment folder under /climate-default/agents/alexeyy/, which survives
Beaker's automatic resume of a preempted job (header line EXPERIMENT_DIR).

Each file starts with a `# KEY: value` header (create_anneal_jobs.read_header)
naming the job, its wandb group, its parent and the climate-default folders it
needs mounted (WEKA_MOUNT; the gantry shim mounts every /climate-default/<dir>
named in the config text, and the ERA5 store is otherwise named only as
`data_path: /climate-default` + `file_pattern`).

Smoke tests (always written): ...-{nc-sfno-v2,nc-swin-v2.2-v2}-fm-c96amip-
s0-maskoff-smoke.yaml, the s0 configs with these changes only, so a short job
exercises a 1-step and a 3-step epoch with the LR cooldown, both validations,
inline inference and checkpointing:
- max_epochs 2: epoch 1 is 1-step at LR 1e-4, epoch 2 is 3-step with the
  1-sqrt cooldown to 0 (288 iterations per epoch). No warmup for either
  architecture (the Swin warmup alone is longer than the smoke run; its
  pieces are checked by validate_fm_c96amip_configs.py).
- samples_per_epoch 256 (64 windows per rank x 9 / 2 = 288 iterations), the
  smallest that fills each rank's time_buffer_pool_size 32 pool.
- log_train_every_n_batches 20.
- Validation as in the real runs (full years; shorter subsets leave the
  validation loaders with no batches at time_buffer 16).
- Inline inference every epoch, each with 100 forward steps (1825 in the
  real runs): aimip_checkpoint_c96 (weight 1) and aimip_checkpoint_era5
  (weight 0); plus weather_2020_era5 (weight 0, 5 steps). The 10yr/46yr and
  weather_1950 entries are dropped.
- wandb group ace2-fm-c96amip-2026-11-18-smoke.

Usage:
    python create_fm_c96amip_jobs.py [--stats-dir-c96amip DIR]
                                     [--stats-dir-era5-1979 DIR]
"""

import argparse
import copy
import pathlib
from dataclasses import dataclass

import yaml
from create_fm_v2_followup_jobs import (
    ITERATIONS_PER_EPOCH,
    PARENT_LR,
    PARENT_WARMUP_FACTORS,
    check_iterations_per_epoch,
    lr_scheduler,
    set_steps,
)
from create_fm_v2_jobs import (
    CONFIG_PREFIX,
    HEAVY_INFERENCE_EPOCHS,
    MULTI_STEP_N_FORWARD_STEPS,
    RUN_CONFIGS_DIR,
    SAMPLES_PER_EPOCH,
    SEEDS,
    run_config_name,
)
from create_fm_v3_jobs import ERA5_STORE, N_GPUS, iterations_per_epoch

ERA5_FILE = ERA5_STORE.split("/")[-1]

EXP_DIR = RUN_CONFIGS_DIR.parent
JOB_PREFIX = "ace2-fm-"
WANDB_GROUP = "ace2-fm-c96amip-2026-11-18"
SMOKE_WANDB_GROUP = f"{WANDB_GROUP}-smoke"

# Scalar stats pooled over the training data (fm-c96amip-stats.yaml), written
# by the agent job agent-alexeyy-fm-c96amip-stats-rru2.
ARM = "fm-c96amip"
ERA5_ARM = "fm-era5-1979"
ARMS = (ARM, ERA5_ARM)
# Scalar stats pooled over each arm's training data
# (scripts/data_process/configs/fm-c96amip-stats.yaml, fm-era5-1979-stats.yaml),
# written by the agent jobs agent-alexeyy-fm-{c96amip,era5-1979}-stats-*.
STATS_DIRS = {
    ARM: "/climate-default/agents/alexeyy/agent-alexeyy-fm-c96amip-stats-rru2/"
    "2026-11-18-fm-c96amip",
    ERA5_ARM: "/climate-default/agents/alexeyy/agent-alexeyy-fm-era5-1979-stats-vziz/"
    "2026-11-18-fm-era5-1979",
}
SFNO = "nc-sfno-v2"
SWIN = "nc-swin-v2.2-v2"
ARCHS = (SFNO, SWIN)
EMBED_DIMS = {SFNO: 1024, SWIN: 256}
BUILDERS = {SFNO: "NoiseConditionedSFNO", SWIN: "NoiseConditionedSwinTransformer"}
WARMUPS = {SFNO: "none", SWIN: "parent"}
PARENT_ARM = "fm-balanced"
# The arm's only label: of its training data, and given to every validation
# and inline inference dataset.
LABELS = {ARM: "c96", ERA5_ARM: "era5"}
PARENT_LABELS = ("c96", "era5", "cm4", "xshield")

C96_AMIP_DATA_PATH = (
    "/climate-default/"
    "2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-dataset"
)
C96_AMIP_FILE = "ic_0001.zarr"
TRAIN_SPLITS = (("1979-01-01", "1993-12-31"), ("1995-01-01", "2008-12-31"))
# The parent's ERA5 training splits; era5-1979 starts the first in 1979.
PARENT_ERA5_SPLITS = (("1940-01-03", "1993-12-31"), ("1995-01-01", "2008-12-31"))
VALIDATION_YEARS = ("1994", "2014")

MAX_EPOCHS = 120
COOLDOWN_EPOCHS = 6  # epochs 115-120: 3 forward steps + 1-sqrt LR cooldown
MULTI_STEP_FROM = MAX_EPOCHS - COOLDOWN_EPOCHS + 1

ERA5_SELECTION = "aimip_checkpoint_era5"
C96_SELECTION = "aimip_checkpoint_c96"
EVERY_EPOCH = {"start": 0, "step": 1}

SMOKE_SEED = 0
SMOKE_MAX_EPOCHS = 2
SMOKE_COOLDOWN_EPOCHS = 1
SMOKE_SAMPLES_PER_EPOCH = 256
SMOKE_LOG_TRAIN_EVERY_N_BATCHES = 20
SMOKE_INFERENCE_N_FORWARD_STEPS = 100
SMOKE_WEATHER_INFERENCE = "weather_2020_era5"

assert iterations_per_epoch(SAMPLES_PER_EPOCH) == ITERATIONS_PER_EPOCH == 3105


@dataclass(frozen=True)
class Run:
    arch: str
    seed: int
    smoke: bool = False
    arm: str = ARM

    @property
    def stem(self) -> str:
        suffix = "-smoke" if self.smoke else ""
        return f"{self.arch}-{self.arm}-s{self.seed}-maskoff{suffix}"

    @property
    def label(self) -> str:
        return LABELS[self.arm]

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
        return RUN_CONFIGS_DIR / run_config_name(
            self.arch, PARENT_ARM, self.seed, False
        )


def runs(arms: tuple[str, ...] = ARMS) -> list[Run]:
    return [
        Run(arch, seed, arm=arm) for arm in arms for arch in ARCHS for seed in SEEDS
    ]


def smoke_runs() -> list[Run]:
    """GPU smoke tests, C96 AMIP arm only (era5-1979 differs only in data)."""
    return [Run(arch, SMOKE_SEED, smoke=True) for arch in ARCHS]


# --- pieces ----------------------------------------------------------------------


def _stats(root: str, scaling: str) -> dict:
    return {
        "global_means_path": f"{root}/centering.nc",
        "global_stds_path": f"{root}/{scaling}.nc",
    }


def make_normalization(parent_normalization: dict, stats_dir: str) -> dict:
    assert set(parent_normalization) == {"network", "residual", "grouped"}
    assert list(parent_normalization["grouped"]["groups"]) == list(PARENT_LABELS)
    return {
        "network": _stats(stats_dir, "scaling-full-field"),
        "residual": _stats(stats_dir, "scaling-residual"),
    }


def _is_c96_amip_ic1(dataset: dict) -> bool:
    return (
        dataset["data_path"] == C96_AMIP_DATA_PATH
        and dataset["file_pattern"] == C96_AMIP_FILE
    )


def _splits(concat: list[dict]) -> list[tuple[str, str]]:
    return [(d["subset"]["start_time"], d["subset"]["stop_time"]) for d in concat]


def make_train_concat(parent_concat: list[dict], arm: str) -> list[dict]:
    if arm == ARM:
        concat = [copy.deepcopy(d) for d in parent_concat if _is_c96_amip_ic1(d)]
        source = {"data_path": C96_AMIP_DATA_PATH, "file_pattern": C96_AMIP_FILE}
    else:
        concat = [copy.deepcopy(d) for d in parent_concat if d["labels"] == ["era5"]]
        assert _splits(concat) == list(PARENT_ERA5_SPLITS)
        concat[0]["subset"]["start_time"] = TRAIN_SPLITS[0][0]
        source = {"data_path": "/climate-default", "file_pattern": ERA5_FILE}
    assert _splits(concat) == list(TRAIN_SPLITS)
    for dataset in concat:
        assert dataset == {
            **source,
            "engine": "zarr",
            "subset": dataset["subset"],
            "labels": [LABELS[arm]],
        }, dataset
    return concat


def _relabel(dataset: dict, label: str) -> None:
    assert dataset["labels"] in (["era5"], ["c96"]), dataset["labels"]
    dataset["labels"] = [label]


def make_validation(parent_validation: list[dict], arm: str) -> list[dict]:
    validation = copy.deepcopy(parent_validation)
    assert [(e["name"], e["weight"]) for e in validation] == [
        ("val", 1.0),
        ("val_c96", 0.0),
    ]
    era5, c96 = validation
    for entry in validation:
        assert entry["evaluate_all_steps"] is False
        concat = entry["loader"]["dataset"]["concat"]
        assert [d["subset"]["start_time"][:4] for d in concat] == list(VALIDATION_YEARS)
    for dataset in c96["loader"]["dataset"]["concat"]:
        assert _is_c96_amip_ic1(dataset) and dataset["labels"] == ["c96"]
        _relabel(dataset, LABELS[arm])
    for dataset in era5["loader"]["dataset"]["concat"]:
        assert dataset["file_pattern"] == ERA5_FILE
        _relabel(dataset, LABELS[arm])
    # Selection on the arm's own data; the other is logged only.
    era5["weight"] = 1.0 if arm == ERA5_ARM else 0.0
    c96["weight"] = 1.0 if arm == ARM else 0.0
    return validation


def make_inference(parent_inference: list[dict], arm: str) -> list[dict]:
    inference = copy.deepcopy(parent_inference)
    by_name = {entry["name"]: entry for entry in inference}
    assert list(by_name)[0] == ERA5_SELECTION
    for name, entry in by_name.items():
        dataset = entry["loader"]["dataset"]
        assert dataset["data_path"] == "/climate-default", name
        assert dataset["file_pattern"] == ERA5_FILE, name
        assert entry["weight"] == (1.0 if name == ERA5_SELECTION else 0.0), name
        expected_epochs = (
            EVERY_EPOCH if name == ERA5_SELECTION else HEAVY_INFERENCE_EPOCHS
        )
        assert entry["epochs"] == expected_epochs, name
        _relabel(dataset, LABELS[arm])
    era5_selection = by_name[ERA5_SELECTION]
    c96 = copy.deepcopy(era5_selection)
    c96["name"] = C96_SELECTION
    c96["loader"]["dataset"] = {
        "data_path": C96_AMIP_DATA_PATH,
        "file_pattern": C96_AMIP_FILE,
        "engine": "zarr",
        "labels": [LABELS[arm]],
    }
    # Selection on the arm's own data; the other is logged only.
    era5_selection["weight"] = 1.0 if arm == ERA5_ARM else 0.0
    c96["weight"] = 1.0 if arm == ARM else 0.0
    return [c96] + inference


# --- configs ---------------------------------------------------------------------


def make_config(run: Run, stats_dir: str) -> dict:
    assert not run.smoke or run.arm == ARM
    parent = yaml.safe_load(run.parent_config.read_text())
    cfg = copy.deepcopy(parent)
    check_iterations_per_epoch(cfg)
    assert parent["seed"] == run.seed
    assert parent["max_epochs"] == 50
    assert parent["ema"] == {"decay": 0.9999}
    assert parent["validate_using_ema"] is True
    assert parent["ema_checkpoint_save_epochs"] == {"start": 1, "step": 1}
    assert parent["logging"]["project"] == "FM"
    cfg["max_epochs"] = MAX_EPOCHS

    loader = cfg["train_loader"]
    assert loader["batch_size"] // N_GPUS == 2
    loader["dataset"]["concat"] = make_train_concat(
        loader["dataset"]["concat"], run.arm
    )
    assert loader["label_sampling"] == {
        "weights": {label: 0.25 for label in PARENT_LABELS},
        "samples_per_epoch": SAMPLES_PER_EPOCH,
    }
    loader["label_sampling"] = {
        "weights": {run.label: 1.0},
        "samples_per_epoch": SAMPLES_PER_EPOCH,
    }
    cfg["validation"] = make_validation(parent["validation"], run.arm)
    cfg["inference"] = make_inference(parent["inference"], run.arm)

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
    if run.arch == SFNO:
        assert "scheduler" not in optimization
    else:
        assert optimization["scheduler"]["schedulers"][0] == {
            "type": "LinearLR",
            "kwargs": {"start_factor": 0.01, "end_factor": 1.0, "total_iters": 2},
            "step_each_iteration": False,
        }
        assert PARENT_WARMUP_FACTORS == (0.01, 0.505)
    optimization["scheduler"] = lr_scheduler(
        WARMUPS[run.arch], MAX_EPOCHS, COOLDOWN_EPOCHS
    )

    step_config = cfg["stepper"]["step"]["config"]
    builder = step_config["builder"]
    assert builder["type"] == BUILDERS[run.arch]
    assert builder["config"]["embed_dim"] == EMBED_DIMS[run.arch]
    assert builder["conditional"] is True
    assert step_config["include_channel_mask_inputs"] is True
    assert "input_dropout" not in step_config
    assert step_config["global_mean_removal"] == {
        "kind": "shared",
        "append_as_input": True,
    }
    step_config["normalization"] = make_normalization(
        parent["stepper"]["step"]["config"]["normalization"], stats_dir
    )
    if run.smoke:
        _make_smoke(cfg)
    return cfg


def _smoke_inference(inference: list[dict]) -> list[dict]:
    by_name = {entry["name"]: entry for entry in inference}
    entries = []
    for name in (C96_SELECTION, ERA5_SELECTION):
        entry = copy.deepcopy(by_name[name])
        assert entry["epochs"] == EVERY_EPOCH
        assert entry["n_forward_steps"] > SMOKE_INFERENCE_N_FORWARD_STEPS
        entry["n_forward_steps"] = SMOKE_INFERENCE_N_FORWARD_STEPS
        entries.append(entry)
    weather = copy.deepcopy(by_name[SMOKE_WEATHER_INFERENCE])
    assert weather["weight"] == 0.0
    weather["epochs"] = dict(EVERY_EPOCH)
    return entries + [weather]


def _make_smoke(cfg: dict) -> None:
    cfg["max_epochs"] = SMOKE_MAX_EPOCHS
    assert "log_train_every_n_batches" not in cfg
    cfg["log_train_every_n_batches"] = SMOKE_LOG_TRAIN_EVERY_N_BATCHES
    loader = cfg["train_loader"]
    assert loader["time_buffer_pool_size"] * loader["batch_size"] == (
        SMOKE_SAMPLES_PER_EPOCH
    )
    loader["label_sampling"]["samples_per_epoch"] = SMOKE_SAMPLES_PER_EPOCH
    cfg["stepper_training"]["n_forward_steps"]["milestones"][0]["epoch"] = (
        SMOKE_MAX_EPOCHS - SMOKE_COOLDOWN_EPOCHS + 1
    )
    cfg["optimization"]["scheduler"] = lr_scheduler(
        "none",
        SMOKE_MAX_EPOCHS,
        SMOKE_COOLDOWN_EPOCHS,
        iterations_per_epoch=iterations_per_epoch(SMOKE_SAMPLES_PER_EPOCH),
    )
    cfg["inference"] = _smoke_inference(cfg["inference"])


def describe(run: Run) -> str:
    if run.smoke:
        return (
            f"ACE FM {run.arch} C96 AMIP-only smoke test: {SMOKE_MAX_EPOCHS} short "
            "epochs (1-step, then 3-step + cooldown)"
        )
    warmup = ", 2-epoch warmup" if WARMUPS[run.arch] == "parent" else ""
    data, short = (
        ("C96 AMIP ic_0001 only", "C96 AMIP")
        if run.arm == ARM
        else ("ERA5 1979-2008 only", "ERA5 1979")
    )
    return (
        f"ACE FM {run.arch} on {data}: {MAX_EPOCHS} epochs from "
        f"scratch, LR {PARENT_LR:g} constant{warmup}, 3-step + 1-sqrt cooldown "
        f"to 0 over epochs {MULTI_STEP_FROM}-{MAX_EPOCHS}, {short} scalar stats, "
        f"selection on {short.split()[0]}, masking off"
    )


def header(run: Run) -> str:
    lines = [
        "Generated by create_fm_c96amip_jobs.py; do not edit by hand.",
        f"JOB_NAME: {run.job_name}",
        f"WANDB_GROUP: {run.group}",
        f"PARENT_CONFIG: run_configs/{run.parent_config.name}",
        f"WEKA_MOUNT: {ERA5_STORE}",
        "EXPERIMENT_DIR: $AGENT_OUTPUT_DIR (set by submit_fm_c96amip_jobs.py)",
        f"DESCRIPTION: {describe(run)}",
        "See create_fm_c96amip_jobs.py for every change from the parent.",
    ]
    return "".join(f"# {line}\n" for line in lines)


def write_config(run: Run, stats_dir: str) -> None:
    body = yaml.safe_dump(
        make_config(run, stats_dir),
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
    for arm in ARMS:
        parser.add_argument(
            f"--stats-dir-{arm.removeprefix('fm-')}",
            dest=arm,
            default=STATS_DIRS[arm],
            help=f"{arm} scalar stats root (default: {STATS_DIRS[arm]}).",
        )
    args = parser.parse_args()
    stats_dirs = {arm: getattr(args, arm).rstrip("/") for arm in ARMS}
    for stats_dir in stats_dirs.values():
        if not stats_dir.startswith("/climate-default/"):
            parser.error(f"{stats_dir} is not a /climate-default/ path")
    for run in runs() + smoke_runs():
        write_config(run, stats_dirs[run.arm])


if __name__ == "__main__":
    main()

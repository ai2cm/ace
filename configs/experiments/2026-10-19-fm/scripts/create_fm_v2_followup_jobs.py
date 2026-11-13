"""Generate the 2026-11-11 follow-ups of the FM v2 runs (create_fm_v2_jobs.py).

Three phases, each a set of run configs derived from the v2 `-maskoff` run
configs (the "parents"; the mask-on arm is not followed up). Everything not
listed below is copied from the parent verbatim (data, label sampling,
normalization, model, batch size, EMA, validation, inline inference and its
epochs, EMA checkpoints every epoch):

- `lrsweep` (phase A): 3 runs ace2-fm-nc-swin-v2.2-v2-lr{1e-4,3e-4,5e-4}-
  fm-balanced-s0-maskoff, from scratch, parent nc-swin-v2.2-v2-fm-balanced-s0-
  maskoff. max_epochs 30, 1 forward step throughout. LR: the parent's 2-epoch
  warmup (0.01, then 0.505, of the peak), the peak LR, then the 1-sqrt cooldown
  to 0 over epochs 29-30. wandb group ace2-fm-swin-lrsweep-2026-11-11.
- `ext80` (phase B): 24 runs ace2-fm-{nc-sfno-v2,nc-swin-v2.2-v2}-ext80-
  {arm}-s{seed}-maskoff, each initialized from its parent run's
  `training_checkpoints/ema_ckpt_0040.tar` (EMA weights after epoch 40, the
  last single-step epoch) and trained 80 more epochs, to parent-equivalent
  epoch 120. As for the sfno-v3 forks (backup/exp-alexeyfm2-eff922d97,
  create_fm_v3_jobs.py) the checkpoint is mounted at /parent_ckpt.tar by
  run-ace-anneal.sh (`stepper_training.parameter_init.weights_path`), which
  loads stepper weights only: a fresh optimizer, an EMA re-initialised from the
  loaded weights, and the epoch counter at 0, so child epoch e is parent
  epoch 40 + e. max_epochs 80; 1 forward step for child epochs 1-74, 3 (the
  parent's multi-step settings) for 75-80 (parent-equivalent 115-120). LR: a
  500-iteration linear warmup (0.01 -> 1, for the fresh optimizer), 1e-4
  constant, then the 1-sqrt cooldown to 0 over child epochs 75-80. The last
  EMA checkpoint is ema_ckpt_0080.tar. sfno keeps its own settings otherwise
  (its parent has no scheduler). wandb groups ace2-fm-{swin,sfno}-ext80-
  2026-11-11.
- `e120` (phase C, swin only, LR X chosen from phase A): 12 runs
  ace2-fm-nc-swin-v2.2-v2-lr{X}-e120-{arm}-s{seed}-maskoff, from scratch,
  max_epochs 120; 1 forward step for epochs 1-114, 3 for 115-120. LR: the
  parent's 2-epoch warmup, peak X, then the 1-sqrt cooldown to 0 over epochs
  115-120. wandb group ace2-fm-swin-lr{X}-e120-<date>. Written only on request
  (`e120 --lr X`); E120_RUNS lists the X values that are configured (and
  tracked by the inference scripts).

The LR schedule is "constant + cooldown" (Hagele et al. 2024; Subramanian et
al. 2026, arXiv 2603.25687): warmup, constant peak, then
lr(t) = peak * (1 - sqrt((t - t0) / (T - t0))) for t in [t0, T]. fme builds
schedulers from torch.optim.lr_scheduler classes only, so the cooldown is a
SequentialLR of K = 16 LinearLR pieces interpolating 1-sqrt at knots
t0 + (T - t0) * (k/K)^2 (largest deviation 1.6% of the peak, in the first
piece). All pieces step every iteration (`step_each_iteration: true`), and the
milestones are iteration counts: ITERATIONS_PER_EPOCH = 3105 batches per epoch,
measured in wandb (`_step` per epoch) for the v2 runs of both architectures
and every arm, in both their 1-step and 3-step epochs, at batch_size 8,
samples_per_epoch 2758, time_buffer 8 and 4 GPUs; the generator asserts those
values. The warmups are per-iteration pieces too: ConstantLR pieces of one
epoch each reproduce the parent's per-epoch LinearLR warmup exactly.

Each file starts with a `# KEY: value` header (create_anneal_jobs.read_header)
naming the job, its wandb group, its parent and, for ext80, the parent
checkpoint.

Usage:
    python create_fm_v2_followup_jobs.py lrsweep
    python create_fm_v2_followup_jobs.py ext80
    python create_fm_v2_followup_jobs.py e120 --lr 3e-4
"""

import argparse
import copy
import math
import pathlib
from dataclasses import dataclass

import yaml
from create_anneal_jobs import MOUNTED_PARENT_CHECKPOINT
from create_fm_v2_jobs import (
    ARMS,
    CONFIG_PREFIX,
    MULTI_STEP_N_FORWARD_STEPS,
    RUN_CONFIGS_DIR,
    SAMPLES_PER_EPOCH,
    SEEDS,
    SINGLE_STEP_EPOCHS,
    run_config_name,
)

EXP_DIR = RUN_CONFIGS_DIR.parent
JOB_PREFIX = "ace2-fm-"
SWIN = "nc-swin-v2.2-v2"
SFNO = "nc-sfno-v2"
PHASES = ("lrsweep", "ext80", "e120")

# Schedule constants (see the module docstring).
ITERATIONS_PER_EPOCH = 3105
BATCH_SIZE = 8
TIME_BUFFER = 8
COOLDOWN_PIECES = 16
PARENT_WARMUP_FACTORS = (0.01, 0.505)  # epochs 1, 2 of the parent's LinearLR
FORK_WARMUP_ITERATIONS = 500
FORK_WARMUP_START_FACTOR = 0.01
PARENT_LR = 1e-4

# Phase A
LRSWEEP_LRS = ("1e-4", "3e-4", "5e-4")
LRSWEEP_ARM = "fm-balanced"
LRSWEEP_SEED = 0
LRSWEEP_MAX_EPOCHS = 30
LRSWEEP_COOLDOWN_EPOCHS = 2
LRSWEEP_GROUP = "ace2-fm-swin-lrsweep-2026-11-11"

# Phase B
EXT80_ARCHS = (SFNO, SWIN)
FORK_EPOCH = SINGLE_STEP_EPOCHS  # 40
PARENT_CHECKPOINT = f"training_checkpoints/ema_ckpt_{FORK_EPOCH:04d}.tar"
EXT80_MAX_EPOCHS = 80
EXT80_COOLDOWN_EPOCHS = 6
EXT80_GROUPS = {
    SFNO: "ace2-fm-sfno-ext80-2026-11-11",
    SWIN: "ace2-fm-swin-ext80-2026-11-11",
}

# Phase C
E120_MAX_EPOCHS = 120
E120_COOLDOWN_EPOCHS = 6
# Chosen peak LRs X (as in the job name) -> wandb group date. Empty until X is
# chosen from phase A; the inference scripts track the e120 runs of these X
# only.
E120_RUNS: dict[str, str] = {}
E120_DEFAULT_DATE = "2026-11-11"


@dataclass(frozen=True)
class Run:
    phase: str
    arch: str  # parent arch tag
    variant: str  # token after the arch tag: ext80, lr3e-4, lr3e-4-e120
    arm: str
    seed: int
    lr: str
    group: str

    @property
    def tag(self) -> str:
        return f"{self.arch}-{self.variant}"

    @property
    def config_name(self) -> str:
        return f"{CONFIG_PREFIX}{self.tag}-{self.arm}-s{self.seed}-maskoff.yaml"

    @property
    def job_name(self) -> str:
        # Same as submit_fm_jobs.config_to_job_name(self.config_name).
        return f"{JOB_PREFIX}{self.tag}-{self.arm}-s{self.seed}-maskoff"

    @property
    def path(self) -> pathlib.Path:
        return RUN_CONFIGS_DIR / self.config_name

    @property
    def parent_config(self) -> pathlib.Path:
        return RUN_CONFIGS_DIR / run_config_name(self.arch, self.arm, self.seed, False)

    @property
    def parent_job(self) -> str:
        return f"{JOB_PREFIX}{self.parent_config.stem.removeprefix(CONFIG_PREFIX)}"

    @property
    def max_epochs(self) -> int:
        return {
            "lrsweep": LRSWEEP_MAX_EPOCHS,
            "ext80": EXT80_MAX_EPOCHS,
            "e120": E120_MAX_EPOCHS,
        }[self.phase]

    @property
    def epoch_offset(self) -> int:
        """Parent epochs before the child's epoch 1 (forks only)."""
        return FORK_EPOCH if self.phase == "ext80" else 0


def lr_value(lr: str) -> float:
    """The float of an LR token such as "3e-4", which must be its canonical
    spelling (lr_token), so job names are unique per LR.
    """
    value = float(lr)
    if lr_token(value) != lr:
        raise ValueError(f"LR {lr!r} is not spelled {lr_token(value)!r}")
    return value


def lr_token(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    if float(f"{mantissa}e{exponent}") != value:
        raise ValueError(f"LR {value} has more than one significant digit")
    return f"{mantissa}e{int(exponent)}"


def lrsweep_runs() -> list[Run]:
    return [
        Run("lrsweep", SWIN, f"lr{lr}", LRSWEEP_ARM, LRSWEEP_SEED, lr, LRSWEEP_GROUP)
        for lr in LRSWEEP_LRS
    ]


def ext80_runs(archs=EXT80_ARCHS) -> list[Run]:
    return [
        Run("ext80", arch, "ext80", arm, seed, lr_token(PARENT_LR), EXT80_GROUPS[arch])
        for arch in archs
        for arm in ARMS
        for seed in SEEDS
    ]


def e120_group(lr: str, date: str) -> str:
    return f"ace2-fm-swin-lr{lr}-e120-{date}"


def e120_runs(lr: str, date: str | None = None) -> list[Run]:
    lr_value(lr)
    date = date or E120_RUNS.get(lr, E120_DEFAULT_DATE)
    return [
        Run("e120", SWIN, f"lr{lr}-e120", arm, seed, lr, e120_group(lr, date))
        for arm in ARMS
        for seed in SEEDS
    ]


def tracked_runs() -> list[Run]:
    """Every configured follow-up run: phases A and B, and C for the X in
    E120_RUNS.
    """
    runs = lrsweep_runs() + ext80_runs()
    for lr in E120_RUNS:
        runs += e120_runs(lr)
    return runs


# --- LR schedule ---------------------------------------------------------------


def _piece(type_: str, **kwargs) -> dict:
    return {"type": type_, "kwargs": kwargs, "step_each_iteration": True}


def cooldown_factor(t: int, n: int) -> float:
    return 1.0 - math.sqrt(t / n)


def lr_scheduler(warmup: str, max_epochs: int, cooldown_epochs: int) -> dict:
    """SequentialLR pieces (each with its length in iterations): `warmup`
    ("parent": the parent's 2-epoch per-epoch warmup; "fork": a
    FORK_WARMUP_ITERATIONS linear warmup), constant 1, then the piecewise-linear
    1-sqrt cooldown to 0 over the last `cooldown_epochs` epochs.
    """
    pieces: list[tuple[dict, int]] = []
    if warmup == "parent":
        for factor in PARENT_WARMUP_FACTORS:
            n = ITERATIONS_PER_EPOCH
            pieces.append((_piece("ConstantLR", factor=factor, total_iters=n), n))
    elif warmup == "fork":
        n = FORK_WARMUP_ITERATIONS
        pieces.append(
            (
                _piece(
                    "LinearLR",
                    start_factor=FORK_WARMUP_START_FACTOR,
                    end_factor=1.0,
                    total_iters=n,
                ),
                n,
            )
        )
    else:
        raise ValueError(warmup)
    total = max_epochs * ITERATIONS_PER_EPOCH
    cooldown = cooldown_epochs * ITERATIONS_PER_EPOCH
    n_constant = total - cooldown - sum(n for _, n in pieces)
    assert n_constant > 0
    pieces.append(
        (_piece("ConstantLR", factor=1.0, total_iters=n_constant), n_constant)
    )
    knots = [
        round(cooldown * (k / COOLDOWN_PIECES) ** 2) for k in range(COOLDOWN_PIECES + 1)
    ]
    for a, b in zip(knots, knots[1:]):
        assert b > a
        pieces.append(
            (
                _piece(
                    "LinearLR",
                    start_factor=round(cooldown_factor(a, cooldown), 8),
                    end_factor=round(cooldown_factor(b, cooldown), 8),
                    total_iters=b - a,
                ),
                b - a,
            )
        )
    assert sum(n for _, n in pieces) == total
    milestones: list[int] = []
    for _, n in pieces[:-1]:
        milestones.append((milestones[-1] if milestones else 0) + n)
    return {"schedulers": [p for p, _ in pieces], "milestones": milestones}


# --- configs ---------------------------------------------------------------------


def check_iterations_per_epoch(cfg: dict) -> None:
    """The schedule's milestones assume ITERATIONS_PER_EPOCH; check the loader
    settings it was measured at.
    """
    loader = cfg["train_loader"]
    assert loader["batch_size"] == BATCH_SIZE
    assert loader["time_buffer"] == TIME_BUFFER
    assert loader["label_sampling"]["samples_per_epoch"] == SAMPLES_PER_EPOCH


def set_steps(cfg: dict, multi_step_from: int | None) -> None:
    """1 forward step, then MULTI_STEP_N_FORWARD_STEPS from epoch
    `multi_step_from` (None: 1 step throughout).
    """
    training = cfg["stepper_training"]
    assert training["n_forward_steps"] == {
        "start_value": 1,
        "milestones": [
            {"epoch": SINGLE_STEP_EPOCHS + 1, "value": MULTI_STEP_N_FORWARD_STEPS}
        ],
    }
    assert training["optimize_last_step_only"] is False
    if multi_step_from is None:
        training["n_forward_steps"] = 1
    else:
        training["n_forward_steps"]["milestones"][0]["epoch"] = multi_step_from


def make_config(run: Run) -> dict:
    parent = yaml.safe_load(run.parent_config.read_text())
    cfg = copy.deepcopy(parent)
    check_iterations_per_epoch(cfg)
    assert parent["max_epochs"] == 50
    assert parent["ema_checkpoint_save_epochs"] == {"start": 1, "step": 1}
    assert "evaluate_before_training" not in parent
    assert "input_dropout" not in cfg["stepper"]["step"]["config"]
    cfg["max_epochs"] = run.max_epochs

    optimization = cfg["optimization"]
    assert optimization["lr"] == PARENT_LR
    parent_scheduler = optimization.get("scheduler")
    assert (parent_scheduler is not None) == (run.arch == SWIN)
    if run.phase == "ext80":
        warmup = "fork"
        cooldown_epochs = EXT80_COOLDOWN_EPOCHS
        training = cfg["stepper_training"]
        assert "parameter_init" not in training
        training["parameter_init"] = {"weights_path": MOUNTED_PARENT_CHECKPOINT}
    else:
        assert run.arch == SWIN
        assert parent_scheduler["schedulers"][0]["kwargs"] == {
            "start_factor": PARENT_WARMUP_FACTORS[0],
            "end_factor": 1.0,
            "total_iters": len(PARENT_WARMUP_FACTORS),
        }
        warmup = "parent"
        cooldown_epochs = (
            LRSWEEP_COOLDOWN_EPOCHS if run.phase == "lrsweep" else E120_COOLDOWN_EPOCHS
        )
    optimization["lr"] = lr_value(run.lr)
    optimization["scheduler"] = lr_scheduler(warmup, run.max_epochs, cooldown_epochs)
    multi_step_from = (
        None if run.phase == "lrsweep" else run.max_epochs - cooldown_epochs + 1
    )
    set_steps(cfg, multi_step_from)
    return cfg


def describe(run: Run) -> str:
    if run.phase == "ext80":
        return (
            f"ACE FM {run.tag}: 80 epochs from {run.parent_job} "
            f"ema_ckpt_{FORK_EPOCH:04d} (parent-equivalent 41-120), "
            f"{FORK_WARMUP_ITERATIONS}-iteration warmup, LR {run.lr} constant, "
            "3-step + 1-sqrt cooldown to 0 over epochs 75-80, masking off"
        )
    steps = (
        "1 step throughout"
        if run.phase == "lrsweep"
        else f"3-step from epoch {run.max_epochs - E120_COOLDOWN_EPOCHS + 1}"
    )
    return (
        f"ACE FM {run.tag}: {run.max_epochs} epochs from scratch, 2-epoch "
        f"warmup, LR {run.lr} constant, 1-sqrt cooldown to 0, {steps}, "
        "masking off"
    )


def header(run: Run) -> str:
    lines = [
        "Generated by create_fm_v2_followup_jobs.py; do not edit by hand.",
        f"JOB_NAME: {run.job_name}",
        f"WANDB_GROUP: {run.group}",
        f"PARENT_CONFIG: run_configs/{run.parent_config.name}",
    ]
    if run.phase == "ext80":
        lines += [
            f"PARENT_JOB: {run.parent_job}",
            f"PARENT_CHECKPOINT: {PARENT_CHECKPOINT} -> {MOUNTED_PARENT_CHECKPOINT}",
        ]
    lines += [
        f"DESCRIPTION: {describe(run)}",
        "See create_fm_v2_followup_jobs.py for every change from the parent.",
    ]
    return "".join(f"# {line}\n" for line in lines)


def write_config(run: Run) -> None:
    body = yaml.safe_dump(
        make_config(run), sort_keys=False, default_flow_style=False, width=1000
    )
    run.path.write_text(header(run) + body)
    print(f"Wrote {run.path.relative_to(EXP_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--lr", help="e120 only: the peak LR X, e.g. 3e-4.")
    parser.add_argument(
        "--date",
        help=(
            "e120 only: date in the wandb group (default: E120_RUNS[lr], else "
            f"{E120_DEFAULT_DATE})."
        ),
    )
    args = parser.parse_args()
    if (args.phase == "e120") != (args.lr is not None):
        parser.error("--lr is required for e120 and only allowed there")
    if args.phase == "lrsweep":
        runs = lrsweep_runs()
    elif args.phase == "ext80":
        runs = ext80_runs()
    else:
        runs = e120_runs(args.lr, args.date)
    for run in runs:
        write_config(run)


if __name__ == "__main__":
    main()

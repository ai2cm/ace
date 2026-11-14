"""Generate the 2026-11-12 fine-tuning ("ft") sweep of the finished ext80 runs
(create_fm_v2_followup_jobs.py, phase B).

Parents: the 6 ext80 runs ace2-fm-{nc-sfno-v2,nc-swin-v2.2-v2}-ext80-
fm-spencer-balanced-s{0,1,2}-maskoff. Each parent gets 4 ERA5 label mixes x 4
initial LRs = 16 runs, 96 in all, named

    <parent job>-ft-era5{25,50,70,100}-lr{1e-4,3e-5,1e-5,3e-6}

in wandb groups ace2-fm-{sfno,swin}-ft-2026-11-12. The configs are written to
run_configs/ft/<parent config stem>-ft-era5{mix}-lr{lr}.yaml. They are kept in
a subdirectory, as the anneal configs are, because everything that globs
run_configs/ (the ext80 inference families, update_beaker_map.py) selects by
the prefix `ace-train-config-4deg-<arch>-ext80-fm-`, which the ft names share.

Changes from the parent ext80 config (everything else is copied unchanged:
data, normalization, model, masking off, batch size, optimizer and weight
decay, validation, the inline inference entries and their epochs, seed):

- Initial weights: `stepper_training.parameter_init.weights_path:
  /parent_ckpt.tar` is the parent's line, unchanged, but the file mounted
  there by run-ace-anneal.sh is now the ext80 parent's own last EMA checkpoint
  `training_checkpoints/ema_ckpt_0080.tar` (the parent mounted its v2 parent's
  ema_ckpt_0040.tar). As for ext80, parameter init loads stepper weights only:
  a fresh optimizer, an EMA re-initialised from the loaded weights, and the
  epoch counter at 0.
- `max_epochs: 8` (parent 80).
- `train_loader.label_sampling.samples_per_epoch: 688` (parent 2758), a
  quarter of the parent's epoch. fme rounds samples_per_epoch up to a multiple
  of the 4 data-parallel ranks (label_sampling.LabelBalancedSampler), each
  rank batches its samples by batch_size / 4 = 2 with drop_last, and with
  time_buffer 8 every such batch of windows yields 9 training batches:
  iterations/epoch = floor(ceil(samples_per_epoch / 4) / 2) * 9. For the
  parent that is ceil(2758 / 4) = 690 per rank, 345 * 9 = 3105
  (ITERATIONS_PER_EPOCH, as measured in wandb). 2758 / 4 = 689.5 is not an
  integer, and 689 and 690 both round up to 173 per rank, an odd count whose
  last window drop_last discards; 688 = 4 * 172 is the largest value with
  nothing rounded or dropped, and gives the same count as 689..692:
  FT_ITERATIONS_PER_EPOCH = 86 * 9 = 774 (parent / 4 = 776.25), so
  8 * 774 = 6192 iterations in all. `iterations_per_epoch` computes this and
  the generator asserts it reproduces the parent's measured 3105.
- `train_loader.label_sampling.weights`: ERA5 at 0.25, 0.50, 0.70 or 1.0
  (`era5{25,50,70,100}`), the other labels (c96, cm4, xshield) sharing the
  remainder in proportion to their parent weights, rounded to 6 decimals (fme
  normalizes the weights): 0.25 each (the control, asserted identical to the
  parent's weights), 0.166667 each, 0.1 each, 0 each. A weight of 0 is
  accepted and that label is not drawn (LabelSamplingConfig; the weights must
  still name every label of the dataset), as for arm a3 of
  create_anneal_jobs.py.
- `stepper_training.n_forward_steps: 3` for every epoch (parent: 1, then 3
  from epoch 75). The parent's other multi-step settings are epoch-independent
  and unchanged: `optimize_last_step_only: false` (loss on every step),
  n_ensemble 2, the EnsembleLoss.
- `optimization.lr`: 1e-4, 3e-5, 1e-5 or 3e-6 (`lr{...}`; parent 1e-4).
- `optimization.scheduler`: a single LinearLR from factor 1.0 to 0.0 over
  total_iters = 6192 with `step_each_iteration: true`, no warmup (parent: a
  500-iteration warmup, constant, 1-sqrt cooldown). The trainer steps the
  scheduler once after every training batch (fme/core/generics/trainer.py),
  so 0-based iteration i trains at lr * (1 - i / 6192): the first at lr, the
  last (i = 6191) at lr / 6192, and the factor is 0 only after the last
  optimizer step.
- `ema.decay: 0.999` (parent 0.9999).
- `ema_checkpoint_save_epochs: {start: 1, step: 1}` is the parent's value,
  kept: ema_ckpt_0001.tar .. ema_ckpt_0008.tar.

The inline inference entries are the parent's, so only `aimip_checkpoint_era5`
(every epoch) runs: the others have `epochs: {start: 9, step: 10}` and an
8-epoch run never reaches them.

Long fine-tunes (`ft_long_runs`, 36 runs): the same recipe run for 32 and for
128 epochs, to find where performance turns from improving to degrading. Each
parent gets ERA5 mixes 25, 70 and 100 at LR 1e-4 only, at both lengths, named

    <parent job>-ft-era5{25,70,100}-lr1e-4-e{32,128}

i.e. the 8-epoch run's name plus `-e{max_epochs}`, in the same wandb groups.
Their configs are in run_configs/ft-e32/ and run_configs/ft-e128/ (the
8-epoch config's file name plus the same suffix), one directory per length
because the inference tooling tells archs sharing a name token apart by
directory and has one last EMA epoch per arch (_inference_common
._add_ft_archs). Each is identical to its 8-epoch sibling except:

- `max_epochs`: 32 or 128.
- The LinearLR `total_iters`: max_epochs * 774 = 24768 or 99072, so the LR
  still reaches 0 exactly at the end.
- `ema_checkpoint_save_epochs`: {start: 2, step: 2} for 32 epochs and
  {start: 8, step: 8} for 128, i.e. 16 EMA checkpoints per run instead of one
  per epoch. The trainer saves ema_ckpt_{N:04d}.tar after the N-th epoch
  (N = epochs trained, 1-based) when N is in
  range(max_epochs + 1)[start::step] (trainer.epoch_checkpoint_enabled):
  ema_ckpt_0002, 0004, .. 0032, and ema_ckpt_0008, 0016, .. 0128, the last
  epoch included.

The parent's other inline inference entries (`epochs: {start: 9, step: 10}`:
the 10- and 46-year ERA5 runs and the 5-day weather ensembles) are unchanged
and do fire in these runs, after epochs 10, 20, 30, .. (3 times in 32 epochs,
12 in 128).

`-e128` is four hex digits, which _beaker_listing.canonical_name would take
for gantry's collision suffix and strip, making the run the namesake of its
8-epoch sibling; canonical_name keeps it (NOT_HASH_SUFFIXES), and the
generator asserts every job name is its own canonical name.

Each file starts with a `# KEY: value` header (create_anneal_jobs.read_header)
naming the job, its wandb group, its parent job and the parent checkpoint;
submit_fm_v2_finetune_jobs.py reads it.

Usage:
    python create_fm_v2_finetune_jobs.py
"""

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import yaml
from _beaker_listing import canonical_name
from create_anneal_jobs import MOUNTED_PARENT_CHECKPOINT
from create_fm_v2_followup_jobs import (
    BATCH_SIZE,
    EXP_DIR,
    EXT80_ARCHS,
    EXT80_COOLDOWN_EPOCHS,
    EXT80_MAX_EPOCHS,
    ITERATIONS_PER_EPOCH,
    PARENT_LR,
    SFNO,
    SWIN,
    TIME_BUFFER,
    Run,
    check_iterations_per_epoch,
    ext80_runs,
    lr_value,
)
from create_fm_v2_jobs import (
    MULTI_STEP_N_FORWARD_STEPS,
    RUN_CONFIGS_DIR,
    SAMPLES_PER_EPOCH,
    SEEDS,
)

FT_DIR = RUN_CONFIGS_DIR / "ft"
PARENT_ARM = "fm-spencer-balanced"
PARENT_CHECKPOINT = f"training_checkpoints/ema_ckpt_{EXT80_MAX_EPOCHS:04d}.tar"
PARENT_EMA_DECAY = 0.9999
FT_GROUPS = {
    SFNO: "ace2-fm-sfno-ft-2026-11-12",
    SWIN: "ace2-fm-swin-ft-2026-11-12",
}

N_GPUS = 4  # data-parallel ranks; submit_fm_v2_finetune_jobs.py passes it on
MAX_EPOCHS = 8
FT_SAMPLES_PER_EPOCH = 688
FT_ITERATIONS_PER_EPOCH = 774
TOTAL_ITERATIONS = MAX_EPOCHS * FT_ITERATIONS_PER_EPOCH  # 6192
EMA_DECAY = 0.999
EMA_CHECKPOINT_SAVE_EPOCHS = {"start": 1, "step": 1}

ERA5 = "era5"
# Job-name token -> ERA5 label weight. "25" is the parent's mix (the control).
MIXES = {"25": 0.25, "50": 0.50, "70": 0.70, "100": 1.0}
CONTROL_MIX = "25"
LRS = ("1e-4", "3e-5", "1e-5", "3e-6")
WEIGHT_DECIMALS = 6

# The long fine-tunes: the MAX_EPOCHS recipe at these lengths, mixes and LRs.
LONG_MAX_EPOCHS = (32, 128)
LONG_MIXES = ("25", "70", "100")
LONG_LRS = ("1e-4",)
LONG_EMA_CHECKPOINTS = 16  # per run, evenly spaced, the last epoch included


def ft_dir(max_epochs: int) -> Path:
    """Directory of the configs of the runs of `max_epochs` epochs."""
    if max_epochs == MAX_EPOCHS:
        return FT_DIR
    return RUN_CONFIGS_DIR / f"{FT_DIR.name}-e{max_epochs}"


def ema_checkpoint_save_epochs(max_epochs: int) -> dict[str, int]:
    """Every epoch for MAX_EPOCHS, else LONG_EMA_CHECKPOINTS evenly spaced
    epochs ending at the last one.
    """
    if max_epochs == MAX_EPOCHS:
        return EMA_CHECKPOINT_SAVE_EPOCHS
    assert max_epochs % LONG_EMA_CHECKPOINTS == 0
    step = max_epochs // LONG_EMA_CHECKPOINTS
    return {"start": step, "step": step}


def ema_checkpoint_epochs(max_epochs: int) -> list[int]:
    """Epochs N whose ema_ckpt_{N:04d}.tar a run of `max_epochs` saves, as
    fme's trainer.epoch_checkpoint_enabled selects them.
    """
    save = ema_checkpoint_save_epochs(max_epochs)
    return list(range(max_epochs + 1)[save["start"] :: save["step"]])


@dataclass(frozen=True)
class FtRun:
    parent: Run  # the ext80 run fine-tuned
    mix: str  # key of MIXES
    lr: str  # one of LRS
    max_epochs: int = MAX_EPOCHS  # or one of LONG_MAX_EPOCHS

    @property
    def suffix(self) -> str:
        epochs = "" if self.max_epochs == MAX_EPOCHS else f"-e{self.max_epochs}"
        return f"-ft-era5{self.mix}-lr{self.lr}{epochs}"

    @property
    def arch(self) -> str:
        return self.parent.arch

    @property
    def seed(self) -> int:
        return self.parent.seed

    @property
    def job_name(self) -> str:
        return f"{self.parent.job_name}{self.suffix}"

    @property
    def config_name(self) -> str:
        return f"{self.parent.path.stem}{self.suffix}.yaml"

    @property
    def path(self) -> Path:
        return ft_dir(self.max_epochs) / self.config_name

    @property
    def group(self) -> str:
        return FT_GROUPS[self.arch]

    @property
    def total_iterations(self) -> int:
        return self.max_epochs * FT_ITERATIONS_PER_EPOCH


def _parents(archs) -> list[Run]:
    parents = [run for run in ext80_runs(archs) if run.arm == PARENT_ARM]
    assert len(parents) == len(archs) * len(SEEDS)
    return parents


def ft_runs(archs=EXT80_ARCHS) -> list[FtRun]:
    """The 96 8-epoch runs (48 per arch), ordered by arch, seed, mix, LR."""
    return [
        FtRun(parent, mix, lr)
        for parent in _parents(archs)
        for mix in MIXES
        for lr in LRS
    ]


def ft_long_runs(archs=EXT80_ARCHS) -> list[FtRun]:
    """The 36 long runs (18 per arch, 9 per arch and length), ordered by
    arch, max_epochs, seed, mix, LR.
    """
    return [
        FtRun(parent, mix, lr, max_epochs)
        for arch in archs
        for max_epochs in LONG_MAX_EPOCHS
        for parent in _parents((arch,))
        for mix in LONG_MIXES
        for lr in LONG_LRS
    ]


def iterations_per_epoch(samples_per_epoch: int) -> int:
    """Optimizer iterations per epoch at BATCH_SIZE, TIME_BUFFER and N_GPUS
    (see the module docstring).
    """
    per_rank = math.ceil(samples_per_epoch / N_GPUS)
    assert BATCH_SIZE % N_GPUS == 0
    return per_rank // (BATCH_SIZE // N_GPUS) * (TIME_BUFFER + 1)


def mix_weights(parent_weights: dict[str, float], era5: float) -> dict[str, float]:
    """`parent_weights` with ERA5 at `era5` and the other labels sharing the
    remainder in proportion to their parent weights, in the parent's order.
    """
    others = sum(w for label, w in parent_weights.items() if label != ERA5)
    assert others > 0
    weights = {
        label: round(
            era5 if label == ERA5 else w * (1.0 - era5) / others, WEIGHT_DECIMALS
        )
        for label, w in parent_weights.items()
    }
    assert abs(sum(weights.values()) - 1.0) < 10.0**-WEIGHT_DECIMALS * len(weights)
    return weights


def lr_scheduler(total_iterations: int) -> dict:
    return {
        "type": "LinearLR",
        "kwargs": {
            "start_factor": 1.0,
            "end_factor": 0.0,
            "total_iters": total_iterations,
        },
        "step_each_iteration": True,
    }


def make_config(run: FtRun) -> dict:
    parent = yaml.safe_load(run.parent.path.read_text())
    cfg = copy.deepcopy(parent)
    assert "resume_results" not in parent
    assert parent["seed"] == run.seed
    assert parent["max_epochs"] == EXT80_MAX_EPOCHS
    cfg["max_epochs"] = run.max_epochs

    assert parent["ema"] == {"decay": PARENT_EMA_DECAY}
    cfg["ema"]["decay"] = EMA_DECAY
    assert parent["validate_using_ema"] is True
    assert parent["ema_checkpoint_save_epochs"] == EMA_CHECKPOINT_SAVE_EPOCHS
    cfg["ema_checkpoint_save_epochs"] = ema_checkpoint_save_epochs(run.max_epochs)
    # The last epoch's EMA checkpoint is the one the inference tooling calls
    # lastema.
    assert ema_checkpoint_epochs(run.max_epochs)[-1] == run.max_epochs

    # The parent's iteration count is the measured one, under the same formula.
    check_iterations_per_epoch(parent)
    assert iterations_per_epoch(SAMPLES_PER_EPOCH) == ITERATIONS_PER_EPOCH
    assert iterations_per_epoch(FT_SAMPLES_PER_EPOCH) == FT_ITERATIONS_PER_EPOCH
    assert FT_SAMPLES_PER_EPOCH % BATCH_SIZE == 0  # nothing rounded or dropped
    label_sampling = cfg["train_loader"]["label_sampling"]
    label_sampling["samples_per_epoch"] = FT_SAMPLES_PER_EPOCH
    parent_weights = parent["train_loader"]["label_sampling"]["weights"]
    assert parent_weights[ERA5] == MIXES[CONTROL_MIX]
    assert sum(parent_weights.values()) == 1.0
    label_sampling["weights"] = mix_weights(parent_weights, MIXES[run.mix])
    if run.mix == CONTROL_MIX:
        assert label_sampling["weights"] == parent_weights

    training = cfg["stepper_training"]
    assert training["parameter_init"] == {"weights_path": MOUNTED_PARENT_CHECKPOINT}
    assert training["n_forward_steps"] == {
        "start_value": 1,
        "milestones": [
            {
                "epoch": EXT80_MAX_EPOCHS - EXT80_COOLDOWN_EPOCHS + 1,
                "value": MULTI_STEP_N_FORWARD_STEPS,
            }
        ],
    }
    assert training["optimize_last_step_only"] is False
    training["n_forward_steps"] = MULTI_STEP_N_FORWARD_STEPS

    optimization = cfg["optimization"]
    assert optimization["lr"] == PARENT_LR
    assert all(
        piece["step_each_iteration"] is True
        for piece in optimization["scheduler"]["schedulers"]
    )
    optimization["lr"] = lr_value(run.lr)
    optimization["scheduler"] = lr_scheduler(run.total_iterations)
    return cfg


def describe(run: FtRun) -> str:
    return (
        f"ACE FM {run.parent.tag} ft: {run.max_epochs} epochs of "
        f"{FT_ITERATIONS_PER_EPOCH} iterations from {run.parent.job_name} "
        f"ema_ckpt_{EXT80_MAX_EPOCHS:04d}, era5 label weight {MIXES[run.mix]:g}, "
        f"LR {run.lr} linear to 0, {MULTI_STEP_N_FORWARD_STEPS}-step, EMA decay "
        f"{EMA_DECAY:g}, masking off"
    )


def header(run: FtRun) -> str:
    lines = [
        "Generated by create_fm_v2_finetune_jobs.py; do not edit by hand.",
        f"JOB_NAME: {run.job_name}",
        f"WANDB_GROUP: {run.group}",
        f"PARENT_CONFIG: run_configs/{run.parent.path.name}",
        f"PARENT_JOB: {run.parent.job_name}",
        f"PARENT_CHECKPOINT: {PARENT_CHECKPOINT} -> {MOUNTED_PARENT_CHECKPOINT}",
        f"DESCRIPTION: {describe(run)}",
        "See create_fm_v2_finetune_jobs.py for every change from the parent.",
    ]
    return "".join(f"# {line}\n" for line in lines)


def write_config(run: FtRun) -> None:
    body = yaml.safe_dump(
        make_config(run), sort_keys=False, default_flow_style=False, width=1000
    )
    run.path.write_text(header(run) + body)
    print(f"Wrote {run.path.relative_to(EXP_DIR)}")


def main() -> None:
    runs = ft_runs() + ft_long_runs()
    assert len({run.job_name for run in runs}) == len(runs)
    # No name ends in what Beaker listings would strip as a collision suffix.
    assert all(canonical_name(run.job_name) == run.job_name for run in runs)
    assert TOTAL_ITERATIONS == ft_runs()[0].total_iterations
    for run in runs:
        run.path.parent.mkdir(exist_ok=True)
        write_config(run)
    for directory in dict.fromkeys(run.path.parent for run in runs):
        count = sum(run.path.parent == directory for run in runs)
        print(f"{count} configs in {directory.relative_to(EXP_DIR)}")


if __name__ == "__main__":
    main()

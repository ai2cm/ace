"""Submit SST-perturbation inference jobs for every saved epoch of the ERA5
fine-tunes.

The best-inference sweep (submit_sst_jobs.py) runs one checkpoint per training
run. This one runs the *trajectory*: each of the ten `nc-sfno` ERA5 fine-tunes
at every epoch it saved, on both forcing grids and all three perturbation
levels, so the climate sensitivity can be read as a function of fine-tuning
epoch rather than at a single endpoint.

    10 runs x 2 grids x 3 levels x 11 epochs = 660 jobs

**Epoch 0 comes from a different dataset.** `evaluate_before_training` measures
the pre-fine-tune state but saves no checkpoint, so the epoch-0 weights live in
the *source* run's result dataset as its `ckpt.tar` -- the same file
`parameter_init` loaded. Epochs 1-10 are the fine-tune's own
`ema_ckpt_XXXX.tar`; EMA rather than raw because `validate_using_ema` is set, so
EMA is the weight set every other metric reads. Note epoch 0 is *not* the
best-inference sweep's checkpoint for the same cell: that sweep uses
`best_inference_ckpt.tar`, which is a different selection.

**The label is a property of the checkpoint, not the grid.** A c96-pretrained
fine-tune has vocabulary {amip, ramped, som} and was trained on ERA5 data
labeled `amip`, so on the ERA5 grid it must be sent `amip`. An fm-pretrained
fine-tune has `era5` in its vocabulary and is sent `era5`. Both are sent `amip`
on the C96 grid. See ERA5_GRID_LABEL_BY_REGIME and generate_sst_configs's
GRID_LABELS; sending the wrong one fails loudly for the grouped cells, silently
for the conditional ones.

Gantry clones the repository at HEAD, so the configs must be committed and
pushed before submitting; this is checked unless --dry-run is given.

Usage:
    python submit_sst_epoch_jobs.py [--dry-run] [--run RUN [RUN ...]]
                                    [--regime {fm,c96}]
                                    [--epoch N [N ...]]
                                    [--perturbation {p0k,p2k,p4k} ...]
                                    [--forcing-grid {era5,c96} ...]
                                    [--skip-if-in-beaker]
                                    [--beaker-workspace WORKSPACE]
                                    [--beaker-cluster CLUSTER [CLUSTER ...]]
                                    [--beaker-priority PRIORITY]
"""

import argparse
import pathlib

from _submit_common import (
    add_beaker_args,
    check_configs_at_head,
    drop_jobs_in_beaker,
    submit_job,
)
from generate_eval_configs import TRAINING_RESULT_DATASETS, WANDB_PROJECT
from generate_norm_ablation_finetune_configs import (
    C96_ERA5_ALIAS,
    DEFAULT_EPOCHS,
    FINETUNE_SUFFIX,
    REGIMES,
    config_to_run_name,
    source_cells,
    source_config_name,
)
from generate_sst_configs import (
    DATASETS,
    RUN_CONFIGS_DIR,
    SST_PERTURBATIONS,
    sst_epoch_config_filename,
)
from submit_sst_jobs import validate_configs

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIRNAME = RUN_CONFIGS_DIR.name
RUN_SCRIPT = HERE / "run-ace-inference.sh"

#: A group of its own, so the existing SST notebooks' group query keeps
#: returning only the best-inference sweep's ~150 runs rather than these 660.
WANDB_GROUP = "ace2-fm-sst-perts-finetune-2026-06-26"

#: Epoch 0 is the pre-fine-tune state, loaded from the source run rather than
#: the fine-tune; see the module docstring.
SOURCE_EPOCH = 0
EPOCHS = tuple(range(SOURCE_EPOCH, DEFAULT_EPOCHS + 1))

#: Checkpoint inside the *source* run's result dataset holding the epoch-0
#: weights: the last-epoch checkpoint that parameter_init warm-started from.
SOURCE_CHECKPOINT_PATH = "training_checkpoints/ckpt.tar"

#: The label to send on the ERA5 grid, by the fine-tune's pretraining regime.
#: `fm` cells hold `era5` in their vocabulary; `c96` cells do not, and saw ERA5
#: under C96_ERA5_ALIAS during fine-tuning.
ERA5_GRID_LABEL_BY_REGIME = {
    "fm": "era5",
    "c96": C96_ERA5_ALIAS,
}


def _finetune_run_name(regime: str, arm: str, conditional: bool) -> str:
    stem = pathlib.Path(source_config_name(regime, arm, conditional)).stem
    return config_to_run_name(f"{stem}{FINETUNE_SUFFIX}.yaml")


def finetune_runs() -> dict[str, tuple[str, str]]:
    """Fine-tune run name -> (pretraining regime, source run name).

    Built from the fine-tune generator's own cell list rather than by parsing
    run names, so a cell added or dropped there shows up here without edits.
    """
    runs: dict[str, tuple[str, str]] = {}
    for regime in REGIMES:
        for arm, conditional in source_cells(regime):
            source_run = config_to_run_name(
                source_config_name(regime, arm, conditional)
            )
            runs[_finetune_run_name(regime, arm, conditional)] = (regime, source_run)
    return runs


def grid_label(grid: str, regime: str) -> str:
    if grid == "era5":
        return ERA5_GRID_LABEL_BY_REGIME[regime]
    return DATASETS[grid].native_label


def epoch_checkpoint(epoch: int, finetune_run: str, source_run: str) -> tuple[str, str]:
    """(Beaker dataset ID, path within it) for one epoch's weights."""
    if epoch == SOURCE_EPOCH:
        return TRAINING_RESULT_DATASETS[source_run], SOURCE_CHECKPOINT_PATH
    return (
        TRAINING_RESULT_DATASETS[finetune_run],
        f"training_checkpoints/ema_ckpt_{epoch:04d}.tar",
    )


def sst_epoch_job_name(run_name: str, grid: str, level: str, epoch: int) -> str:
    """Job name, with the epoch as a zero-padded trailing segment so the
    trajectory sorts. The label is deliberately absent: it is implied by the
    run, and leaving it out keeps these names comparable to the best-inference
    sweep's `{run}-sst-{grid}-{level}`.
    """
    return f"{run_name}-sst-{grid}-{level}-e{epoch:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs="+",
        default=None,
        metavar="RUN",
        help="Restrict to these fine-tune run names (default: all ten).",
    )
    parser.add_argument(
        "--regime",
        choices=REGIMES,
        default=None,
        help="Restrict to fine-tunes of this pretraining regime.",
    )
    parser.add_argument(
        "--epoch",
        nargs="+",
        type=int,
        default=None,
        choices=EPOCHS,
        help=f"Restrict to these epochs (default: {EPOCHS[0]}-{EPOCHS[-1]}).",
    )
    parser.add_argument(
        "--perturbation",
        nargs="+",
        default=None,
        choices=list(SST_PERTURBATIONS),
        help="Restrict to these SST perturbation levels (default: all).",
    )
    parser.add_argument(
        "--forcing-grid",
        nargs="+",
        default=None,
        choices=list(DATASETS),
        help="Restrict to these forcing grids (default: both).",
    )
    add_beaker_args(
        parser,
        default_workspace="ai2/ace",
        default_cluster=["ai2/titan", "ai2/jupiter"],
        default_priority="normal",
    )
    args = parser.parse_args()

    runs = finetune_runs()
    if args.regime is not None:
        runs = {name: value for name, value in runs.items() if value[0] == args.regime}
    if args.run is not None:
        unknown = sorted(set(args.run) - set(runs))
        if unknown:
            raise KeyError(
                f"unknown fine-tune run(s) {unknown} — available: {sorted(runs)}"
            )
        runs = {name: runs[name] for name in args.run}

    grids = args.forcing_grid or list(DATASETS)
    levels = args.perturbation or list(SST_PERTURBATIONS)
    epochs = args.epoch or list(EPOCHS)

    missing = sorted(
        {
            source_run if epoch == SOURCE_EPOCH else finetune_run
            for finetune_run, (_, source_run) in runs.items()
            for epoch in epochs
        }
        - set(TRAINING_RESULT_DATASETS)
    )
    if missing:
        raise KeyError(
            f"No Beaker dataset ID for {missing} — refresh with "
            "update_beaker_map.py."
        )

    jobs: list[tuple[str, str, str, int, str, str, str]] = []
    for finetune_run, (regime, source_run) in sorted(runs.items()):
        for grid in grids:
            label = grid_label(grid, regime)
            for level in levels:
                config_filename = sst_epoch_config_filename(grid, label, level)
                for epoch in epochs:
                    dataset_id, checkpoint_path = epoch_checkpoint(
                        epoch, finetune_run, source_run
                    )
                    jobs.append(
                        (
                            finetune_run,
                            grid,
                            level,
                            epoch,
                            config_filename,
                            dataset_id,
                            checkpoint_path,
                        )
                    )

    jobs = drop_jobs_in_beaker(jobs, lambda job: sst_epoch_job_name(*job[:4]), args)

    needed_configs = sorted({job[4] for job in jobs})
    for config_filename in needed_configs:
        if not (RUN_CONFIGS_DIR / config_filename).exists():
            raise FileNotFoundError(
                f"{config_filename} not found — run generate_sst_configs.py first"
            )

    if not args.dry_run:
        check_configs_at_head([RUN_CONFIGS_DIR / name for name in needed_configs])
        validate_configs(needed_configs)

    print(f"{len(jobs)} job(s).")
    for (
        run_name,
        grid,
        level,
        epoch,
        config_filename,
        dataset_id,
        checkpoint_path,
    ) in jobs:
        submit_job(
            RUN_SCRIPT,
            [
                f"{RUN_CONFIGS_DIRNAME}/{config_filename}",
                sst_epoch_job_name(run_name, grid, level, epoch),
                WANDB_GROUP,
                dataset_id,
                checkpoint_path,
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=HERE,
            extra_env={"SKIP_VALIDATE": "1"},
        )


if __name__ == "__main__":
    main()

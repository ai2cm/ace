"""Checkpoints of a training result dataset that an evaluator suite can run against.

A training run's result dataset holds several checkpoints of the same model, and
each one can be evaluated on its own. One evaluation produces one wandb run named
after the training run plus that checkpoint's suffix, e.g.::

    ace2-var-mask-nc-sfno-era5-gmron-mask20-seed0-v5-mstepft-lastepoch

That (name, path, suffix) triple is the single concept this module owns, because
three different scripts need three different projections of it:

  - submit_eval_jobs.py    all three fields: --checkpoint choices, the file to
                           mount at /ckpt.tar, the evaluator run name
  - generate_eval_configs  the suffixes, i.e. which wandb runs must exist before
                           a suite config counts as fully evaluated
  - update_beaker_map.py   is_derived_run_name, i.e. which wandb runs are
                           evaluations rather than training runs

Keeping them in one place makes the invariant structural: a checkpoint added here
is automatically counted toward suite completeness and excluded from the training
run -> result dataset map, rather than needing the same suffix spelled into three
separate literals.
"""

import dataclasses
from collections.abc import Iterable


@dataclasses.dataclass(frozen=True)
class EvalCheckpoint:
    """One checkpoint of a training result dataset that a suite can evaluate.

    Attributes:
        name: The ``--checkpoint`` choice that selects this checkpoint.
        suffix: Appended to the training run name to name the evaluator run.
            Part of the wandb run name that ``--skip-evaluated`` matches on, so
            it is permanent: renaming it orphans every existing run.
        path: Path to the checkpoint within the result dataset.
    """

    name: str
    suffix: str
    path: str


# Every selectable checkpoint, in the canonical order jobs are submitted in.
# Adding an entry here makes it a --checkpoint choice, counts it toward suite
# completeness in generate_eval_configs.py, and excludes its runs from
# update_beaker_map.py's training run -> dataset map.
EVAL_CHECKPOINTS: tuple[EvalCheckpoint, ...] = (
    EvalCheckpoint("besttrain", "-besttrain", "training_checkpoints/best_ckpt.tar"),
    EvalCheckpoint(
        "bestinf", "-bestinf", "training_checkpoints/best_inference_ckpt.tar"
    ),
    EvalCheckpoint("lastepoch", "-lastepoch", "training_checkpoints/ckpt.tar"),
)


def names() -> tuple[str, ...]:
    """Every ``--checkpoint`` choice, in canonical order."""
    return tuple(checkpoint.name for checkpoint in EVAL_CHECKPOINTS)


def suffixes() -> tuple[str, ...]:
    """Every evaluator run name suffix, in canonical order."""
    return tuple(checkpoint.suffix for checkpoint in EVAL_CHECKPOINTS)


def by_names(wanted: Iterable[str]) -> tuple[EvalCheckpoint, ...]:
    """The named checkpoints, in canonical order and without duplicates.

    Iterating EVAL_CHECKPOINTS rather than ``wanted`` is what keeps a repeated
    ``--checkpoint`` name from submitting the same job twice.
    """
    selected = set(wanted)
    unknown = selected - set(names())
    if unknown:
        raise ValueError(
            f"unknown checkpoint(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(names())}"
        )
    return tuple(
        checkpoint for checkpoint in EVAL_CHECKPOINTS if checkpoint.name in selected
    )


def is_derived_run_name(run_name: str) -> bool:
    """Whether a wandb run is an evaluation of a training run rather than one.

    Checks every registered suffix, which is the point: suffixes can be prefixes
    of one another, so a hand-maintained list that omits the longer one silently
    classifies those evaluator runs as training runs.
    """
    return run_name.endswith(suffixes())

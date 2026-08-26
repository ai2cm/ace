"""Checkpoints of a training result dataset that an evaluator suite can run against.

A training run's result dataset holds several checkpoints of the same model, and
each one can be evaluated on its own. One evaluation produces one wandb run named
after the training run plus that checkpoint's suffix, e.g.::

    ace2-var-mask-nc-sfno-era5-gmron-mask20-seed0-v5-mstepft-lastepochema

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

EMA vs raw weights
------------------
``validate_using_ema: true`` makes the Trainer run validation and inline
inference under ``EMATracker.applied_params``, so every number on a training
run's own charts is the EMA-averaged model. Of the checkpoints written:

  - best_ckpt.tar and best_inference_ckpt.tar are saved inside that EMA context
    (``Trainer.save_all_checkpoints``), so they hold EMA weights.
  - ckpt.tar is the restart checkpoint, saved outside it, so it holds the raw
    weights. Its EMA state is a separate ``"ema"`` key that ``load_stepper``
    does not read, so evaluating ckpt.tar evaluates the raw model.
  - ema_ckpt_XXXX.tar is saved inside the EMA context, at the epochs selected by
    ``ema_checkpoint_save_epochs``.

So ``lastepoch`` and ``lastepochema`` evaluate the same epoch of the same run and
still disagree: only the latter is the model the training charts were made from.
"""

import dataclasses
import functools
import json
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence

# Where in a result dataset the epoch-based EMA checkpoints live, and the epoch
# number embedded in their filenames.
EMA_CHECKPOINT_PREFIX = "training_checkpoints/ema_ckpt"
EMA_CHECKPOINT_RE = re.compile(r"ema_ckpt_(\d+)\.tar$")

# Signature of a "list the paths in a beaker dataset under a prefix" callable.
# Injectable so tests do not shell out to beaker.
DatasetLister = Callable[[str, str], Sequence[str]]


def _beaker_dataset_ls(dataset_id: str, prefix: str) -> list[str]:
    """Paths in a beaker dataset that start with ``prefix``."""
    proc = subprocess.run(
        ["beaker", "dataset", "ls", dataset_id, prefix, "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [entry["path"] for entry in json.loads(proc.stdout)]


def latest_ema_checkpoint(
    dataset_id: str, lister: DatasetLister = _beaker_dataset_ls
) -> str | None:
    """Highest-epoch ``ema_ckpt_XXXX.tar`` in a result dataset, or None if it has none.

    Read from the dataset rather than computed from the training config's
    ``max_epochs`` and ``ema_checkpoint_save_epochs``: those two only predict what
    was written for a run that finished on the save grid, and a run killed before
    the first selected epoch has no EMA checkpoint at all.
    """
    epoch_to_path = {}
    for path in lister(dataset_id, EMA_CHECKPOINT_PREFIX):
        match = EMA_CHECKPOINT_RE.search(path)
        if match is not None:
            epoch_to_path[int(match.group(1))] = path
    if not epoch_to_path:
        return None
    return epoch_to_path[max(epoch_to_path)]


# Cached so that resolving the same dataset for several suites, or for a
# --dry-run followed by the real submission, shells out to beaker once.
_cached_latest_ema_checkpoint = functools.cache(latest_ema_checkpoint)


@dataclasses.dataclass(frozen=True)
class EvalCheckpoint:
    """One checkpoint of a training result dataset that a suite can evaluate.

    Attributes:
        name: The ``--checkpoint`` choice that selects this checkpoint.
        suffix: Appended to the training run name to name the evaluator run.
            Part of the wandb run name that ``--skip-evaluated`` matches on, so
            it is permanent: renaming it orphans every existing run.
        path: Path to the checkpoint within the result dataset, when it is the
            same for every run. None for a checkpoint whose filename encodes the
            epoch it was written at, which ``resolve`` looks up per dataset.
    """

    name: str
    suffix: str
    path: str | None = None

    def resolve(self, dataset_id: str) -> str | None:
        """Path to evaluate in ``dataset_id``, or None if it holds no candidate."""
        if self.path is not None:
            return self.path
        return _cached_latest_ema_checkpoint(dataset_id)


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
    # Final-epoch EMA weights. Same epoch as lastepoch, different model: see
    # "EMA vs raw weights" above. The path is resolved per dataset because the
    # final EMA epoch differs by family (20 for the fine-tunes, 150 for
    # pre-training).
    EvalCheckpoint("lastepochema", "-lastepochema"),
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

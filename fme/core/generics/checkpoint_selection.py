"""Drift-gated selection of a window of epochs to average into a soup checkpoint.

Each epoch with inline inference contributes a record of its weighted
inference error ("score") and of the per-sample global-mean drift slopes of
each weighted inference entry. A window of the most recent records passes the
drift gate if, for every (entry, variable), the slopes pooled over the
window's epochs and samples are not significantly different from zero. The
chosen window is the passing window with the lowest median score, or, if no
window has passed, the window with the lowest maximum drift t-statistic.
"""

import collections
import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch

STATUS_PASSED = "passed"
STATUS_FALLBACK = "fallback"
STATUS_PARTIAL = "partial"
STATUS_UNGATED = "ungated"

# inference entry name -> variable name -> [n_sample] slopes
EntryDriftSlopes = Mapping[str, Mapping[str, torch.Tensor]]


@dataclasses.dataclass
class CheckpointSelectionConfig:
    """
    Configuration for selecting a window of epochs whose EMA checkpoints are
    averaged into ``soup_ckpt.tar`` during training.

    After each epoch with inline inference, the most recent ``window`` epochs
    are evaluated. For each inference entry with ``weight > 0`` and each
    variable, the per-sample global-mean drift slopes are pooled across the
    window's epochs and samples, and their t-statistic
    ``mean / (std / sqrt(n))`` is computed. The window passes the drift gate
    if the maximum ``|t|`` is at most ``t_cutoff``; if no drift slopes are
    available (e.g. rollouts too short for the drift metric) the window is
    ungated and treated as passing. A window's score is the median of its
    epochs' weighted inference errors (the error used for
    ``best_inference_ckpt.tar``).

    The soup is the uniform average of the windows' epoch EMA checkpoints
    (``ema_ckpt_XXXX.tar``), so it averages EMA weights regardless of
    ``validate_using_ema``; when ``validate_using_ema`` is False the scores and
    drift slopes are computed with the non-EMA weights. EMA checkpoints must
    be saved every epoch (``ema_checkpoint_save_epochs`` covering every
    epoch).

    Parameters:
        window: Number of consecutive inference epochs in a window.
        t_cutoff: Maximum allowed absolute drift t-statistic for a window to
            pass. The default corresponds roughly to a two-sided 5% test with
            Bonferroni correction over ~30 variables.
        mode: ``"best_window"`` evaluates a candidate window after every
            inference epoch and keeps the best one, writing the soup each time
            the chosen window changes (until a full window exists, the most
            recent partial window is written). ``"final_window"`` writes the
            soup of the last ``window`` inference epochs once, at the end of
            training.
    """

    window: int = 5
    t_cutoff: float = 3.1
    mode: Literal["best_window", "final_window"] = "best_window"

    def __post_init__(self):
        if self.window < 1:
            raise ValueError(f"window must be at least 1, got {self.window}")
        if not self.t_cutoff > 0:
            raise ValueError(f"t_cutoff must be positive, got {self.t_cutoff}")
        if self.mode not in ("best_window", "final_window"):
            raise ValueError(
                f"mode must be 'best_window' or 'final_window', got {self.mode!r}"
            )


@dataclasses.dataclass
class EpochRecord:
    """Selection inputs recorded for one epoch with inline inference."""

    epoch: int
    score: float
    valid_loss: float
    drift_slopes: dict[str, dict[str, torch.Tensor]]

    def get_state(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "EpochRecord":
        return cls(
            epoch=int(state["epoch"]),
            score=float(state["score"]),
            valid_loss=float(state["valid_loss"]),
            drift_slopes={
                entry: {name: torch.as_tensor(v) for name, v in slopes.items()}
                for entry, slopes in state["drift_slopes"].items()
            },
        )


@dataclasses.dataclass
class WindowEvaluation:
    """The result of evaluating a window of epoch records.

    Attributes:
        epochs: The window's epochs, in increasing order.
        score: Median of the epochs' scores.
        max_abs_t: Maximum over (entry, variable) of the absolute drift
            t-statistic, NaN if no t-statistic could be computed.
        complete: Whether the window has the configured number of epochs.
        passed: Whether the window passes the drift gate.
    """

    epochs: list[int]
    score: float
    max_abs_t: float
    complete: bool
    passed: bool

    @property
    def ungated(self) -> bool:
        return math.isnan(self.max_abs_t)

    def get_state(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "WindowEvaluation":
        return cls(
            epochs=[int(e) for e in state["epochs"]],
            score=float(state["score"]),
            max_abs_t=float(state["max_abs_t"]),
            complete=bool(state["complete"]),
            passed=bool(state["passed"]),
        )


def _t_statistic(values: torch.Tensor) -> float:
    values = values.detach().to("cpu", torch.float64).flatten()
    values = values[torch.isfinite(values)]
    n = values.numel()
    if n < 2:
        return math.nan
    std = float(values.std(unbiased=True))
    mean = float(values.mean())
    if std == 0.0:
        return 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    return mean / (std / math.sqrt(n))


def max_abs_drift_t(records: Sequence[EpochRecord]) -> float:
    """Maximum absolute t-statistic of the drift slopes pooled over records.

    For each (entry, variable), the finite slopes of all records and samples
    are pooled into one t-statistic. Returns NaN if none could be computed.
    """
    pooled: dict[tuple[str, str], list[torch.Tensor]] = collections.defaultdict(list)
    for record in records:
        for entry, slopes in record.drift_slopes.items():
            for name, values in slopes.items():
                pooled[(entry, name)].append(values.flatten())
    abs_t = [abs(_t_statistic(torch.cat(values))) for values in pooled.values()]
    abs_t = [t for t in abs_t if not math.isnan(t)]
    if len(abs_t) == 0:
        return math.nan
    return max(abs_t)


def _nan_to_inf(value: float) -> float:
    return math.inf if math.isnan(value) else value


class CheckpointSelector:
    """Tracks epoch records and the chosen window of epochs to average.

    Only the root rank holds drift slopes, so only the root rank's selection
    is meaningful; the trainer updates the selector on the root rank only.
    """

    def __init__(self, config: CheckpointSelectionConfig):
        self._config = config
        self._records: collections.deque[EpochRecord] = collections.deque(
            maxlen=config.window
        )
        self._best_passing: WindowEvaluation | None = None
        self._best_fallback: WindowEvaluation | None = None
        self._latest_partial: WindowEvaluation | None = None

    def record(
        self,
        epoch: int,
        score: float,
        valid_loss: float,
        drift_slopes: EntryDriftSlopes,
    ) -> None:
        """Record the selection inputs of an epoch with inline inference."""
        self._records.append(
            EpochRecord(
                epoch=epoch,
                score=float(score),
                valid_loss=float(valid_loss),
                drift_slopes={
                    entry: {
                        name: values.detach().to("cpu", torch.float64).clone()
                        for name, values in slopes.items()
                    }
                    for entry, slopes in drift_slopes.items()
                },
            )
        )

    def evaluate_current_window(self) -> WindowEvaluation | None:
        """Evaluate the window of current records as a selection candidate."""
        if len(self._records) == 0:
            return None
        records = list(self._records)
        max_abs_t = max_abs_drift_t(records)
        evaluation = WindowEvaluation(
            epochs=[r.epoch for r in records],
            score=float(np.median([r.score for r in records])),
            max_abs_t=max_abs_t,
            complete=len(records) == self._config.window,
            passed=math.isnan(max_abs_t) or max_abs_t <= self._config.t_cutoff,
        )
        if not evaluation.complete:
            self._latest_partial = evaluation
        elif evaluation.passed:
            if self._best_passing is None or _nan_to_inf(
                evaluation.score
            ) < _nan_to_inf(self._best_passing.score):
                self._best_passing = evaluation
        elif self._best_fallback is None or (
            _nan_to_inf(evaluation.max_abs_t),
            _nan_to_inf(evaluation.score),
        ) < (
            _nan_to_inf(self._best_fallback.max_abs_t),
            _nan_to_inf(self._best_fallback.score),
        ):
            self._best_fallback = evaluation
        return evaluation

    @property
    def chosen(self) -> WindowEvaluation | None:
        """The chosen window: best passing, else best fallback, else partial."""
        if self._best_passing is not None:
            return self._best_passing
        if self._best_fallback is not None:
            return self._best_fallback
        return self._latest_partial

    @property
    def status(self) -> str | None:
        chosen = self.chosen
        if chosen is None:
            return None
        if not chosen.complete:
            return STATUS_PARTIAL
        if chosen.ungated:
            return STATUS_UNGATED
        if chosen.passed:
            return STATUS_PASSED
        return STATUS_FALLBACK

    def get_logs(self) -> dict[str, Any]:
        chosen = self.chosen
        if chosen is None:
            return {}
        return {
            "soup/status": self.status,
            "soup/epochs": f"{chosen.epochs[0]}-{chosen.epochs[-1]}",
            "soup/first_epoch": chosen.epochs[0],
            "soup/last_epoch": chosen.epochs[-1],
            "soup/score": chosen.score,
            "soup/max_abs_t": chosen.max_abs_t,
        }

    def get_state(self) -> dict[str, Any]:
        def _maybe_state(evaluation: WindowEvaluation | None):
            return None if evaluation is None else evaluation.get_state()

        return {
            "records": [record.get_state() for record in self._records],
            "best_passing": _maybe_state(self._best_passing),
            "best_fallback": _maybe_state(self._best_fallback),
            "latest_partial": _maybe_state(self._latest_partial),
            "logs": self.get_logs(),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        def _maybe_evaluation(value) -> WindowEvaluation | None:
            return None if value is None else WindowEvaluation.from_state(value)

        self._records.clear()
        for record in state["records"]:
            self._records.append(EpochRecord.from_state(record))
        self._best_passing = _maybe_evaluation(state["best_passing"])
        self._best_fallback = _maybe_evaluation(state["best_fallback"])
        self._latest_partial = _maybe_evaluation(state["latest_partial"])

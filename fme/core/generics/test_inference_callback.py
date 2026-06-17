"""Unit tests for build_inference_callback's sequential dispatch (generic).

These cover the per-task sequential fan-out and the log/error aggregation
shared by ACE and coupled training. (Concurrent grouping is added in a later
change and tested alongside it.)"""

from unittest.mock import MagicMock, patch

import pytest

from fme.core.generics.aggregator import InferenceSummary
from fme.core.generics.trainer import InferenceTask, build_inference_callback


def _make_task(name: str, weight: float = 0.0, epochs=(1,)) -> InferenceTask:
    return InferenceTask(
        name=name,
        data=MagicMock(),
        aggregator_factory=MagicMock(),
        epoch_set=frozenset(epochs),
        weight=weight,
    )


def _build(tasks, side_effect, inference_epochs=(1,)):
    stepper = MagicMock()
    mock = patch(
        "fme.core.generics.trainer.inference_one_epoch",
        side_effect=side_effect,
    )
    callback = build_inference_callback(
        tasks=tasks, inference_epochs=list(inference_epochs), stepper=stepper
    )
    return callback, mock


def test_epoch_not_in_inference_epochs_returns_empty():
    callback, mock = _build([_make_task("a", weight=1.0)], side_effect=[])
    with mock as m:
        logs, error = callback(epoch=2)
    assert logs == {}
    assert error is None
    assert m.call_count == 0


def test_runs_each_active_task_and_combines_weighted_error():
    tasks = [_make_task("a", weight=2.0), _make_task("b", weight=3.0)]
    callback, mock = _build(
        tasks,
        side_effect=[
            InferenceSummary(logs={"a/loss": 0.1}, loss=0.1),
            InferenceSummary(logs={"b/loss": 0.2}, loss=0.2),
        ],
    )
    with mock as m:
        logs, error = callback(epoch=1)
    assert m.call_count == 2
    assert error == pytest.approx(2.0 * 0.1 + 3.0 * 0.2)
    assert logs == {"a/loss": 0.1, "b/loss": 0.2}


def test_zero_weight_task_excluded_from_error():
    tasks = [_make_task("a", weight=1.0), _make_task("b", weight=0.0)]
    callback, mock = _build(
        tasks,
        side_effect=[
            InferenceSummary(logs={"a/loss": 0.3}, loss=0.3),
            InferenceSummary(logs={"b/loss": 9.0}, loss=9.0),
        ],
    )
    with mock:
        logs, error = callback(epoch=1)
    assert error == pytest.approx(0.3)
    assert "b/loss" in logs


def test_inactive_task_skipped():
    tasks = [
        _make_task("a", weight=1.0, epochs=(1,)),
        _make_task("b", weight=1.0, epochs=(2,)),
    ]
    callback, mock = _build(
        tasks,
        side_effect=[InferenceSummary(logs={"a/loss": 0.5}, loss=0.5)],
    )
    with mock as m:
        logs, error = callback(epoch=1)
    assert m.call_count == 1
    assert error == pytest.approx(0.5)
    assert "b/loss" not in logs


def test_overlapping_log_keys_raise():
    tasks = [_make_task("a", weight=0.0), _make_task("b", weight=0.0)]
    callback, mock = _build(
        tasks,
        side_effect=[
            InferenceSummary(logs={"shared": 0.1}, loss=None),
            InferenceSummary(logs={"shared": 0.2}, loss=None),
        ],
    )
    with mock, pytest.raises(RuntimeError, match="overlap"):
        callback(epoch=1)


def test_weighted_task_without_loss_raises():
    tasks = [_make_task("a", weight=1.0)]
    callback, mock = _build(
        tasks,
        side_effect=[InferenceSummary(logs={"a/x": 1.0}, loss=None)],
    )
    with mock, pytest.raises(RuntimeError, match="did not produce a loss"):
        callback(epoch=1)

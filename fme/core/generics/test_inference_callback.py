"""Unit tests for build_inference_callback (generic, used by ACE and coupled).

Covers both the sequential per-task dispatch and the concurrent grouping that
batches tasks sharing a ``concurrent_group`` through a single ``BatchedPredictor``."""

from collections.abc import Sequence
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from fme.core.generics.aggregator import InferenceAggregatorABC, InferenceSummary
from fme.core.generics.data import SimpleInferenceData
from fme.core.generics.inference import BatchedPredictor
from fme.core.generics.trainer import InferenceTask, build_inference_callback

# ---------------------------------------------------------------------------
# Sequential dispatch (no concurrent_group / build_batched_predictor)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Concurrent grouping dispatch (build_batched_predictor + concurrent_group)
# ---------------------------------------------------------------------------


@dataclass
class _FakeFD:
    samples: list[int]


@dataclass
class _FakePS:
    samples: list[int]


@dataclass
class _FakeSD:
    samples: list[int]


def _concat_fd(items: Sequence[_FakeFD]) -> _FakeFD:
    out: list[int] = []
    for item in items:
        out.extend(item.samples)
    return _FakeFD(samples=out)


def _concat_ps(items: Sequence[_FakePS]) -> _FakePS:
    out: list[int] = []
    for item in items:
        out.extend(item.samples)
    return _FakePS(samples=out)


def _split(item, sizes):
    out = []
    start = 0
    for size in sizes:
        out.append(type(item)(samples=list(item.samples[start : start + size])))
        start += size
    return out


def _fake_predict(initial_condition, forcing, compute_derived_variables=True):
    samples = [a + b for a, b in zip(initial_condition.samples, forcing.samples)]
    return _FakeSD(samples=samples), _FakePS(samples=samples)


class _CapturingAggregator(InferenceAggregatorABC):
    """Fake inference aggregator returning a fixed summary (unprefixed logs)."""

    def __init__(self, logs: dict[str, float], loss: float | None):
        self._logs = logs
        self._loss = loss
        self.batches: list = []
        self.flush_calls: list = []

    def record_initial_condition(self, initial_condition):
        return []

    def record_batch(self, data):
        self.batches.append(data)
        return []

    def get_summary(self) -> InferenceSummary:
        return InferenceSummary(logs=dict(self._logs), loss=self._loss)

    def flush_diagnostics(self, subdir):
        self.flush_calls.append(subdir)


def _make_stepper():
    stepper = MagicMock()
    stepper.set_eval = MagicMock()
    stepper.predict_paired = MagicMock(side_effect=_fake_predict)
    return stepper


def _make_predictor(stepper) -> BatchedPredictor:
    return BatchedPredictor(
        base_predict=stepper.predict_paired,
        concat_ic=_concat_ps,
        concat_forcing=_concat_fd,
        split_output=_split,
        split_state=_split,
        sample_size_of_state=lambda ps: len(ps.samples),
    )


def _make_grouped_task(
    name: str,
    logs: dict[str, float],
    loss: float | None = None,
    weight: float = 0.0,
    concurrent_group=None,
    n_windows: int = 2,
):
    data = SimpleInferenceData(
        initial_condition=_FakePS(samples=[0]),
        loader=[_FakeFD(samples=[i + 1]) for i in range(n_windows)],
    )
    aggregator = _CapturingAggregator(logs, loss)
    return (
        InferenceTask(
            name=name,
            data=data,
            aggregator_factory=lambda: aggregator,
            epoch_set=frozenset({1}),
            weight=weight,
            concurrent_group=concurrent_group,
        ),
        aggregator,
    )


def test_concurrent_group_batches_one_forward_per_window():
    """Two tasks with same concurrent_group key share a single forward pass."""
    stepper = _make_stepper()
    task_a, agg_a = _make_grouped_task(
        "a",
        {"time_mean_norm/rmse/channel_mean": 0.1},
        loss=0.1,
        weight=1.0,
        concurrent_group=(2,),
    )
    task_b, agg_b = _make_grouped_task(
        "b",
        {"time_mean_norm/rmse/channel_mean": 0.2},
        loss=0.2,
        weight=1.0,
        concurrent_group=(2,),
    )
    callback = build_inference_callback(
        tasks=[task_a, task_b],
        inference_epochs=[1],
        stepper=stepper,
        build_batched_predictor=lambda: _make_predictor(stepper),
    )
    logs, error = callback(epoch=1)
    # 2 windows shared across both tasks => 2 forward passes
    assert stepper.predict_paired.call_count == 2
    assert error == pytest.approx(0.1 + 0.2)
    assert "a/time_mean_norm/rmse/channel_mean" in logs
    assert "b/time_mean_norm/rmse/channel_mean" in logs
    assert agg_a.flush_calls == ["epoch_0001"]
    assert agg_b.flush_calls == ["epoch_0001"]


def test_distinct_concurrent_groups_run_separately():
    """Tasks with different concurrent_group keys do not share a forward pass."""
    stepper = _make_stepper()
    task_a, _ = _make_grouped_task(
        "a",
        {"time_mean_norm/rmse/channel_mean": 0.1},
        loss=0.1,
        weight=1.0,
        concurrent_group=(2,),
    )
    task_b, _ = _make_grouped_task(
        "b",
        {"time_mean_norm/rmse/channel_mean": 0.2},
        loss=0.2,
        weight=1.0,
        concurrent_group=(4,),
    )
    callback = build_inference_callback(
        tasks=[task_a, task_b],
        inference_epochs=[1],
        stepper=stepper,
        build_batched_predictor=lambda: _make_predictor(stepper),
    )
    # When groups differ, each task falls into its own group of size 1 — both
    # use the sequential path, hitting inference_one_epoch.
    with patch(
        "fme.core.generics.trainer.inference_one_epoch",
        side_effect=[
            InferenceSummary(
                logs={"a/time_mean_norm/rmse/channel_mean": 0.1}, loss=0.1
            ),
            InferenceSummary(
                logs={"b/time_mean_norm/rmse/channel_mean": 0.2}, loss=0.2
            ),
        ],
    ) as mock_inference:
        logs, error = callback(epoch=1)
    assert mock_inference.call_count == 2
    assert error == pytest.approx(0.3)
    assert {
        "a/time_mean_norm/rmse/channel_mean",
        "b/time_mean_norm/rmse/channel_mean",
    } <= set(logs)


def test_no_predictor_factory_runs_everything_sequentially():
    """Without build_batched_predictor, concurrent_group is ignored."""
    stepper = _make_stepper()
    task_a, _ = _make_grouped_task(
        "a",
        {"time_mean_norm/rmse/channel_mean": 0.1},
        loss=0.1,
        weight=1.0,
        concurrent_group=(2,),
    )
    task_b, _ = _make_grouped_task(
        "b",
        {"time_mean_norm/rmse/channel_mean": 0.2},
        loss=0.2,
        weight=1.0,
        concurrent_group=(2,),
    )
    callback = build_inference_callback(
        tasks=[task_a, task_b],
        inference_epochs=[1],
        stepper=stepper,
    )
    with patch(
        "fme.core.generics.trainer.inference_one_epoch",
        side_effect=[
            InferenceSummary(
                logs={"a/time_mean_norm/rmse/channel_mean": 0.1}, loss=0.1
            ),
            InferenceSummary(
                logs={"b/time_mean_norm/rmse/channel_mean": 0.2}, loss=0.2
            ),
        ],
    ) as mock_inference:
        logs, error = callback(epoch=1)
    assert mock_inference.call_count == 2
    assert error == pytest.approx(0.3)


def test_heterogeneous_lengths_in_concurrent_group():
    """A short task can drop out mid-group while others continue."""
    stepper = _make_stepper()
    task_short, agg_short = _make_grouped_task(
        "short",
        {"time_mean_norm/rmse/channel_mean": 0.1},
        loss=0.1,
        weight=1.0,
        concurrent_group=(2,),
        n_windows=1,
    )
    task_long, agg_long = _make_grouped_task(
        "long",
        {"time_mean_norm/rmse/channel_mean": 0.2},
        loss=0.2,
        weight=1.0,
        concurrent_group=(2,),
        n_windows=3,
    )
    callback = build_inference_callback(
        tasks=[task_short, task_long],
        inference_epochs=[1],
        stepper=stepper,
        build_batched_predictor=lambda: _make_predictor(stepper),
    )
    logs, error = callback(epoch=1)
    # First window batched (2), remaining 2 windows solo (long only)
    assert stepper.predict_paired.call_count == 3
    assert len(agg_short.batches) == 1
    assert len(agg_long.batches) == 3
    assert error == pytest.approx(0.3)

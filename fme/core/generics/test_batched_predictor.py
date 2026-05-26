"""Unit tests for BatchedPredictor, PredictionPromise, LazyLooper, and
run_concurrent_inference using a fake predict function (no model needed)."""

import unittest.mock
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from fme.core.generics.aggregator import InferenceAggregatorABC
from fme.core.generics.data import SimpleInferenceData
from fme.core.generics.inference import (
    BatchedPredictor,
    ConcurrentInferenceEntry,
    LazyLooper,
    PredictionPromise,
    run_concurrent_inference,
)
from fme.core.generics.writer import NullDataWriter
from fme.core.testing.wandb import mock_wandb
from fme.core.timing import GlobalTimer


@dataclass
class FakeFD:
    """Fake forcing-data batch: a list of integers, one per sample."""

    samples: list[int]


@dataclass
class FakePS:
    """Fake prognostic state: a list of integers, one per sample."""

    samples: list[int]


@dataclass
class FakeSD:
    """Fake stepped-data batch: a list of integers, one per sample."""

    samples: list[int]


def _concat_fd(items: Sequence[FakeFD]) -> FakeFD:
    out: list[int] = []
    for item in items:
        out.extend(item.samples)
    return FakeFD(samples=out)


def _concat_ps(items: Sequence[FakePS]) -> FakePS:
    out: list[int] = []
    for item in items:
        out.extend(item.samples)
    return FakePS(samples=out)


def _split_sd(item: FakeSD, sample_sizes: Sequence[int]) -> list[FakeSD]:
    out: list[FakeSD] = []
    start = 0
    for size in sample_sizes:
        out.append(FakeSD(samples=list(item.samples[start : start + size])))
        start += size
    return out


def _split_ps(item: FakePS, sample_sizes: Sequence[int]) -> list[FakePS]:
    out: list[FakePS] = []
    start = 0
    for size in sample_sizes:
        out.append(FakePS(samples=list(item.samples[start : start + size])))
        start += size
    return out


def _sample_size_of(fd: FakeFD) -> int:
    return len(fd.samples)


def _fake_predict(
    initial_condition: FakePS,
    forcing: FakeFD,
    compute_derived_variables: bool = True,
) -> tuple[FakeSD, FakePS]:
    """Returns ic + forcing as next output; new state is the output."""
    if len(initial_condition.samples) != len(forcing.samples):
        raise AssertionError(
            "Mismatched ic / forcing sample counts in fake predict: "
            f"{len(initial_condition.samples)} vs {len(forcing.samples)}"
        )
    samples = [a + b for a, b in zip(initial_condition.samples, forcing.samples)]
    return FakeSD(samples=samples), FakePS(samples=samples)


def _make_predictor() -> BatchedPredictor[FakePS, FakeFD, FakeSD]:
    return BatchedPredictor(
        base_predict=_fake_predict,
        concat_ic=_concat_ps,
        concat_forcing=_concat_fd,
        split_output=_split_sd,
        split_state=_split_ps,
        sample_size_of=_sample_size_of,
    )


def test_single_submission_resolves_to_underlying_predict():
    predictor = _make_predictor()
    promise = predictor.submit(FakePS(samples=[10]), FakeFD(samples=[1]))
    sd, ps = promise.resolve()
    assert sd.samples == [11]
    assert ps.samples == [11]


def test_two_submissions_share_one_forward_pass():
    predictor = _make_predictor()
    mock_base = unittest.mock.MagicMock(side_effect=_fake_predict)
    predictor._base_predict = mock_base
    p1 = predictor.submit(FakePS(samples=[10]), FakeFD(samples=[1]))
    p2 = predictor.submit(FakePS(samples=[20, 30]), FakeFD(samples=[2, 3]))
    sd1, ps1 = p1.resolve()
    sd2, ps2 = p2.resolve()
    assert mock_base.call_count == 1
    merged_ic, merged_forcing, *_ = mock_base.call_args.args
    assert merged_ic.samples == [10, 20, 30]
    assert merged_forcing.samples == [1, 2, 3]
    assert sd1.samples == [11]
    assert sd2.samples == [22, 33]
    assert ps1.samples == [11]
    assert ps2.samples == [22, 33]


def test_resolve_is_idempotent():
    predictor = _make_predictor()
    mock_base = unittest.mock.MagicMock(side_effect=_fake_predict)
    predictor._base_predict = mock_base
    p = predictor.submit(FakePS(samples=[10]), FakeFD(samples=[1]))
    r1 = p.resolve()
    r2 = p.resolve()
    assert r1 == r2
    assert mock_base.call_count == 1


def test_submit_after_flush_raises():
    predictor = _make_predictor()
    p = predictor.submit(FakePS(samples=[10]), FakeFD(samples=[1]))
    p.resolve()
    with pytest.raises(RuntimeError, match="Cannot submit after flush"):
        predictor.submit(FakePS(samples=[20]), FakeFD(samples=[2]))


def test_reset_allows_reuse():
    predictor = _make_predictor()
    p = predictor.submit(FakePS(samples=[10]), FakeFD(samples=[1]))
    p.resolve()
    predictor.reset()
    p2 = predictor.submit(FakePS(samples=[20]), FakeFD(samples=[2]))
    sd, _ = p2.resolve()
    assert sd.samples == [22]


def test_mismatched_compute_derived_variables_raises():
    predictor = _make_predictor()
    predictor.submit(
        FakePS(samples=[1]), FakeFD(samples=[1]), compute_derived_variables=True
    )
    with pytest.raises(ValueError, match="compute_derived_variables"):
        predictor.submit(
            FakePS(samples=[2]),
            FakeFD(samples=[2]),
            compute_derived_variables=False,
        )


def test_flush_without_submissions_raises():
    predictor = _make_predictor()
    with pytest.raises(RuntimeError, match="no pending submissions"):
        predictor._ensure_flushed()


def test_lazy_looper_iterates_and_updates_state():
    predictor = _make_predictor()
    loader: list[FakeFD] = [FakeFD(samples=[1]), FakeFD(samples=[2])]
    data = SimpleInferenceData(initial_condition=FakePS(samples=[0]), loader=loader)
    looper = LazyLooper(predictor, data)
    with GlobalTimer():
        looper.submit()
        sd = looper.commit()
        assert sd.samples == [1]
        assert looper.get_prognostic_state().samples == [1]
        predictor.reset()
        looper.submit()
        sd = looper.commit()
        assert sd.samples == [3]
        assert looper.get_prognostic_state().samples == [3]
        predictor.reset()
        with pytest.raises(StopIteration):
            looper.submit()


def test_lazy_looper_double_submit_raises():
    predictor = _make_predictor()
    data = SimpleInferenceData(
        initial_condition=FakePS(samples=[0]),
        loader=[FakeFD(samples=[1])],
    )
    looper = LazyLooper(predictor, data)
    with GlobalTimer():
        looper.submit()
        with pytest.raises(RuntimeError, match="commit"):
            looper.submit()


def test_lazy_looper_commit_without_submit_raises():
    predictor = _make_predictor()
    data = SimpleInferenceData(
        initial_condition=FakePS(samples=[0]),
        loader=[FakeFD(samples=[1])],
    )
    looper = LazyLooper(predictor, data)
    with pytest.raises(RuntimeError, match="submit"):
        looper.commit()


class _FakeAggregator(InferenceAggregatorABC[FakePS, FakeSD]):
    def __init__(self):
        self.initial_conditions: list[FakePS] = []
        self.batches: list[FakeSD] = []

    def record_initial_condition(self, initial_condition: FakePS):
        self.initial_conditions.append(initial_condition)
        return []

    def record_batch(self, data: FakeSD):
        self.batches.append(data)
        return []

    def get_summary_logs(self):
        return {}

    def flush_diagnostics(self, subdir):
        pass


def test_run_concurrent_inference_two_equal_length_runs():
    predictor = _make_predictor()
    mock_base = unittest.mock.MagicMock(side_effect=_fake_predict)
    predictor._base_predict = mock_base
    data_a = SimpleInferenceData(
        initial_condition=FakePS(samples=[0]),
        loader=[FakeFD(samples=[1]), FakeFD(samples=[1])],
    )
    data_b = SimpleInferenceData(
        initial_condition=FakePS(samples=[100]),
        loader=[FakeFD(samples=[10]), FakeFD(samples=[10])],
    )
    agg_a = _FakeAggregator()
    agg_b = _FakeAggregator()
    entries = [
        ConcurrentInferenceEntry(
            name="a",
            data=data_a,
            aggregator=agg_a,
            writer=NullDataWriter(),
            record_logs=lambda logs: None,
        ),
        ConcurrentInferenceEntry(
            name="b",
            data=data_b,
            aggregator=agg_b,
            writer=NullDataWriter(),
            record_logs=lambda logs: None,
        ),
    ]
    with GlobalTimer():
        run_concurrent_inference(predictor, entries)
    # One batched forward pass per window
    assert mock_base.call_count == 2
    assert [b.samples for b in agg_a.batches] == [[1], [2]]
    assert [b.samples for b in agg_b.batches] == [[110], [120]]


def test_run_concurrent_inference_heterogeneous_lengths():
    """Loopers with shorter datasets drop out; remaining ones continue."""
    predictor = _make_predictor()
    mock_base = unittest.mock.MagicMock(side_effect=_fake_predict)
    predictor._base_predict = mock_base
    data_short = SimpleInferenceData(
        initial_condition=FakePS(samples=[0]),
        loader=[FakeFD(samples=[1])],
    )
    data_long = SimpleInferenceData(
        initial_condition=FakePS(samples=[100]),
        loader=[
            FakeFD(samples=[10]),
            FakeFD(samples=[10]),
            FakeFD(samples=[10]),
        ],
    )
    agg_short = _FakeAggregator()
    agg_long = _FakeAggregator()
    entries = [
        ConcurrentInferenceEntry(
            name="short",
            data=data_short,
            aggregator=agg_short,
            writer=NullDataWriter(),
            record_logs=lambda logs: None,
        ),
        ConcurrentInferenceEntry(
            name="long",
            data=data_long,
            aggregator=agg_long,
            writer=NullDataWriter(),
            record_logs=lambda logs: None,
        ),
    ]
    with GlobalTimer():
        run_concurrent_inference(predictor, entries)
    # 3 forward passes: first is batched (2 entries), next two are solo (long only)
    assert mock_base.call_count == 3
    assert [b.samples for b in agg_short.batches] == [[1]]
    assert [b.samples for b in agg_long.batches] == [[110], [120], [130]]


def test_run_concurrent_inference_writes_initial_and_restart():
    predictor = _make_predictor()
    data = SimpleInferenceData(
        initial_condition=FakePS(samples=[5]),
        loader=[FakeFD(samples=[1])],
    )
    agg = _FakeAggregator()
    writer = unittest.mock.MagicMock(spec=NullDataWriter)
    entry: ConcurrentInferenceEntry[FakePS, FakeFD, FakeSD] = ConcurrentInferenceEntry(
        name="x",
        data=data,
        aggregator=agg,
        writer=writer,
        record_logs=lambda logs: None,
    )
    with GlobalTimer():
        run_concurrent_inference(predictor, [entry])
    write_calls = writer.write.call_args_list
    filenames = [c.args[1] for c in write_calls]
    assert filenames == ["initial_condition.nc", "restart.nc"]
    assert agg.initial_conditions[0].samples == [5]


def test_promise_resolve_returns_correct_slot():
    predictor = _make_predictor()
    p1 = predictor.submit(FakePS(samples=[0, 0]), FakeFD(samples=[1, 2]))
    p2 = predictor.submit(FakePS(samples=[100]), FakeFD(samples=[3]))
    p3 = predictor.submit(FakePS(samples=[0, 0, 0]), FakeFD(samples=[4, 5, 6]))
    sd1, _ = p1.resolve()
    sd3, _ = p3.resolve()
    sd2, _ = p2.resolve()
    assert sd1.samples == [1, 2]
    assert sd2.samples == [103]
    assert sd3.samples == [4, 5, 6]


def test_isinstance_promise_type():
    predictor = _make_predictor()
    p = predictor.submit(FakePS(samples=[1]), FakeFD(samples=[1]))
    assert isinstance(p, PredictionPromise)


def test_concurrent_inference_default_record_logs_uses_wandb():
    predictor = _make_predictor()
    data = SimpleInferenceData(
        initial_condition=FakePS(samples=[0]),
        loader=[FakeFD(samples=[1])],
    )
    entry = ConcurrentInferenceEntry(
        name="x",
        data=data,
        aggregator=_FakeAggregator(),
    )
    with mock_wandb() as wandb, GlobalTimer():
        wandb.configure(log_to_wandb=True)
        # Default record_logs comes from get_record_to_wandb; make sure the
        # default path doesn't error even with empty logs.
        run_concurrent_inference(predictor, [entry])

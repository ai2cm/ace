import collections
import pathlib

import numpy as np
import pytest
import xarray as xr

from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.getters import get_gridded_data
from fme.ace.data_loading.label_sampling import (
    LabelBalancedSampler,
    LabelSamplingConfig,
    _get_quotas,
)
from fme.ace.inference.evaluator import ValidationConfig
from fme.ace.requirements import DataRequirements
from fme.ace.train.train_config import InlineValidationConfig
from fme.core.dataset.concat import ConcatDatasetConfig, XarrayConcat
from fme.core.dataset.testing import MockDataset
from fme.core.dataset.xarray import XarrayDataConfig


def _make_sampler(
    n_per_label: dict[str, int],
    weights: dict[str, float],
    samples_per_epoch: int | None = None,
    num_replicas: int = 1,
    rank: int = 0,
) -> LabelBalancedSampler:
    dataset = XarrayConcat(
        [
            MockDataset.new(
                n_times=n, varnames=["var"], sample_n_times=1, labels={label}
            )
            for label, n in n_per_label.items()
        ]
    )
    return LabelBalancedSampler(
        dataset,
        sample_labels=[label for label, n in n_per_label.items() for _ in range(n)],
        weights=weights,
        samples_per_epoch=samples_per_epoch,
        num_replicas=num_replicas,
        rank=rank,
        seed=0,
    )


@pytest.mark.parametrize(
    "weights, total, expected",
    [
        ({"a": 1, "b": 1}, 10, {"a": 5, "b": 5}),
        (
            {"a": 0.4, "b": 0.25, "c": 0.25, "d": 0.1},
            100,
            {"a": 40, "b": 25, "c": 25, "d": 10},
        ),
        ({"a": 1, "b": 1, "c": 1}, 10, {"a": 4, "b": 3, "c": 3}),
        ({"a": 1, "b": 0}, 7, {"a": 7, "b": 0}),
        ({"a": 1000, "b": 1}, 10, {"a": 9, "b": 1}),
    ],
)
def test_get_quotas(weights, total, expected):
    quotas = _get_quotas(weights, total)
    assert quotas == expected
    assert sum(quotas.values()) == total


def test_get_quotas_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least one sample"):
        _get_quotas({"a": 1, "b": 1, "c": 1}, 2)


def test_sampler_draws_quota_per_label():
    sampler = _make_sampler(
        {"big": 30, "small": 3},
        weights={"big": 0.5, "small": 0.5},
        samples_per_epoch=12,
    )
    indices = list(sampler)
    assert len(indices) == len(sampler) == 12
    big_indices = [i for i in indices if i < 30]
    small_indices = [i for i in indices if i >= 30]
    assert len(big_indices) == len(small_indices) == 6
    # big has more samples than its quota, so none repeat
    assert len(set(big_indices)) == 6
    # small cycles through all of its samples before repeating any
    assert collections.Counter(small_indices) == {30: 2, 31: 2, 32: 2}


def test_sampler_default_samples_per_epoch_is_dataset_length():
    sampler = _make_sampler({"a": 8, "b": 2}, weights={"a": 1, "b": 1})
    assert len(list(sampler)) == 10
    assert sampler.quotas == {"a": 5, "b": 5}


def test_sampler_zero_weight_excludes_label():
    sampler = _make_sampler({"a": 4, "b": 4}, weights={"a": 1, "b": 0})
    assert all(i < 4 for i in sampler)


def test_sampler_order_depends_only_on_epoch():
    sampler = _make_sampler({"a": 50, "b": 50}, weights={"a": 1, "b": 1})
    other = _make_sampler({"a": 50, "b": 50}, weights={"a": 1, "b": 1})
    epoch_0 = list(sampler)
    assert epoch_0 == list(sampler)
    assert epoch_0 == list(other)
    sampler.set_epoch(1)
    assert list(sampler) != epoch_0
    other.set_epoch(1)
    assert list(sampler) == list(other)


def test_sampler_ranks_split_one_global_order():
    n_per_label = {"a": 10, "b": 4}
    weights = {"a": 1.0, "b": 1.0}
    num_replicas = 3
    per_rank = [
        list(_make_sampler(n_per_label, weights, num_replicas=num_replicas, rank=r))
        for r in range(num_replicas)
    ]
    # 14 samples round up to 15 so every rank draws the same number
    assert [len(indices) for indices in per_rank] == [5, 5, 5]
    single = _make_sampler(n_per_label, weights, samples_per_epoch=15)
    interleaved = [per_rank[i % num_replicas][i // num_replicas] for i in range(15)]
    assert interleaved == list(single)


def test_build_sampler_rejects_missing_weight():
    config = LabelSamplingConfig(weights={"a": 1.0})
    dataset = XarrayConcat(
        [
            MockDataset.new(
                n_times=4, varnames=["var"], sample_n_times=1, labels={label}
            )
            for label in ["a", "b"]
        ]
    )
    with pytest.raises(ValueError, match="exactly the dataset's labels"):
        config.build_sampler(dataset)


def test_build_sampler_rejects_weight_for_absent_label():
    config = LabelSamplingConfig(weights={"a": 1.0, "b": 1.0})
    dataset = MockDataset.new(
        n_times=4, varnames=["var"], sample_n_times=1, labels={"a"}
    )
    with pytest.raises(ValueError, match="exactly the dataset's labels"):
        config.build_sampler(dataset)


@pytest.mark.parametrize("labels", [{"a", "b"}, set(), None])
def test_build_sampler_requires_one_label_per_sample(labels):
    config = LabelSamplingConfig(weights={"a": 1.0, "b": 1.0})
    dataset = MockDataset.new(
        n_times=4, varnames=["var"], sample_n_times=1, labels=labels
    )
    with pytest.raises(ValueError, match="exactly one label"):
        config.build_sampler(dataset)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"weights": {"a": -1.0, "b": 1.0}}, "non-negative"),
        ({"weights": {"a": 0.0}}, "positive weight"),
        ({"weights": {"a": 1.0}, "samples_per_epoch": 0}, "samples_per_epoch"),
    ],
)
def test_label_sampling_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LabelSamplingConfig(**kwargs)


def _save_dataset(path: pathlib.Path, n_times: int):
    path.mkdir()
    time = xr.date_range("2000-01-01", periods=n_times, freq="D", use_cftime=True)
    ds = xr.Dataset(
        {"foo": (("time", "lat", "lon"), np.zeros((n_times, 2, 4), dtype=np.float32))},
        coords={"time": time, "lat": np.arange(2.0), "lon": np.arange(4.0)},
    )
    ds.to_netcdf(path / "data.nc")


def _labeled_concat(tmp_path: pathlib.Path, n_times_per_label: dict[str, int]):
    configs = []
    for label, n_times in n_times_per_label.items():
        _save_dataset(tmp_path / label, n_times)
        configs.append(
            XarrayDataConfig(data_path=str(tmp_path / label), labels=[label])
        )
    return ConcatDatasetConfig(concat=configs)


def test_data_loader_config_rejects_label_sampling_with_replacement(tmp_path):
    with pytest.raises(ValueError, match="cannot both be set"):
        DataLoaderConfig(
            dataset=_labeled_concat(tmp_path, {"a": 4}),
            batch_size=1,
            sample_with_replacement=4,
            label_sampling=LabelSamplingConfig(weights={"a": 1.0}),
        )


def test_data_loader_config_rejects_label_sampling_without_labels(tmp_path):
    _save_dataset(tmp_path / "a", 4)
    with pytest.raises(ValueError, match="labeled dataset"):
        DataLoaderConfig(
            dataset=XarrayDataConfig(data_path=str(tmp_path / "a")),
            batch_size=1,
            label_sampling=LabelSamplingConfig(weights={"a": 1.0}),
        )


@pytest.mark.parametrize(
    "validation_config", [InlineValidationConfig, ValidationConfig]
)
def test_validation_loader_rejects_label_sampling(tmp_path, validation_config):
    loader = DataLoaderConfig(
        dataset=_labeled_concat(tmp_path, {"a": 4}),
        batch_size=1,
        label_sampling=LabelSamplingConfig(weights={"a": 1.0}),
    )
    with pytest.raises(ValueError, match="only supported for the training loader"):
        validation_config(loader=loader)


@pytest.mark.xfail(
    reason=(
        "Concatenating labeled datasets merges every member's labels into the "
        "first member's label set until #1588 lands, so the first member's "
        "samples carry all labels."
    )
)
@pytest.mark.parametrize("time_buffer", [0, 2])
def test_gridded_data_draws_configured_label_fractions(tmp_path, time_buffer):
    config = DataLoaderConfig(
        dataset=_labeled_concat(tmp_path, {"big": 61, "small": 13}),
        batch_size=2,
        time_buffer=time_buffer,
        label_sampling=LabelSamplingConfig(
            weights={"big": 1.0, "small": 1.0}, samples_per_epoch=20
        ),
    )
    data = get_gridded_data(
        config, train=True, requirements=DataRequirements(["foo"], 2)
    )
    counts: collections.Counter[str] = collections.Counter()
    for batch in data.loader:
        assert batch.labels is not None
        for row in batch.labels.tensor:
            counts[batch.labels.names[int(row.argmax())]] += 1
    n_per_window = time_buffer + 1
    assert counts == {"big": 10 * n_per_window, "small": 10 * n_per_window}

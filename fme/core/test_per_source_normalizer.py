import dataclasses
import pathlib
import tempfile

import dacite
import torch

from fme.ace.testing.fv3gfs_data import get_scalar_dataset
from fme.core.device import move_tensordict_to_device
from fme.core.labels import BatchLabels
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.per_source_normalizer import (
    PerSourceNormalizationConfig,
    PerSourceNormalizer,
)


def _make_normalizer(mean_val: float, std_val: float) -> StandardNormalizer:
    means = move_tensordict_to_device(
        {"a": torch.tensor(mean_val), "b": torch.tensor(mean_val)}
    )
    stds = move_tensordict_to_device(
        {"a": torch.tensor(std_val), "b": torch.tensor(std_val)}
    )
    return StandardNormalizer(means=means, stds=stds)


def _make_labels(label_indices: list[int], names: list[str]) -> BatchLabels:
    n = len(label_indices)
    tensor = torch.zeros(n, len(names))
    for i, idx in enumerate(label_indices):
        tensor[i, idx] = 1.0
    tensor = move_tensordict_to_device({"_t": tensor})["_t"]
    return BatchLabels(tensor=tensor, names=names)


def _to_device(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return move_tensordict_to_device(tensors)


def test_per_source_normalizer_selects_by_label():
    default = _make_normalizer(0.0, 1.0)
    model_a = _make_normalizer(10.0, 2.0)
    model_b = _make_normalizer(20.0, 5.0)
    normalizer = PerSourceNormalizer(
        normalizers={"model_a": model_a, "model_b": model_b},
        default=default,
    )
    labels = _make_labels([0, 1], names=["model_a", "model_b"])
    tensors = _to_device(
        {"a": torch.tensor([10.0, 20.0]), "b": torch.tensor([10.0, 20.0])}
    )
    result = normalizer.normalize(tensors, labels)
    torch.testing.assert_close(
        result["a"], torch.tensor([0.0, 0.0], device=result["a"].device)
    )


def test_per_source_normalizer_denormalize_round_trip():
    default = _make_normalizer(0.0, 1.0)
    model_a = _make_normalizer(10.0, 2.0)
    normalizer = PerSourceNormalizer(normalizers={"model_a": model_a}, default=default)
    labels = _make_labels([0, 0], names=["model_a"])
    tensors = _to_device(
        {"a": torch.tensor([5.0, 15.0]), "b": torch.tensor([3.0, 7.0])}
    )
    normed = normalizer.normalize(tensors, labels)
    recovered = normalizer.denormalize(normed, labels)
    torch.testing.assert_close(recovered["a"], tensors["a"])
    torch.testing.assert_close(recovered["b"], tensors["b"])


def test_per_source_normalizer_fallback_to_default():
    default = _make_normalizer(100.0, 10.0)
    normalizer = PerSourceNormalizer(normalizers={}, default=default)
    labels = _make_labels([0], names=["unknown_model"])
    tensors = _to_device({"a": torch.tensor([100.0]), "b": torch.tensor([100.0])})
    result = normalizer.normalize(tensors, labels)
    torch.testing.assert_close(
        result["a"], torch.tensor([0.0], device=result["a"].device)
    )


def test_per_source_normalizer_no_labels_uses_default():
    default = _make_normalizer(5.0, 1.0)
    model_a = _make_normalizer(0.0, 1.0)
    normalizer = PerSourceNormalizer(normalizers={"model_a": model_a}, default=default)
    tensors = _to_device({"a": torch.tensor([5.0])})
    result = normalizer.normalize(tensors, labels=None)
    torch.testing.assert_close(
        result["a"], torch.tensor([0.0], device=result["a"].device)
    )


def test_per_source_normalizer_mixed_label_batch():
    default = _make_normalizer(0.0, 1.0)
    model_a = _make_normalizer(10.0, 1.0)
    model_b = _make_normalizer(20.0, 1.0)
    normalizer = PerSourceNormalizer(
        normalizers={"model_a": model_a, "model_b": model_b},
        default=default,
    )
    labels = _make_labels([0, 1, 0], names=["model_a", "model_b"])
    tensors = _to_device({"a": torch.tensor([12.0, 23.0, 11.0])})
    result = normalizer.normalize(tensors, labels)
    torch.testing.assert_close(
        result["a"], torch.tensor([2.0, 3.0, 1.0], device=result["a"].device)
    )


def test_per_source_normalization_config_load_from_directory():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        norm_dir = tmp_path / "per_source_normalization"
        for label, fill in [("model_a", 1.0), ("model_b", 2.0)]:
            label_dir = norm_dir / label
            label_dir.mkdir(parents=True)
            get_scalar_dataset(["a", "b"], fill_value=fill).to_netcdf(
                label_dir / "centering.nc"
            )
            get_scalar_dataset(["a", "b"], fill_value=fill + 10).to_netcdf(
                label_dir / "scaling.nc"
            )
        config = PerSourceNormalizationConfig(data_dir=str(tmp_path))
        config.load()
        assert config.data_dir is None
        assert "model_a" in config.sources
        assert "model_b" in config.sources
        normalizer = config.build(["a", "b"], default=_make_normalizer(0.0, 1.0))
        assert "model_a" in normalizer._normalizers
        assert "model_b" in normalizer._normalizers


def test_per_source_normalization_config_explicit_sources():
    config = PerSourceNormalizationConfig(
        sources={
            "model_a": NormalizationConfig(
                means={"a": 1.0, "b": 1.0}, stds={"a": 2.0, "b": 2.0}
            ),
        }
    )
    normalizer = config.build(["a", "b"], default=_make_normalizer(0.0, 1.0))
    assert "model_a" in normalizer._normalizers


def test_per_source_normalization_config_round_trip():
    config = PerSourceNormalizationConfig(
        sources={
            "model_a": NormalizationConfig(means={"a": 1.0}, stds={"a": 2.0}),
        }
    )
    state = dataclasses.asdict(config)
    restored = dacite.from_dict(
        PerSourceNormalizationConfig,
        data=state,
        config=dacite.Config(strict=True),
    )
    assert "model_a" in restored.sources
    assert restored.sources["model_a"].means == {"a": 1.0}

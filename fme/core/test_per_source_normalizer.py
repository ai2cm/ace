import dataclasses
import pathlib
import tempfile

import dacite
import pytest
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


def _make_normalizer_for_keys(
    keys: list[str], mean_val: float, std_val: float
) -> StandardNormalizer:
    """``_make_normalizer`` but with a configurable key set so we can
    exercise the "per-source normalizer lacks a key the loader has"
    path without hard-coding {a, b}."""
    means = move_tensordict_to_device({k: torch.tensor(mean_val) for k in keys})
    stds = move_tensordict_to_device({k: torch.tensor(std_val) for k in keys})
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


def test_per_source_normalizer_passes_through_var_missing_from_source():
    """If a source's per-source normalizer lacks a variable that the
    loader still provides (NaN-filled under
    ``allow_missing_variables``), that variable's slice for the
    source's samples must pass through unchanged — not get dropped
    (KeyError downstream in Packer) and not get filled with
    ``empty_like`` garbage (the prior bug)."""
    default = _make_normalizer_for_keys(["a", "b"], 0.0, 1.0)
    # model_a publishes both vars; model_b only publishes ``a``.
    model_a = _make_normalizer_for_keys(["a", "b"], 10.0, 1.0)
    model_b = _make_normalizer_for_keys(["a"], 20.0, 1.0)
    normalizer = PerSourceNormalizer(
        normalizers={"model_a": model_a, "model_b": model_b},
        default=default,
    )
    labels = _make_labels([0, 1, 0], names=["model_a", "model_b"])
    nan = float("nan")
    tensors = _to_device(
        {
            "a": torch.tensor([12.0, 23.0, 11.0]),
            # model_b's batch position (index 1) is NaN-filled by the
            # loader because model_b doesn't publish ``b``.
            "b": torch.tensor([15.0, nan, 14.0]),
        }
    )

    result = normalizer.normalize(tensors, labels)

    # ``a`` is normalized per-source for every sample.
    torch.testing.assert_close(
        result["a"],
        torch.tensor([2.0, 3.0, 1.0], device=result["a"].device),
    )
    # ``b`` is normalized for model_a's samples and passes through
    # NaN for model_b's sample (the source lacks stats for it).
    torch.testing.assert_close(
        result["b"][0], torch.tensor(5.0, device=result["b"].device)
    )
    torch.testing.assert_close(
        result["b"][2], torch.tensor(4.0, device=result["b"].device)
    )
    assert torch.isnan(result["b"][1])


def test_per_source_normalizer_single_source_missing_var_passes_through():
    """The single-source fast-path code also has to handle variables
    the source lacks. Pre-fix this returned a dict missing the key,
    which crashes the downstream Packer."""
    default = _make_normalizer_for_keys(["a", "b"], 0.0, 1.0)
    model_b = _make_normalizer_for_keys(["a"], 20.0, 1.0)
    normalizer = PerSourceNormalizer(
        normalizers={"model_b": model_b},
        default=default,
    )
    labels = _make_labels([0, 0], names=["model_b"])
    nan = float("nan")
    tensors = _to_device(
        {
            "a": torch.tensor([22.0, 25.0]),
            "b": torch.tensor([nan, nan]),
        }
    )

    result = normalizer.normalize(tensors, labels)

    # Both ``a`` and ``b`` are present in the output dict (no key
    # dropped). ``a`` is normalized by model_b; ``b`` passes through.
    assert "a" in result
    assert "b" in result
    torch.testing.assert_close(
        result["a"], torch.tensor([2.0, 5.0], device=result["a"].device)
    )
    assert torch.isnan(result["b"]).all()


def test_per_source_normalizer_does_not_propagate_empty_like_values():
    """Regression guard for the ``empty_like`` bug: every value in
    the output must be deterministic, even for variables some source
    in the batch lacks. Specifically, the output value for a
    missing-from-source sample must equal the input value, not
    uninitialized memory."""
    default = _make_normalizer_for_keys(["a", "b"], 0.0, 1.0)
    model_a = _make_normalizer_for_keys(["a", "b"], 0.0, 1.0)
    model_b = _make_normalizer_for_keys(["a"], 0.0, 1.0)
    normalizer = PerSourceNormalizer(
        normalizers={"model_a": model_a, "model_b": model_b},
        default=default,
    )
    labels = _make_labels([0, 1], names=["model_a", "model_b"])
    sentinel = torch.tensor([7.5, 9.25])
    tensors = _to_device({"a": torch.tensor([1.0, 2.0]), "b": sentinel.clone()})

    result = normalizer.normalize(tensors, labels)
    # model_b's slice of ``b`` (index 1) must equal the input
    # sentinel (9.25), not whatever ``empty_like`` happened to
    # allocate. model_a's slice (index 0) gets normalized
    # ((7.5 - 0) / 1 = 7.5).
    torch.testing.assert_close(result["b"], sentinel.to(device=result["b"].device))


def test_per_source_normalizer_data_mask_consistency_raises():
    """If the data_mask says a sample is genuinely present for a
    variable, but the source's per-source normalizer has no stats
    for that variable, that's a config mismatch (loader and stats
    disagree). Raise rather than silently producing NaN."""
    default = _make_normalizer_for_keys(["a", "b"], 0.0, 1.0)
    model_b = _make_normalizer_for_keys(["a"], 20.0, 1.0)
    normalizer = PerSourceNormalizer(
        normalizers={"model_b": model_b},
        default=default,
    )
    labels = _make_labels([0, 0], names=["model_b"])
    tensors = _to_device(
        {"a": torch.tensor([22.0, 25.0]), "b": torch.tensor([15.0, 14.0])}
    )
    # Loader claims ``b`` is present for both samples — contradicts
    # model_b's normalizer missing ``b``.
    data_mask = _to_device(
        {"a": torch.tensor([True, True]), "b": torch.tensor([True, True])}
    )

    with pytest.raises(ValueError, match=r"no stats for variable 'b'"):
        normalizer.normalize(tensors, labels, data_mask=data_mask)


def test_per_source_normalizer_data_mask_all_masked_no_raise():
    """When the data_mask correctly reports the variable as absent
    for every sample in the source-lacking-stats slice, the
    normalizer passes through without raising."""
    default = _make_normalizer_for_keys(["a", "b"], 0.0, 1.0)
    model_b = _make_normalizer_for_keys(["a"], 20.0, 1.0)
    normalizer = PerSourceNormalizer(
        normalizers={"model_b": model_b},
        default=default,
    )
    labels = _make_labels([0, 0], names=["model_b"])
    nan = float("nan")
    tensors = _to_device(
        {"a": torch.tensor([22.0, 25.0]), "b": torch.tensor([nan, nan])}
    )
    data_mask = _to_device(
        {
            "a": torch.tensor([True, True]),
            "b": torch.tensor([False, False]),  # consistent with model_b lacking 'b'
        }
    )

    result = normalizer.normalize(tensors, labels, data_mask=data_mask)
    torch.testing.assert_close(
        result["a"], torch.tensor([2.0, 5.0], device=result["a"].device)
    )
    assert torch.isnan(result["b"]).all()


def test_per_source_normalizer_data_mask_only_complains_for_lacking_source():
    """In a multi-source batch, one source has stats for ``b`` and
    another doesn't. The consistency check should raise only when
    the unmasked-yet-lacking-stats condition is violated for the
    source actually lacking stats — model_a (which has ``b``) can
    have unmasked ``b`` samples without triggering."""
    default = _make_normalizer_for_keys(["a", "b"], 0.0, 1.0)
    model_a = _make_normalizer_for_keys(["a", "b"], 10.0, 1.0)
    model_b = _make_normalizer_for_keys(["a"], 20.0, 1.0)
    normalizer = PerSourceNormalizer(
        normalizers={"model_a": model_a, "model_b": model_b},
        default=default,
    )
    labels = _make_labels([0, 1, 0], names=["model_a", "model_b"])
    nan = float("nan")
    tensors = _to_device(
        {"a": torch.tensor([12.0, 23.0, 11.0]), "b": torch.tensor([15.0, nan, 14.0])}
    )
    # model_a samples have ``b`` present (True), model_b's sample
    # has ``b`` masked (False). This is internally consistent.
    data_mask = _to_device(
        {
            "a": torch.tensor([True, True, True]),
            "b": torch.tensor([True, False, True]),
        }
    )

    result = normalizer.normalize(tensors, labels, data_mask=data_mask)
    torch.testing.assert_close(
        result["b"][0], torch.tensor(5.0, device=result["b"].device)
    )
    assert torch.isnan(result["b"][1])
    torch.testing.assert_close(
        result["b"][2], torch.tensor(4.0, device=result["b"].device)
    )


def test_per_source_normalizer_denormalize_passes_through_var_missing_from_source():
    """The denormalize path has the same exposure as normalize: a
    source lacking stats for a variable must not drop the key or
    propagate uninitialized memory."""
    default = _make_normalizer_for_keys(["a", "b"], 0.0, 1.0)
    model_a = _make_normalizer_for_keys(["a", "b"], 10.0, 1.0)
    model_b = _make_normalizer_for_keys(["a"], 20.0, 1.0)
    normalizer = PerSourceNormalizer(
        normalizers={"model_a": model_a, "model_b": model_b},
        default=default,
    )
    labels = _make_labels([0, 1], names=["model_a", "model_b"])
    sentinel = torch.tensor([5.0, 9.25])
    tensors = _to_device({"a": torch.tensor([1.0, 2.0]), "b": sentinel.clone()})

    result = normalizer.denormalize(tensors, labels)
    # model_b's slice (index 1) of ``b`` passes through unchanged.
    torch.testing.assert_close(result["b"][1].cpu(), sentinel[1].cpu())
    # model_a's slice (index 0) gets denormalized: 5.0 * 1.0 + 10.0 = 15.0.
    torch.testing.assert_close(result["b"][0].cpu(), torch.tensor(15.0))


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

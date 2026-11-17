import dataclasses
import logging
import pathlib
import tempfile

import dacite
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.testing.fv3gfs_data import get_scalar_dataset
from fme.core.device import get_device, move_tensordict_to_device
from fme.core.labels import BatchLabels
from fme.core.normalizer import (
    GroupedNormalizationConfig,
    GroupedNormalizer,
    NetworkAndLossNormalizationConfig,
    NormalizationConfig,
    NormalizeFn,
    StandardNormalizer,
    _combine_normalizers,
)
from fme.core.testing import write_normalization_stats


def test_normalize_depends_on_mean():
    means = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    stds = {"a": torch.tensor(1.0), "b": torch.tensor(1.0)}
    normalizer = StandardNormalizer(means=means, stds=stds)
    tensors = {"a": torch.tensor(1.0), "b": torch.tensor(1.0)}
    normalized = normalizer.normalize(tensors)
    assert normalized["a"] == torch.tensor(0.0)
    assert normalized["b"] == torch.tensor(-1.0)


def test_normalize_depends_on_std():
    means = {"a": torch.tensor(0.0), "b": torch.tensor(0.0)}
    stds = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    normalizer = StandardNormalizer(means=means, stds=stds)
    tensors = {"a": torch.tensor(1.0), "b": torch.tensor(1.0)}
    normalized = normalizer.normalize(tensors)
    assert normalized["a"] == torch.tensor(1.0)
    assert normalized["b"] == torch.tensor(0.5)


def test_normalize_without_mean_divides_by_std_only():
    means = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    stds = {"a": torch.tensor(2.0), "b": torch.tensor(4.0)}
    normalizer = StandardNormalizer(means=means, stds=stds)
    tensors = {"a": torch.tensor(3.0), "b": torch.tensor(3.0), "c": torch.tensor(3.0)}
    normalized = normalizer.normalize(tensors, apply_mean=False)
    assert normalized["a"] == torch.tensor(1.5)
    assert normalized["b"] == torch.tensor(0.75)
    assert "c" not in normalized


def test_normalize_applies_mean_by_default():
    means = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    stds = {"a": torch.tensor(2.0), "b": torch.tensor(4.0)}
    normalizer = StandardNormalizer(means=means, stds=stds)
    tensors = {"a": torch.tensor(3.0), "b": torch.tensor(3.0)}
    normalized = normalizer.normalize(tensors)
    torch.testing.assert_close(
        normalized, normalizer.normalize(tensors, apply_mean=True)
    )
    assert normalized["a"] == torch.tensor(1.0)
    assert normalized["b"] == torch.tensor(0.25)


def test_normalize_satisfies_normalize_fn_protocol():
    normalizer = StandardNormalizer(
        means={"a": torch.tensor(1.0)}, stds={"a": torch.tensor(2.0)}
    )
    # the annotation makes mypy verify that normalize satisfies the protocol
    normalize: NormalizeFn = normalizer.normalize
    tensors = {"a": torch.tensor(3.0)}
    assert normalize(tensors)["a"] == torch.tensor(1.0)
    assert normalize(tensors, apply_mean=False)["a"] == torch.tensor(1.5)


def test_denormalize_depends_on_mean():
    means = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    stds = {"a": torch.tensor(1.0), "b": torch.tensor(1.0)}
    normalizer = StandardNormalizer(means=means, stds=stds)
    tensors = {"a": torch.tensor(0.0), "b": torch.tensor(-1.0)}
    denormalized = normalizer.denormalize(tensors)
    assert denormalized["a"] == torch.tensor(1.0)
    assert denormalized["b"] == torch.tensor(1.0)


def test_denormalize_depends_on_std():
    means = {"a": torch.tensor(0.0), "b": torch.tensor(0.0)}
    stds = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    normalizer = StandardNormalizer(means=means, stds=stds)
    tensors = {"a": torch.tensor(1.0), "b": torch.tensor(0.5)}
    denormalized = normalizer.denormalize(tensors)
    assert denormalized["a"] == torch.tensor(1.0)
    assert denormalized["b"] == torch.tensor(1.0)


def test_normalize_and_denormalize_random_tensor():
    torch.manual_seed(0)
    # randomly set means and stds
    means = move_tensordict_to_device({"a": torch.randn(1), "b": torch.randn(1)})
    stds = move_tensordict_to_device({"a": torch.randn(1), "b": torch.randn(1)})
    normalizer = StandardNormalizer(means=means, stds=stds)
    tensors = move_tensordict_to_device({"a": torch.randn(10), "b": torch.randn(10)})
    denormalized = normalizer.denormalize(normalizer.normalize(tensors))
    torch.testing.assert_close(denormalized["a"], tensors["a"])
    torch.testing.assert_close(denormalized["b"], tensors["b"])


def test_missing_normalization_build_raises_error():
    normalization = NormalizationConfig(
        means={"a": 1.0, "b": 2.0},
        stds={"a": 1.0, "b": 1.0},
    )
    all_names = ["a", "b", "c"]
    with pytest.raises(ValueError, match=r"No normalization constants for .*\['c'\]"):
        normalization.build(all_names)


def test_tensors_with_missing_normalization_stats_get_filtered():
    normalization = NormalizationConfig(
        means={"a": 1.0, "b": 2.0},
        stds={"a": 1.0, "b": 1.0},
    ).build(["a", "b"])
    sample_input = {"a": torch.zeros(1), "b": torch.zeros(1), "c": torch.zeros(1)}
    sample_input = move_tensordict_to_device(sample_input)

    normalized = normalization.normalize(sample_input)
    assert "c" not in normalized

    denormalized = normalization.denormalize(sample_input)
    assert "c" not in denormalized


@pytest.mark.parametrize("fill_nans_on_normalize", [True, False])
@pytest.mark.parametrize("fill_nans_on_denormalize", [True, False])
def test_normalization_with_nans(fill_nans_on_normalize, fill_nans_on_denormalize):
    means = {"a": 1.0, "b": 2.0}
    stds = {"a": 1.0, "b": 2.0}
    normalization = NormalizationConfig(
        means=means,
        stds=stds,
        fill_nans_on_normalize=fill_nans_on_normalize,
        fill_nans_on_denormalize=fill_nans_on_denormalize,
    ).build(["a", "b"])
    denormalized_input = {
        "a": torch.tensor([-1.0, float("nan"), 1.0]),
        "b": torch.tensor([0.0, float("nan"), 4.0]),
    }
    denormalized_input = move_tensordict_to_device(denormalized_input)
    normalized = normalization.normalize(denormalized_input)
    if fill_nans_on_normalize:
        assert not torch.isnan(normalized["a"][1]), "normalized_nans_removed_a"
        assert normalized["a"][1] == torch.tensor(0), "normalized_nans_filled_means_a"
        assert not torch.isnan(normalized["b"][1]), "normalized_nans_removed_b"
        assert normalized["b"][1] == torch.tensor(0), "normalized_nans_filled_means_b"
    else:
        assert torch.isnan(normalized["a"][1]), "normalized_nans_not_removed_a"
        assert torch.isnan(normalized["b"][1]), "normalized_nans_not_removed_b"

    normalized_input = {
        "a": torch.tensor([-1.0, float("nan"), 1.0]),
        "b": torch.tensor([-1.0, float("nan"), 1.0]),
    }
    normalized_input = move_tensordict_to_device(normalized_input)
    denormalized = normalization.denormalize(normalized_input)
    if fill_nans_on_denormalize:
        assert not torch.isnan(denormalized["a"][1]), "denormalized_nans_removed_a"
        assert denormalized["a"][1] == torch.tensor(
            means["a"]
        ), "denormalized_nans_filled_means_a"
        assert not torch.isnan(denormalized["b"][1]), "denormalized_nans_removed_b"
        assert denormalized["b"][1] == torch.tensor(
            means["b"]
        ), "denormalized_nans_filled_means_b"
    else:
        assert torch.isnan(denormalized["a"][1]), "denormalized_nans_not_removed_a"
        assert torch.isnan(denormalized["b"][1]), "denormalized_nans_not_removed_b"


def test_combined_normalization_uses_network_normalizer_for_loss():
    torch.manual_seed(0)
    network_config = NormalizationConfig(
        means={"a": 1.0, "b": 2.0},
        stds={"a": 1.0, "b": 2.0},
    )
    combined_config = NetworkAndLossNormalizationConfig(
        network=network_config,
    )
    direct_normalizer = network_config.build(["a", "b"])
    loss_normalizer = combined_config.get_loss_normalizer(
        names=["a", "b"],
        residual_scaled_names=["a", "b"],
    )
    data = move_tensordict_to_device({"a": torch.randn(10), "b": torch.randn(10)})
    direct_normalized = direct_normalizer.normalize(data)
    loss_normalized = loss_normalizer.normalize(data)
    torch.testing.assert_close(direct_normalized["a"], loss_normalized["a"])
    torch.testing.assert_close(direct_normalized["b"], loss_normalized["b"])


@pytest.mark.parametrize("are_prognostic", [True, False])
def test_combined_normalization_uses_loss_normalizer_for_loss(are_prognostic: bool):
    torch.manual_seed(0)
    network_config = NormalizationConfig(
        means={"a": torch.randn(1), "b": torch.randn(1)},
        stds={"a": torch.randn(1), "b": torch.randn(1)},
    )
    loss_config = NormalizationConfig(
        means={"a": torch.randn(1), "b": torch.randn(1)},
        stds={"a": torch.randn(1), "b": torch.randn(1)},
    )
    combined_config = NetworkAndLossNormalizationConfig(
        network=network_config,
        loss=loss_config,
    )
    direct_normalizer = loss_config.build(["a", "b"])
    if are_prognostic:
        prognostic_names = ["a", "b"]
    else:
        prognostic_names = []
    loss_normalizer = combined_config.get_loss_normalizer(
        names=["a", "b"],
        residual_scaled_names=prognostic_names,
    )
    data = move_tensordict_to_device({"a": torch.randn(10), "b": torch.randn(10)})
    direct_normalized = direct_normalizer.normalize(data)
    loss_normalized = loss_normalizer.normalize(data)
    torch.testing.assert_close(direct_normalized["a"], loss_normalized["a"])
    torch.testing.assert_close(direct_normalized["b"], loss_normalized["b"])


def test_combined_normalization_uses_residual_normalizer_for_prognostic_loss():
    torch.manual_seed(0)
    network_config = NormalizationConfig(
        means={"a": torch.randn(1), "b": torch.randn(1)},
        stds={"a": torch.randn(1), "b": torch.randn(1)},
    )
    residual_config = NormalizationConfig(
        means={"a": torch.randn(1), "b": torch.randn(1)},
        stds={"a": torch.randn(1), "b": torch.randn(1)},
    )
    combined_config = NetworkAndLossNormalizationConfig(
        network=network_config,
        residual=residual_config,
    )
    direct_residual_normalizer = residual_config.build(["a", "b"])
    direct_network_normalizer = network_config.build(["a", "b"])
    loss_normalizer = combined_config.get_loss_normalizer(
        names=["a", "b"],
        residual_scaled_names=["a"],
    )
    data = move_tensordict_to_device({"a": torch.randn(10), "b": torch.randn(10)})
    direct_residual_normalized = direct_residual_normalizer.normalize(data)
    direct_network_noramlized = direct_network_normalizer.normalize(data)
    loss_normalized = loss_normalizer.normalize(data)
    torch.testing.assert_close(direct_residual_normalized["a"], loss_normalized["a"])
    torch.testing.assert_close(direct_network_noramlized["b"], loss_normalized["b"])


def test_normalization_config_with_means_and_stds_round_trip():
    config = NormalizationConfig(
        means={"a": 1.0, "b": 2.0},
        stds={"a": 1.0, "b": 2.0},
    )
    round_tripped = dacite.from_dict(
        NormalizationConfig,
        data=dataclasses.asdict(config),
        config=dacite.Config(
            strict=True,
        ),
    )
    assert config == round_tripped


def test__combine_normalizers():
    vars = ["prog_0", "prog_1", "diag_0"]
    full_field_normalizer = StandardNormalizer(
        means={var: torch.rand(3) for var in vars},
        stds={var: torch.rand(3) for var in vars},
        fill_nans_on_normalize=True,
        fill_nans_on_denormalize=True,
    )
    residual_normalizer = StandardNormalizer(
        means={var: torch.rand(3) for var in ["prog_0", "prog_1"]},
        stds={var: torch.rand(3) for var in ["prog_0", "prog_1"]},
    )
    combined_normalizer = _combine_normalizers(
        override_normalizer=residual_normalizer,
        base_normalizer=full_field_normalizer,
    )
    assert combined_normalizer.fill_nans_on_normalize
    assert combined_normalizer.fill_nans_on_denormalize
    for var in combined_normalizer.means:
        if "prog" in var:
            assert torch.allclose(
                combined_normalizer.means[var], residual_normalizer.means[var]
            )
            assert torch.allclose(
                combined_normalizer.stds[var], residual_normalizer.stds[var]
            )
        else:
            assert torch.allclose(
                combined_normalizer.means[var], full_field_normalizer.means[var]
            )
            assert torch.allclose(
                combined_normalizer.stds[var], full_field_normalizer.stds[var]
            )


def test_build_from_files():
    mean_ds = get_scalar_dataset(["a", "b", "c"], fill_value=1.0)
    std_ds = get_scalar_dataset(["a", "b", "c"], fill_value=2.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        mean_ds.to_netcdf(tmp_path / "mean.nc")
        std_ds.to_netcdf(tmp_path / "std.nc")
        normalizer = NormalizationConfig(
            global_means_path=tmp_path / "mean.nc",
            global_stds_path=tmp_path / "std.nc",
        ).build(["a", "b"])
        for name in ["a", "b"]:
            assert normalizer.means[name] == 1.0
            assert normalizer.stds[name] == 2.0
        assert "c" not in normalizer.means
        assert "c" not in normalizer.stds


@pytest.mark.parametrize("fill_nans_on_normalize", [True, False])
@pytest.mark.parametrize("fill_nans_on_denormalize", [True, False])
def test_load_from_files(fill_nans_on_normalize: bool, fill_nans_on_denormalize: bool):
    mean_ds = get_scalar_dataset(["a", "b", "c"], fill_value=1.0)
    std_ds = get_scalar_dataset(["a", "b", "c"], fill_value=2.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        mean_ds.to_netcdf(tmp_path / "mean.nc")
        std_ds.to_netcdf(tmp_path / "std.nc")
        config = NormalizationConfig(
            global_means_path=tmp_path / "mean.nc",
            global_stds_path=tmp_path / "std.nc",
            fill_nans_on_normalize=fill_nans_on_normalize,
            fill_nans_on_denormalize=fill_nans_on_denormalize,
        )
        config.load()
    assert config.fill_nans_on_normalize == fill_nans_on_normalize
    assert config.fill_nans_on_denormalize == fill_nans_on_denormalize
    normalizer = config.build(["a", "b"])
    for name in ["a", "b"]:
        assert normalizer.means[name] == 1.0
        assert normalizer.stds[name] == 2.0
    assert "c" not in normalizer.means
    assert "c" not in normalizer.stds


def test_cannot_build_without_load_or_files():
    mean_ds = get_scalar_dataset(["a", "b", "c"], fill_value=1.0)
    std_ds = get_scalar_dataset(["a", "b", "c"], fill_value=2.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        mean_ds.to_netcdf(tmp_path / "mean.nc")
        std_ds.to_netcdf(tmp_path / "std.nc")
        config = NormalizationConfig(
            global_means_path=tmp_path / "mean.nc",
            global_stds_path=tmp_path / "std.nc",
        )
    with pytest.raises(FileNotFoundError):
        config.build(["a", "b"])


def test_cannot_load_without_files():
    mean_ds = get_scalar_dataset(["a", "b", "c"], fill_value=1.0)
    std_ds = get_scalar_dataset(["a", "b", "c"], fill_value=2.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        mean_ds.to_netcdf(tmp_path / "mean.nc")
        std_ds.to_netcdf(tmp_path / "std.nc")
        config = NormalizationConfig(
            global_means_path=tmp_path / "mean.nc",
            global_stds_path=tmp_path / "std.nc",
        )
    with pytest.raises(FileNotFoundError):
        config.load()


def test_can_create_config_without_files():
    NormalizationConfig(
        global_means_path="/not/a/real/path",
        global_stds_path="/not/a/real/path",
    )


def _grouped_config(
    default_label: str = "era5",
    pinned_variables: list[str] | None = None,
) -> GroupedNormalizationConfig:
    """Two groups whose constants differ, for testing per-sample selection."""
    return GroupedNormalizationConfig(
        groups={
            "c96": NormalizationConfig(
                means={"a": 10.0, "pinned": 100.0},
                stds={"a": 2.0, "pinned": 200.0},
            ),
            "era5": NormalizationConfig(
                means={"a": 20.0, "pinned": 300.0},
                stds={"a": 4.0, "pinned": 400.0},
            ),
        },
        default_label=default_label,
        pinned_variables=pinned_variables if pinned_variables is not None else [],
    )


def _pooled_normalizer() -> StandardNormalizer:
    return NormalizationConfig(
        means={"a": 0.0, "pinned": 1.0},
        stds={"a": 1.0, "pinned": 3.0},
    ).build(["a", "pinned"])


def _build(
    config: GroupedNormalizationConfig, n_spatial_dims: int = 2
) -> GroupedNormalizer:
    """Build against the pooled normalizer, on a dataset carrying every label."""
    return config.build(
        pooled=_pooled_normalizer(),
        names=["a", "pinned"],
        n_spatial_dims=n_spatial_dims,
        dataset_labels=set(config.groups),
    )


def _tensor(data) -> torch.Tensor:
    """A tensor on the device the grouped normalizer's constants live on."""
    return torch.tensor(data, device=get_device())


def _labels(names: list[str], rows: list[list[float]]) -> BatchLabels:
    return BatchLabels(tensor=_tensor(rows), names=names)


def test_grouped_normalizer_uses_per_sample_constants():
    """Samples from different groups are normalized by their own constants."""
    grouped = _build(_grouped_config())
    # Sample 0 is c96 (mean 10, std 2), sample 1 is era5 (mean 20, std 4).
    labels = _labels(["c96", "era5"], [[1.0, 0.0], [0.0, 1.0]])
    normalizer = grouped.bind(labels)
    tensors = {"a": _tensor([[[12.0]], [[24.0]]])}
    normalized = normalizer.normalize(tensors)
    torch.testing.assert_close(normalized["a"], _tensor([[[1.0]], [[1.0]]]))


def test_grouped_normalizer_roundtrips_mixed_group_batch():
    grouped = _build(_grouped_config())
    labels = _labels(["c96", "era5"], [[1, 0], [0, 1], [1, 0]])
    normalizer = grouped.bind(labels)
    tensors = {
        "a": _tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]),
        "pinned": _tensor([[[7.0, 8.0]], [[9.0, 10.0]], [[11.0, 12.0]]]),
    }
    roundtripped = normalizer.denormalize(normalizer.normalize(tensors))
    for name, value in tensors.items():
        torch.testing.assert_close(roundtripped[name], value)


def test_grouped_normalizer_fills_denormalized_nans_with_group_means():
    """NaN filling on denormalize uses each sample's own group mean."""
    pooled = NormalizationConfig(
        means={"a": 0.0, "pinned": 1.0},
        stds={"a": 1.0, "pinned": 3.0},
        fill_nans_on_denormalize=True,
    ).build(["a", "pinned"])
    config = _grouped_config()
    grouped = config.build(
        pooled=pooled,
        names=["a", "pinned"],
        n_spatial_dims=2,
        dataset_labels=set(config.groups),
    )
    labels = _labels(["c96", "era5"], [[1.0, 0.0], [0.0, 1.0]])
    nan = float("nan")
    denormalized = grouped.bind(labels).denormalize(
        {"a": _tensor([[[nan, 0.0]], [[nan, 0.0]]])}
    )
    # c96 mean 10, era5 mean 20; the non-NaN cells denormalize to the same.
    torch.testing.assert_close(
        denormalized["a"], _tensor([[[10.0, 10.0]], [[20.0, 20.0]]])
    )


def test_scalar_normalizer_serializes_to_equivalent_normalizer():
    """get_state and get_normalization_config round-trip scalar constants."""
    normalizer = NormalizationConfig(
        means={"a": 1.5, "b": -2.0},
        stds={"a": 3.0, "b": 0.5},
        fill_nans_on_denormalize=True,
    ).build(["a", "b"])
    for reloaded in (
        StandardNormalizer.from_state(normalizer.get_state()),
        normalizer.get_normalization_config().build(["a", "b"]),
    ):
        assert reloaded.fill_nans_on_denormalize
        assert not reloaded.fill_nans_on_normalize
        for name in ["a", "b"]:
            assert float(reloaded.means[name]) == float(normalizer.means[name])
            assert float(reloaded.stds[name]) == float(normalizer.stds[name])


def test_bound_grouped_normalizer_cannot_be_serialized():
    """A bound normalizer's per-sample constants are batch state, not config."""
    grouped = _build(_grouped_config())
    normalizer = grouped.bind(_labels(["c96", "era5"], [[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="per-sample normalization means"):
        normalizer.get_state()
    with pytest.raises(ValueError, match="per-sample normalization means"):
        normalizer.get_normalization_config()


def test_grouped_normalizer_pinned_variable_uses_pooled_constants():
    """A pinned variable is normalized identically regardless of a sample's group."""
    grouped = _build(_grouped_config(pinned_variables=["pinned"]))
    labels = _labels(["c96", "era5"], [[1.0, 0.0], [0.0, 1.0]])
    normalizer = grouped.bind(labels)
    # Pooled constants for "pinned" are mean 1, std 3.
    tensors = {"pinned": _tensor([[[4.0]], [[4.0]]])}
    normalized = normalizer.normalize(tensors)
    torch.testing.assert_close(
        normalized["pinned"],
        _tensor([[[1.0]], [[1.0]]]),
    )
    # The non-pinned variable still varies by group, so pinning is selective.
    assert not torch.equal(normalizer.means["a"][0], normalizer.means["a"][1])


def test_grouped_normalizer_single_group_matches_standard_normalizer():
    """A degenerate one-group config reproduces plain pooled normalization.

    This is what makes it valid to skip the per-group arms of an ablation for
    regimes containing only one group.
    """
    means, stds = {"a": 10.0, "pinned": 100.0}, {"a": 2.0, "pinned": 200.0}
    standard = NormalizationConfig(means=means, stds=stds).build(["a", "pinned"])
    grouped = GroupedNormalizationConfig(
        groups={"era5": NormalizationConfig(means=means, stds=stds)},
        default_label="era5",
    ).build(
        pooled=standard,
        names=["a", "pinned"],
        n_spatial_dims=2,
        dataset_labels={"era5"},
    )
    labels = _labels(["era5"], [[1.0], [1.0]])
    tensors = {
        "a": _tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        "pinned": _tensor([[[5.0, 6.0]], [[7.0, 8.0]]]),
    }
    grouped_out = grouped.bind(labels).normalize(tensors)
    standard_out = standard.normalize(tensors)
    for name in tensors:
        torch.testing.assert_close(grouped_out[name], standard_out[name])


@pytest.mark.parametrize("labels", [None, "empty"])
def test_grouped_normalizer_falls_back_to_default_label_without_labels(labels):
    """An unlabeled batch uses the default group, not the pooled constants.

    A model trained on this normalizer never saw its network inputs on the
    pooled scale, so falling back to pooled would silently evaluate it against
    a distribution it was never trained on.
    """
    grouped = _build(_grouped_config(default_label="era5", pinned_variables=["pinned"]))
    batch_labels = None if labels is None else _labels([], [[], []])
    normalizer = grouped.bind(batch_labels)
    # era5 group constants for "a" are mean 20, std 4; "pinned" is pinned to
    # the pooled mean 1, std 3.
    tensors = {
        "a": _tensor([[[24.0]], [[24.0]]]),
        "pinned": _tensor([[[4.0]], [[4.0]]]),
    }
    normalized = normalizer.normalize(tensors)
    for name in tensors:
        torch.testing.assert_close(
            normalized[name],
            _tensor([[[1.0]], [[1.0]]]),
        )


def test_grouped_normalizer_default_label_choice_changes_unlabeled_constants():
    """The default group is a real choice, not a formality."""
    tensors = {"a": _tensor([[[24.0]]])}
    era5 = _build(_grouped_config(default_label="era5"))
    c96 = _build(_grouped_config(default_label="c96"))
    assert not torch.equal(
        era5.bind(None).normalize(tensors)["a"], c96.bind(None).normalize(tensors)["a"]
    )


@pytest.mark.parametrize("labeled, n_warnings", [(False, 1), (True, 0)])
def test_grouped_normalizer_warns_once_for_unlabeled_batches(
    labeled: bool, n_warnings: int, caplog
):
    """Unlabeled batches warn once per normalizer, not once per step."""
    grouped = _build(_grouped_config(default_label="era5"))
    labels = _labels(["c96", "era5"], [[0.0, 1.0]]) if labeled else None
    with caplog.at_level(logging.WARNING):
        grouped.bind(labels)
        grouped.bind(labels)
    messages = [
        r.getMessage() for r in caplog.records if "default_label" in r.getMessage()
    ]
    assert len(messages) == n_warnings
    assert all("era5" in m for m in messages)


@pytest.mark.parametrize("n_spatial_dims", [2, 3])
def test_grouped_normalizer_broadcasts_against_n_spatial_dims(n_spatial_dims):
    """Per-sample constants broadcast against however many spatial dims exist.

    HEALPix data carries a leading face dimension, so a hard-coded rank would
    align the sample dimension with the faces.
    """
    grouped = _build(_grouped_config(), n_spatial_dims=n_spatial_dims)
    labels = _labels(["c96", "era5"], [[1.0, 0.0], [0.0, 1.0]])
    normalizer = grouped.bind(labels)
    assert normalizer.means["a"].shape == (2, *(1,) * n_spatial_dims)
    # Sample 0 is c96 (mean 10, std 2), sample 1 is era5 (mean 20, std 4).
    spatial = (3,) * n_spatial_dims
    tensors = {
        "a": torch.stack(
            [
                torch.full(spatial, 12.0, device=get_device()),
                torch.full(spatial, 24.0, device=get_device()),
            ]
        )
    }
    normalized = normalizer.normalize(tensors)
    torch.testing.assert_close(
        normalized["a"], torch.ones(2, *spatial, device=get_device())
    )


@pytest.mark.parametrize(
    "names, rows, match",
    [
        (["c96", "era5"], [[1.0, 1.0]], "other than exactly one"),
        (["c96", "era5"], [[0.0, 0.0]], "other than exactly one"),
        (["mystery"], [[1.0]], "have no normalization group"),
    ],
    ids=["two_labels", "no_label", "unknown_label"],
)
def test_grouped_normalizer_rejects_unresolvable_labels(
    names: list[str], rows: list[list[float]], match: str
):
    grouped = _build(_grouped_config())
    with pytest.raises(ValueError, match=match):
        grouped.bind(_labels(names, rows))


def test_grouped_normalizer_requires_every_group_to_cover_every_variable():
    config = GroupedNormalizationConfig(
        groups={
            "c96": NormalizationConfig(
                means={"a": 1.0, "b": 1.0}, stds={"a": 1.0, "b": 1.0}
            ),
            "era5": NormalizationConfig(means={"a": 1.0}, stds={"a": 1.0}),
        },
        default_label="era5",
    )
    pooled = NormalizationConfig(
        means={"a": 0.0, "b": 0.0}, stds={"a": 1.0, "b": 1.0}
    ).build(["a", "b"])
    with pytest.raises(
        ValueError, match=r"'era5' has no constants for variables \['b'\]"
    ):
        config.build(
            pooled=pooled,
            names=["a", "b"],
            n_spatial_dims=2,
            dataset_labels={"c96", "era5"},
        )


def test_grouped_normalizer_does_not_require_groups_to_cover_pinned_variables():
    """A pinned variable uses the pooled constants, so groups may omit it."""
    config = GroupedNormalizationConfig(
        groups={"era5": NormalizationConfig(means={"a": 1.0}, stds={"a": 2.0})},
        default_label="era5",
        pinned_variables=["pinned"],
    )
    grouped = _build(config)
    normalizer = grouped.bind(_labels(["era5"], [[1.0]]))
    normalized = normalizer.normalize(
        {"a": _tensor([[[5.0]]]), "pinned": _tensor([[[4.0]]])}
    )
    # "a" uses the group (1, 2); "pinned" the pooled (1, 3).
    torch.testing.assert_close(normalized["a"], _tensor([[[2.0]]]))
    torch.testing.assert_close(normalized["pinned"], _tensor([[[1.0]]]))


def _config_with_constant_group(pinned_variables: list[str]):
    """Groups in which "pinned" is constant within the c96 group."""
    config = _grouped_config(pinned_variables=pinned_variables)
    config.groups["c96"] = NormalizationConfig(
        means={"a": 10.0, "pinned": 100.0}, stds={"a": 2.0, "pinned": 0.0}
    )
    return config


def test_grouped_normalizer_rejects_variable_constant_within_a_group():
    """A variable that should have been pinned fails at build, not in training.

    Its per-group std is ~0, so normalizing it per group would blow up the
    network's input. Pinning it resolves the error.
    """
    with pytest.raises(ValueError, match=r"Add it to pinned_variables.*") as err:
        _build(_config_with_constant_group(pinned_variables=[]))
    assert "'pinned'" in str(err.value)
    assert "['c96']" in str(err.value)
    _build(_config_with_constant_group(pinned_variables=["pinned"]))


@pytest.mark.parametrize("group_labels", [["era5"], ["c96", "era5"]])
def test_grouped_normalizer_accepts_variable_constant_in_pooled_stats(group_labels):
    """A variable constant everywhere is not a grouping problem.

    Its pooled std is also 0, so pinning would not help; plain pooled
    normalization would behave the same.
    """
    means, stds = {"a": 10.0, "c": 5.0}, {"a": 2.0, "c": 0.0}
    config = GroupedNormalizationConfig(
        groups={
            label: NormalizationConfig(means=means, stds=stds) for label in group_labels
        },
        default_label="era5",
    )
    config.build(
        pooled=NormalizationConfig(means=means, stds=stds).build(["a", "c"]),
        names=["a", "c"],
        n_spatial_dims=2,
        dataset_labels=set(group_labels),
    )


@pytest.mark.parametrize(
    "dataset_labels, match",
    [
        (set(), "requires a labeled dataset"),
        ({"c96", "era5", "mystery"}, r"\['mystery'\] have no normalization group"),
    ],
    ids=["unlabeled", "label_outside_every_group"],
)
def test_grouped_config_rejects_dataset_labels_that_cannot_select_a_group(
    dataset_labels: set[str], match: str
):
    with pytest.raises(ValueError, match=match):
        _grouped_config().build(
            pooled=_pooled_normalizer(),
            names=["a", "pinned"],
            n_spatial_dims=2,
            dataset_labels=dataset_labels,
        )


@pytest.mark.parametrize(
    "on_normalize, on_denormalize",
    [(True, False), (False, True)],
    ids=["normalize", "denormalize"],
)
def test_normalization_group_rejects_nan_filling(
    on_normalize: bool, on_denormalize: bool
):
    """A group cannot set NaN filling, which is taken from the pooled config."""
    with pytest.raises(ValueError, match="NaN filling is not supported"):
        GroupedNormalizationConfig(
            groups={
                "era5": NormalizationConfig(
                    means={"a": 0.0},
                    stds={"a": 1.0},
                    fill_nans_on_normalize=on_normalize,
                    fill_nans_on_denormalize=on_denormalize,
                ),
            },
            default_label="era5",
        )


def test_grouped_config_rejects_unknown_default_label():
    with pytest.raises(ValueError, match="default_label"):
        GroupedNormalizationConfig(
            groups={"c96": NormalizationConfig(means={"a": 1.0}, stds={"a": 1.0})},
            default_label="era5",
        )


def test_grouped_config_rejects_empty_groups():
    with pytest.raises(ValueError, match="At least one normalization group"):
        GroupedNormalizationConfig(groups={}, default_label="era5")


def test_grouped_config_rejects_pinned_variable_that_is_not_normalized():
    """A typo'd pinned name is rejected rather than silently un-pinning."""
    _grouped_config(pinned_variables=["pinned"]).validate_pinned_variables(
        ["a", "pinned"]
    )
    config = _grouped_config(pinned_variables=["pinnd"])
    with pytest.raises(ValueError, match="are not normalized variables"):
        config.validate_pinned_variables(["a", "pinned"])


def test_grouped_normalization_loads_into_explicit_constants():
    """Group constants are inlined at load, as for the pooled constants."""
    mean_ds = get_scalar_dataset(["a"], fill_value=1.0)
    std_ds = get_scalar_dataset(["a"], fill_value=2.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        mean_ds.to_netcdf(tmp_path / "mean.nc")
        std_ds.to_netcdf(tmp_path / "std.nc")
        config = NetworkAndLossNormalizationConfig(
            network=NormalizationConfig(
                global_means_path=tmp_path / "mean.nc",
                global_stds_path=tmp_path / "std.nc",
            ),
            grouped=GroupedNormalizationConfig(
                groups={
                    "era5": NormalizationConfig(
                        global_means_path=tmp_path / "mean.nc",
                        global_stds_path=tmp_path / "std.nc",
                    )
                },
                default_label="era5",
            ),
        )
        config.load()
    assert config.grouped is not None
    group = config.grouped.groups["era5"]
    assert group.global_means_path is None
    assert group.means["a"] == 1.0
    # The config no longer depends on the netCDF files, which have been removed.
    config.grouped.build(
        pooled=config.get_network_normalizer(["a"]),
        names=["a"],
        n_spatial_dims=2,
        dataset_labels={"era5"},
    )


N_LAT, N_LON = 3, 4
LAT, LON = np.linspace(-60.0, 60.0, N_LAT), 90.0 * np.arange(N_LON)


def _cell_values(seed: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """Distinct [N_LAT, N_LON] values, for per-grid-cell constants."""
    generator = torch.Generator().manual_seed(seed)
    return low + (high - low) * torch.rand(N_LAT, N_LON, generator=generator)


def _per_grid_cell_file_config(
    tmp_path: pathlib.Path,
    means: dict[str, float | torch.Tensor],
    stds: dict[str, float | torch.Tensor],
    encoding: dict[str, dict] | None = None,
    **kwargs,
) -> NormalizationConfig:
    """Config reading constants on (lat, lon) from netCDF files."""
    coords = {"lat": LAT, "lon": LON}
    return NormalizationConfig(
        global_means_path=write_normalization_stats(
            tmp_path / "means.nc", means, coords, encoding=encoding
        ),
        global_stds_path=write_normalization_stats(
            tmp_path / "stds.nc", stds, coords, encoding=encoding
        ),
        **kwargs,
    )


def _on_device(tensor: torch.Tensor | float) -> torch.Tensor:
    return torch.as_tensor(tensor).to(get_device())


@pytest.mark.parametrize("load", [False, True], ids=["from_files", "loaded"])
def test_per_grid_cell_normalization_uses_each_cells_constants(tmp_path, load):
    """Per-grid-cell constants broadcast against [batch, (time,) lat, lon]
    tensors, mixed with scalar constants in the same files."""
    means, stds = _cell_values(0), _cell_values(1, low=1.0, high=2.0)
    config = _per_grid_cell_file_config(
        tmp_path,
        means={"a": means, "b": 3.0},
        stds={"a": stds, "b": 2.0},
        per_grid_cell=True,
    )
    if load:
        config.load()
    normalizer = config.build(["a", "b"])
    for shape in [(2, N_LAT, N_LON), (2, 5, N_LAT, N_LON)]:
        tensors = {name: _on_device(torch.randn(shape)) for name in ["a", "b"]}
        normalized = normalizer.normalize(tensors)
        torch.testing.assert_close(
            normalized["a"], (tensors["a"] - _on_device(means)) / _on_device(stds)
        )
        torch.testing.assert_close(normalized["b"], (tensors["b"] - 3.0) / 2.0)
        roundtripped = normalizer.denormalize(normalized)
        for name, value in tensors.items():
            torch.testing.assert_close(roundtripped[name], value)


def test_per_grid_cell_constants_require_opt_in(tmp_path):
    """Without per_grid_cell, non-scalar variables are skipped when loading
    every variable and rejected when requested or given explicitly."""
    config = _per_grid_cell_file_config(
        tmp_path,
        means={"a": _cell_values(0), "b": 3.0},
        stds={"a": _cell_values(1, low=1.0, high=2.0), "b": 2.0},
    )
    with pytest.raises(ValueError, match="set per_grid_cell"):
        config.build(["a", "b"])
    config.load()
    assert "a" not in config.means
    assert config.means["b"] == 3.0
    with pytest.raises(ValueError, match="set per_grid_cell"):
        NormalizationConfig(means={"a": _cell_values(0)}, stds={"a": 1.0})


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_per_grid_cell_min_std_fraction_must_be_a_fraction(fraction):
    with pytest.raises(ValueError, match="per_grid_cell_min_std_fraction"):
        NormalizationConfig(
            means={"a": 0.0}, stds={"a": 1.0}, per_grid_cell_min_std_fraction=fraction
        )


def test_per_grid_cell_stds_are_floored(tmp_path, caplog):
    """Zero and near-zero stds are floored to a fraction of the mean over
    finite, non-negative cells; NaN and negative cells are left to be
    rejected on build, and scalar stds are not floored."""
    stds = torch.full((N_LAT, N_LON), 2.0)
    stds[0, 0] = float("nan")
    stds[0, 1] = -0.5
    stds[1, 1] = 0.0
    stds[1, 2] = 1e-9
    config = _per_grid_cell_file_config(
        tmp_path,
        means={"a": torch.zeros(N_LAT, N_LON), "b": 0.0},
        stds={"a": stds, "b": 0.0},
        per_grid_cell=True,
        per_grid_cell_min_std_fraction=0.5,
    )
    with caplog.at_level(logging.INFO):
        config.load()
    # Half the mean of the 10 finite, non-negative cells' stds.
    floor = 0.5 * 16.0 / 10.0
    assert f"'a' to {floor:.6g} in 2 of 12 grid cells" in caplog.text
    floored = config.stds["a"]
    assert isinstance(floored, torch.Tensor)
    assert float(floored[1, 1]) == pytest.approx(floor)
    assert float(floored[1, 2]) == pytest.approx(floor)
    assert float(floored[2, 2]) == 2.0
    assert float(floored[0, 1]) == -0.5
    assert torch.isnan(floored[0, 0])
    assert config.stds["b"] == 0.0
    with pytest.raises(ValueError, match="stds for 'a' are non-finite in 1 of 12"):
        config.build(["a"])


@pytest.mark.parametrize("value", [0.0, -0.5])
def test_per_grid_cell_non_positive_std_cell_is_rejected(value):
    stds = torch.ones(N_LAT, N_LON)
    stds[1, 2] = value
    config = NormalizationConfig(
        means={"a": torch.zeros(N_LAT, N_LON)}, stds={"a": stds}, per_grid_cell=True
    )
    with pytest.raises(ValueError, match="not positive in every grid cell"):
        config.build(["a"])


@pytest.mark.parametrize("attribute", [None, "_FillValue", "missing_value"])
@pytest.mark.parametrize("constant", ["means", "stds"])
def test_per_grid_cell_missing_cells_are_rejected_as_non_finite(
    tmp_path, attribute, constant
):
    """A NaN or fill-value cell (e.g. an ocean variable over land) is
    rejected rather than used as a mean or std of 1e20."""
    constants = {"means": _cell_values(0), "stds": _cell_values(1, 1.0, 2.0)}
    constants[constant][0, 0] = float("nan")
    config = _per_grid_cell_file_config(
        tmp_path,
        means={"a": constants["means"]},
        stds={"a": constants["stds"]},
        encoding=None if attribute is None else {"a": {attribute: 1e20}},
        per_grid_cell=True,
    )
    with pytest.raises(ValueError, match=rf"{constant} for 'a' are non-finite in 1 of"):
        config.build(["a"])


def test_per_grid_cell_packed_constants_and_coordinates_are_unpacked(tmp_path):
    means, stds = _cell_values(0), _cell_values(1, 1.0, 2.0)
    config = _per_grid_cell_file_config(
        tmp_path,
        means={"a": means},
        stds={"a": stds},
        encoding={
            "a": {"dtype": "int16", "scale_factor": 0.001, "add_offset": 1.0},
            "lat": {"dtype": "int16", "scale_factor": 0.5},
        },
        per_grid_cell=True,
    )
    normalizer = config.build(["a"])
    torch.testing.assert_close(
        normalizer.means["a"], _on_device(means), atol=1e-3, rtol=0
    )
    torch.testing.assert_close(
        normalizer.stds["a"], _on_device(stds), atol=1e-3, rtol=0
    )
    normalizer.validate_spatial_shape((N_LAT, N_LON), (LAT, LON))


def test_per_grid_cell_constant_with_time_like_units_is_not_decoded(tmp_path):
    means = _cell_values(0)
    xr.Dataset(
        {"a": (("lat", "lon"), means.numpy(), {"units": "days since 2000-01-01"})},
        coords={"lat": LAT, "lon": LON},
    ).to_netcdf(tmp_path / "means.nc")
    config = NormalizationConfig(
        global_means_path=tmp_path / "means.nc",
        global_stds_path=write_normalization_stats(
            tmp_path / "stds.nc",
            {"a": np.ones((N_LAT, N_LON))},
            {"lat": LAT, "lon": LON},
        ),
        per_grid_cell=True,
    )
    normalizer = config.build(["a"])
    torch.testing.assert_close(normalizer.means["a"], _on_device(means))


def test_per_grid_cell_constants_round_trip_through_serialization(tmp_path):
    """Per-grid-cell constants survive both checkpoint paths: the loaded
    config (through asdict and dacite) and the normalizer state."""
    config = _per_grid_cell_file_config(
        tmp_path,
        means={"a": _cell_values(0), "b": 3.0},
        stds={"a": _cell_values(1, low=1.0, high=2.0), "b": 2.0},
        per_grid_cell=True,
    )
    config.load()
    normalizer = config.build(["a", "b"])
    for path in tmp_path.glob("*.nc"):
        path.unlink()
    reloaded_config = dacite.from_dict(
        NormalizationConfig,
        dataclasses.asdict(config),
        config=dacite.Config(strict=True),
    )
    for reloaded in (
        reloaded_config.build(["a", "b"]),
        StandardNormalizer.from_state(normalizer.get_state()),
        normalizer.get_normalization_config().build(["a", "b"]),
    ):
        for name in ["a", "b"]:
            torch.testing.assert_close(reloaded.means[name], normalizer.means[name])
            torch.testing.assert_close(reloaded.stds[name], normalizer.stds[name])
        assert reloaded.grid_coords == {"a": {"lat": LAT.tolist(), "lon": LON.tolist()}}
    assert normalizer.get_normalization_config().per_grid_cell
    assert normalizer.get_state()["means"]["b"] == 3.0


def test_per_grid_cell_denormalize_fills_nans_with_cell_means():
    means = _cell_values(0)
    normalizer = NormalizationConfig(
        means={"a": means},
        stds={"a": torch.ones(N_LAT, N_LON)},
        per_grid_cell=True,
        fill_nans_on_denormalize=True,
    ).build(["a"])
    nans = _on_device(torch.full((2, N_LAT, N_LON), float("nan")))
    denormalized = normalizer.denormalize({"a": nans})
    torch.testing.assert_close(
        denormalized["a"], _on_device(means).expand(2, N_LAT, N_LON)
    )


def test_validate_spatial_shape():
    normalizer = NormalizationConfig(
        means={"a": _cell_values(0), "b": 0.0},
        stds={"a": torch.ones(N_LAT, N_LON), "b": 1.0},
        per_grid_cell=True,
    ).build(["a", "b"])
    assert normalizer.is_per_grid_cell
    normalizer.validate_spatial_shape((N_LAT, N_LON))
    with pytest.raises(ValueError, match="does not match the data's spatial shape"):
        normalizer.validate_spatial_shape((N_LON, N_LAT))
    scalar = NormalizationConfig(means={"a": 0.0}, stds={"a": 1.0}).build(["a"])
    assert not scalar.is_per_grid_cell
    scalar.validate_spatial_shape((N_LON, N_LAT))


def _gridded_config(
    tmp_path: pathlib.Path,
    means_coords: dict[str, np.ndarray],
    stds_coords: dict[str, np.ndarray] | None = None,
) -> NormalizationConfig:
    shape = tuple(len(values) for values in means_coords.values())
    return NormalizationConfig(
        global_means_path=write_normalization_stats(
            tmp_path / "means.nc", {"a": np.zeros(shape)}, means_coords
        ),
        global_stds_path=write_normalization_stats(
            tmp_path / "stds.nc", {"a": np.ones(shape)}, stds_coords or means_coords
        ),
        per_grid_cell=True,
    )


SQUARE_LAT, SQUARE_LON = np.linspace(-67.5, 67.5, 4), 90.0 * np.arange(4)
SQUARE_LON_180 = np.where(SQUARE_LON >= 180.0, SQUARE_LON - 360.0, SQUARE_LON)


@pytest.mark.parametrize("load", [False, True], ids=["from_files", "loaded"])
@pytest.mark.parametrize(
    "coords, match",
    [
        ({"lat": SQUARE_LAT, "lon": SQUARE_LON}, None),
        ({"lat": SQUARE_LAT[::-1], "lon": SQUARE_LON}, "reversed"),
        ({"lon": SQUARE_LON, "lat": SQUARE_LAT}, "transposed"),
        ({"lat": SQUARE_LAT, "lon": SQUARE_LON - 180.0}, "do not match"),
        ({"lat": SQUARE_LAT, "lon": SQUARE_LON_180}, "0..360 vs -180..180"),
        ({"lat": SQUARE_LAT, "lon": np.sort(SQUARE_LON_180)}, "0..360 vs -180..180"),
        ({"lat": np.roll(SQUARE_LAT, 1), "lon": SQUARE_LON}, "dimension 'lat'"),
    ],
    ids=[
        "matching",
        "flipped_lat",
        "transposed",
        "shifted_lon",
        "lon_180",
        "rolled_lon_180",
        "rolled_lat",
    ],
)
def test_per_grid_cell_constants_are_checked_against_data_coords(
    tmp_path, load, coords, match
):
    """Constants on a square grid with the data's shape but flipped,
    transposed or shifted coordinates are rejected."""
    config = _gridded_config(tmp_path, coords)
    if load:
        config.load()
    normalizer = config.build(["a"])
    data_coords = (SQUARE_LAT.astype(np.float32), SQUARE_LON.astype(np.float32))
    if match is None:
        normalizer.validate_spatial_shape((4, 4), data_coords)
    else:
        with pytest.raises(ValueError, match=match):
            normalizer.validate_spatial_shape((4, 4), data_coords)
    # Without the data's coordinates, only the shape can be checked.
    normalizer.validate_spatial_shape((4, 4))


def test_per_grid_cell_means_and_stds_must_share_a_grid(tmp_path):
    config = _gridded_config(
        tmp_path,
        means_coords={"lat": SQUARE_LAT, "lon": SQUARE_LON},
        stds_coords={"lat": SQUARE_LAT[::-1], "lon": SQUARE_LON},
    )
    with pytest.raises(ValueError, match="different dimensions or coordinates"):
        config.load()


def test_per_grid_cell_coords_must_match_constants():
    means, stds = {"a": torch.zeros(N_LAT, N_LON)}, {"a": torch.ones(N_LAT, N_LON)}
    coords = {"lat": LAT.tolist(), "lon": LON.tolist()}
    # Coordinates may describe a variable which has only means.
    NormalizationConfig(
        means=means | {"b": torch.zeros(N_LAT, N_LON)},
        stds=stds,
        per_grid_cell=True,
        per_grid_cell_coords={"a": coords, "b": coords},
    )
    with pytest.raises(ValueError, match="'b', which has neither means nor stds"):
        NormalizationConfig(
            means=means,
            stds=stds,
            per_grid_cell=True,
            per_grid_cell_coords={"a": coords, "b": coords},
        )
    with pytest.raises(ValueError, match=r"means for 'a' have shape \(3, 4\)"):
        NormalizationConfig(
            means=means,
            stds=stds,
            per_grid_cell=True,
            per_grid_cell_coords={"a": {"lon": LON.tolist(), "lat": LAT.tolist()}},
        )


def test_per_grid_cell_coords_cannot_be_given_with_paths(tmp_path):
    with pytest.raises(ValueError, match="coordinates are read from the files"):
        NormalizationConfig(
            global_means_path=tmp_path / "means.nc",
            global_stds_path=tmp_path / "stds.nc",
            per_grid_cell=True,
            per_grid_cell_coords={"a": {"lat": LAT.tolist()}},
        )


def test_per_grid_cell_load_skips_unrelated_variables(tmp_path):
    """Loading every variable skips coordinates, non-numeric variables and
    those with a dimension lacking coordinates, and keeps unused bad fields,
    which are rejected only if requested."""
    datasets = {}
    for kind, fill in [("means", 0.0), ("stds", 1.0)]:
        datasets[kind] = xr.Dataset(
            {
                "a": (("lat", "lon"), np.full((N_LAT, N_LON), fill)),
                "unused_nan": (("lat", "lon"), np.full((N_LAT, N_LON), np.nan)),
                "unused_zero": (("lat", "lon"), np.zeros((N_LAT, N_LON))),
                "lat_bnds": (("lat", "nv"), np.zeros((N_LAT, 2))),
                "valid_time": (
                    ("lat", "lon"),
                    np.full((N_LAT, N_LON), np.datetime64("2000-01-01", "ns")),
                ),
                "reference_time": ((), np.datetime64("2000-01-01", "ns")),
            },
            coords={"lat": LAT, "lon": LON},
        )
        datasets[kind].to_netcdf(tmp_path / f"{kind}.nc")
    config = NormalizationConfig(
        global_means_path=tmp_path / "means.nc",
        global_stds_path=tmp_path / "stds.nc",
        per_grid_cell=True,
    )
    config.load()
    assert set(config.means) == {"a", "unused_nan", "unused_zero", "x", "y", "z"}
    normalizer = config.build(["a"])
    torch.testing.assert_close(
        normalizer.stds["a"], _on_device(torch.ones(N_LAT, N_LON))
    )
    with pytest.raises(ValueError, match="non-finite"):
        config.build(["unused_nan"])
    with pytest.raises(ValueError, match="zero in every grid cell"):
        config.build(["unused_zero"])


def _healpix_config(tmp_path: pathlib.Path) -> NormalizationConfig:
    """Per-grid-cell "a" on index dimensions without coordinates, and a
    scalar "s"."""
    paths = {
        kind: write_normalization_stats(
            tmp_path / f"{kind}.nc",
            {"a": np.full((12, 2, 2), fill), "s": 1.0},
            dims=("face", "height", "width"),
        )
        for kind, fill in [("means", 0.0), ("stds", 1.0)]
    }
    return NormalizationConfig(
        global_means_path=paths["means"],
        global_stds_path=paths["stds"],
        per_grid_cell=True,
    )


def test_per_grid_cell_variable_without_coordinates(tmp_path):
    """Requested by name, it is reported; loaded without names, it is
    skipped, and building it then says why it may be missing."""
    match = "dimension 'face' without numeric coordinates"
    with pytest.raises(ValueError, match=match):
        _healpix_config(tmp_path).build(["a"])
    with pytest.raises(ValueError, match=match):
        _healpix_config(tmp_path).load(names=["a"])
    config = _healpix_config(tmp_path)
    config.load()
    with pytest.raises(ValueError, match=r"\['a'\].*numeric coordinates"):
        config.build(["a", "s"])


def test_per_grid_cell_load_by_name_loads_only_requested_per_cell_variables(
    tmp_path,
):
    """Only requested per-grid-cell variables are loaded, but every scalar;
    requested names missing from the files are left for build, and a second
    load is a no-op."""
    config = _per_grid_cell_file_config(
        tmp_path,
        means={"a": _cell_values(0), "unused": _cell_values(1), "s": 3.0},
        stds={"a": _cell_values(2, 1.0, 2.0), "unused": _cell_values(3), "s": 2.0},
        per_grid_cell=True,
    )
    config.load(names=["a", "not_in_files"])
    assert set(config.means) == {"a", "s", "x", "y", "z"}
    assert set(config.stds) == {"a", "s", "x", "y", "z"}
    assert set(config.per_grid_cell_coords) == {"a"}
    normalizer = config.build(["a", "s"])
    torch.testing.assert_close(normalizer.means["a"], _on_device(_cell_values(0)))
    assert float(normalizer.stds["s"]) == 2.0
    loaded = dataclasses.asdict(config)
    for path in tmp_path.glob("*.nc"):
        path.unlink()
    config.load(names=["unused"])
    config.load()
    assert set(config.means) == set(loaded["means"])
    assert set(config.per_grid_cell_coords) == {"a"}
    with pytest.raises(ValueError, match=r"\['unused'\].*only loaded if requested"):
        config.build(["unused"])


@pytest.mark.parametrize("shared_with", ["loss", "residual", "residual_no_loss"])
def test_config_shared_by_network_and_loss_loads_loss_only_names(
    tmp_path, shared_with: str
):
    """A per-grid-cell network config reused for the loss or residual (or
    read by the loss for lack of a loss block) loads the loss-only names."""
    shared = _per_grid_cell_file_config(
        tmp_path,
        means={"a": _cell_values(0), "b": _cell_values(1)},
        stds={"a": _cell_values(2, 1.0, 2.0), "b": _cell_values(3, 1.0, 2.0)},
        per_grid_cell=True,
    )
    scalar = NormalizationConfig(means={"a": 0.0, "b": 0.0}, stds={"a": 1.0, "b": 1.0})
    loss = {"loss": shared, "residual": scalar, "residual_no_loss": None}[shared_with]
    config = NetworkAndLossNormalizationConfig(
        network=shared,
        loss=loss,
        residual=None if shared_with == "loss" else shared,
    )
    config.load(names=["a"], extra_loss_names=["b"])
    residual_scaled_names = {"loss": [], "residual": ["b"], "residual_no_loss": ["a"]}
    normalizer = config.get_loss_normalizer(
        names=["a", "b"], residual_scaled_names=residual_scaled_names[shared_with]
    )
    torch.testing.assert_close(normalizer.means["b"], _on_device(_cell_values(1)))


def test_per_grid_cell_residual_with_means_only_variable_reloads(tmp_path):
    """A per-grid-cell residual block whose stds file lacks a variable in its
    means file survives a checkpoint's reload of its loaded config."""
    scalar = NormalizationConfig(
        means={"prog": 0.0, "forcing": 0.0}, stds={"prog": 1.0, "forcing": 1.0}
    )
    config = NetworkAndLossNormalizationConfig(
        network=scalar,
        loss=dataclasses.replace(scalar),
        residual=_per_grid_cell_file_config(
            tmp_path,
            means={"prog": _cell_values(0), "forcing": _cell_values(1)},
            stds={"prog": _cell_values(2, 1.0, 2.0)},
            per_grid_cell=True,
        ),
    )
    config.load(names=["prog", "forcing"])
    for path in tmp_path.glob("*.nc"):
        path.unlink()
    reloaded = dacite.from_dict(
        NetworkAndLossNormalizationConfig,
        dataclasses.asdict(config),
        config=dacite.Config(strict=True),
    )
    names, residual_names = ["prog", "forcing"], ["prog"]
    expected = config.get_loss_normalizer(names, residual_names)
    actual = reloaded.get_loss_normalizer(names, residual_names)
    for name in names:
        torch.testing.assert_close(actual.means[name], expected.means[name])
        torch.testing.assert_close(actual.stds[name], expected.stds[name])
    assert actual.grid_coords == {"prog": {"lat": LAT.tolist(), "lon": LON.tolist()}}


def test_loss_normalizer_combines_per_grid_cell_loss_and_residual():
    loss_stds = _cell_values(0, low=1.0, high=2.0)
    residual_stds = _cell_values(1, low=0.1, high=0.2)
    per_cell = NormalizationConfig(
        means={"a": 0.0, "b": _cell_values(2)},
        stds={"a": 1.0, "b": loss_stds},
        per_grid_cell=True,
    )
    config = NetworkAndLossNormalizationConfig(
        network=per_cell,
        loss=per_cell,
        residual=NormalizationConfig(
            means={"a": _cell_values(3)}, stds={"a": residual_stds}, per_grid_cell=True
        ),
    )
    loss_normalizer = config.get_loss_normalizer(
        names=["a", "b"], residual_scaled_names=["a"]
    )
    torch.testing.assert_close(loss_normalizer.stds["a"], _on_device(residual_stds))
    torch.testing.assert_close(loss_normalizer.stds["b"], _on_device(loss_stds))
    loss_normalizer.validate_spatial_shape((N_LAT, N_LON))


def test_loss_normalizer_keeps_grid_coords_of_overriding_constants(tmp_path):
    network = _gridded_config(tmp_path, {"lat": SQUARE_LAT, "lon": SQUARE_LON})
    network.load()
    config = NetworkAndLossNormalizationConfig(
        network=network,
        loss=network,
        residual=NormalizationConfig(means={"a": 0.0}, stds={"a": 2.0}),
    )
    assert config.get_loss_normalizer(["a"], []).grid_coords == {
        "a": {"lat": SQUARE_LAT.tolist(), "lon": SQUARE_LON.tolist()}
    }
    # The scalar residual constants replace the per-grid-cell ones.
    assert config.get_loss_normalizer(["a"], ["a"]).grid_coords == {}


def _per_grid_cell_group(seed: int) -> NormalizationConfig:
    return NormalizationConfig(
        means={"a": _cell_values(seed, low=-10.0, high=10.0), "pinned": 0.0},
        stds={"a": _cell_values(seed + 100, low=1.0, high=3.0), "pinned": 1.0},
        per_grid_cell=True,
    )


def _per_grid_cell_pooled() -> StandardNormalizer:
    return NormalizationConfig(
        means={"a": _cell_values(-1), "pinned": _cell_values(-2)},
        stds={
            "a": _cell_values(-3, low=1.0, high=2.0),
            "pinned": _cell_values(-4, low=1.0, high=2.0),
        },
        per_grid_cell=True,
    ).build(["a", "pinned"])


def _build_per_grid_cell_grouped(
    groups: dict[str, NormalizationConfig],
    pinned_variables: list[str] | None = None,
    pooled: StandardNormalizer | None = None,
) -> GroupedNormalizer:
    config = GroupedNormalizationConfig(
        groups=groups,
        default_label=sorted(groups)[0],
        pinned_variables=pinned_variables or [],
    )
    return config.build(
        pooled=pooled or _per_grid_cell_pooled(),
        names=["a", "pinned"],
        n_spatial_dims=2,
        dataset_labels=set(groups),
    )


def _one_sample_per_label(labels: list[str]) -> BatchLabels:
    return _labels(labels, torch.eye(len(labels)).tolist())


def test_grouped_per_grid_cell_normalizer_uses_group_cell_constants():
    """Non-pinned variables use each group's per-grid-cell constants, and
    pinned ones the pooled per-grid-cell constants."""
    labels = ["amip", "ramped", "som", "era5"]
    groups = {label: _per_grid_cell_group(seed) for seed, label in enumerate(labels)}
    pooled = _per_grid_cell_pooled()
    grouped = _build_per_grid_cell_grouped(groups, ["pinned"], pooled=pooled)
    assert grouped.is_per_grid_cell
    normalizer = grouped.bind(_one_sample_per_label(labels))
    x = _on_device(torch.randn(len(labels), N_LAT, N_LON))
    normalized = normalizer.normalize({"a": x, "pinned": x})
    for i, label in enumerate(labels):
        mean = _on_device(groups[label].means["a"])
        std = _on_device(groups[label].stds["a"])
        torch.testing.assert_close(normalized["a"][i], (x[i] - mean) / std)
    torch.testing.assert_close(
        normalized["pinned"], (x - pooled.means["pinned"]) / pooled.stds["pinned"]
    )
    torch.testing.assert_close(normalizer.denormalize(normalized)["a"], x)


def test_grouped_normalizer_mixes_scalar_and_per_grid_cell_groups():
    """A variable may be scalar in one group and per-grid-cell in another, on
    top of scalar pooled constants; every group's grid is validated."""
    groups = {
        "c96": NormalizationConfig(
            means={"a": 5.0, "pinned": 0.0}, stds={"a": 2.0, "pinned": 1.0}
        ),
        "era5": _per_grid_cell_group(0),
    }
    grouped = _build_per_grid_cell_grouped(groups, pooled=_pooled_normalizer())
    assert grouped.is_per_grid_cell
    grouped.validate_spatial_shape((N_LAT, N_LON))
    with pytest.raises(ValueError, match="does not match the data's spatial shape"):
        grouped.validate_spatial_shape((N_LON, N_LAT))
    normalizer = grouped.bind(_one_sample_per_label(["c96", "era5"]))
    with pytest.raises(ValueError, match="GroupedNormalizer"):
        normalizer.is_per_grid_cell
    x = _on_device(torch.randn(2, N_LAT, N_LON))
    normalized = normalizer.normalize({"a": x})["a"]
    torch.testing.assert_close(normalized[0], (x[0] - 5.0) / 2.0)
    era5 = groups["era5"]
    torch.testing.assert_close(
        normalized[1],
        (x[1] - _on_device(era5.means["a"])) / _on_device(era5.stds["a"]),
    )


def test_grouped_rejects_per_grid_cell_constants_of_different_shapes():
    other_grid = NormalizationConfig(
        means={"a": torch.zeros(N_LON, N_LAT), "pinned": 0.0},
        stds={"a": torch.ones(N_LON, N_LAT), "pinned": 1.0},
        per_grid_cell=True,
    )
    groups = {"c96": _per_grid_cell_group(0), "era5": other_grid}
    with pytest.raises(ValueError, match="different shapes"):
        _build_per_grid_cell_grouped(groups)


def test_grouped_degenerate_check_skips_only_cells_constant_when_pooled():
    """A cell constant in the pooled stats is skipped, but the group stds
    are still checked in every other cell."""
    pooled_stds = torch.ones(N_LAT, N_LON)
    pooled_stds[0, 0] = 0.0
    pooled = StandardNormalizer(
        means={"a": torch.zeros(N_LAT, N_LON), "pinned": torch.tensor(0.0)},
        stds={"a": pooled_stds, "pinned": torch.tensor(1.0)},
    )
    stds = _cell_values(100, low=1.0, high=3.0)
    stds[0, 0] = 0.0
    group_normalizers = {
        label: StandardNormalizer(
            means={"a": torch.zeros(N_LAT, N_LON), "pinned": torch.tensor(0.0)},
            stds={"a": group_stds, "pinned": torch.tensor(1.0)},
        )
        for label, group_stds in [("c96", stds), ("era5", torch.ones(N_LAT, N_LON))]
    }
    GroupedNormalizer(
        pooled=pooled, groups=group_normalizers, default_label="c96", n_spatial_dims=2
    )
    stds[1, 1] = 1e-8
    group_normalizers["c96"].stds["a"] = _on_device(stds)
    with pytest.raises(ValueError, match="pinned_variables"):
        GroupedNormalizer(
            pooled=pooled,
            groups=group_normalizers,
            default_label="c96",
            n_spatial_dims=2,
        )


def test_grouped_load_skips_pinned_per_grid_cell_constants(tmp_path):
    group = _per_grid_cell_file_config(
        tmp_path,
        means={"a": _cell_values(0), "pinned": _cell_values(1)},
        stds={"a": _cell_values(2, 1.0, 2.0), "pinned": _cell_values(3, 1.0, 2.0)},
        per_grid_cell=True,
    )
    config = GroupedNormalizationConfig(
        groups={"era5": group}, default_label="era5", pinned_variables=["pinned"]
    )
    config.load(names=["a", "pinned"])
    assert set(group.per_grid_cell_coords) == {"a"}
    assert "pinned" not in group.means
    grouped = config.build(
        pooled=_per_grid_cell_pooled(),
        names=["a", "pinned"],
        n_spatial_dims=2,
        dataset_labels={"era5"},
    )
    assert grouped.is_per_grid_cell


def test_raise_if_unsupported_names_the_offending_config():
    scalar = NormalizationConfig(means={"a": 0.0}, stds={"a": 1.0})
    per_cell = _per_grid_cell_group(0)
    NetworkAndLossNormalizationConfig(network=scalar).raise_if_unsupported("SomeStep")
    grouped = NetworkAndLossNormalizationConfig(
        network=scalar,
        grouped=GroupedNormalizationConfig(
            groups={"c96": scalar, "era5": per_cell}, default_label="c96"
        ),
    )
    with pytest.raises(ValueError, match="SomeStep does not support grouped"):
        grouped.raise_if_unsupported("SomeStep")
    for key in ["network", "loss", "residual"]:
        config = NetworkAndLossNormalizationConfig(network=scalar, loss=scalar)
        setattr(config, key, per_cell)
        with pytest.raises(ValueError, match=rf"SomeStep \({key} normalization\)"):
            config.raise_if_unsupported("SomeStep")

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
    with pytest.raises(KeyError):
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


def test_combined_normalization_cannot_set_both_loss_and_residual():
    network_config = NormalizationConfig(
        means={"a": torch.randn(1), "b": torch.randn(1)},
        stds={"a": torch.randn(1), "b": torch.randn(1)},
    )
    with pytest.raises(ValueError):
        NetworkAndLossNormalizationConfig(
            network=network_config,
            loss=network_config,
            residual=network_config,
        )


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
    with pytest.raises(ValueError, match="'era5' has no constants for variable 'b'"):
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


def _cell_values(seed: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """Distinct [N_LAT, N_LON] values, for per-grid-cell constants."""
    generator = torch.Generator().manual_seed(seed)
    return low + (high - low) * torch.rand(N_LAT, N_LON, generator=generator)


def _write_constants(
    path: pathlib.Path, constants: dict[str, float | torch.Tensor]
) -> pathlib.Path:
    """Write constants to netCDF: per-grid-cell ones on (lat, lon), which also
    carry coordinates, and scalars without dimensions.
    """
    data_vars = {
        name: xr.DataArray(
            np.asarray(value), dims=("lat", "lon") if np.ndim(value) else ()
        )
        for name, value in constants.items()
    }
    coords = {"lat": np.linspace(-60.0, 60.0, N_LAT), "lon": 90.0 * np.arange(N_LON)}
    xr.Dataset(data_vars, coords=coords).to_netcdf(path)
    return path


def _per_grid_cell_file_config(
    tmp_path: pathlib.Path,
    means: dict[str, float | torch.Tensor],
    stds: dict[str, float | torch.Tensor],
    **kwargs,
) -> NormalizationConfig:
    return NormalizationConfig(
        global_means_path=_write_constants(tmp_path / "means.nc", means),
        global_stds_path=_write_constants(tmp_path / "stds.nc", stds),
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
    """Without per_grid_cell, files with non-scalar variables behave as
    before: those are skipped when loading every variable, rejected by name."""
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


def test_per_grid_cell_load_skips_coordinates(tmp_path):
    config = _per_grid_cell_file_config(
        tmp_path,
        means={"a": _cell_values(0)},
        stds={"a": _cell_values(1, low=1.0, high=2.0)},
        per_grid_cell=True,
    )
    config.load()
    assert set(config.means) == {"a", "x", "y", "z"}


def test_explicit_per_grid_cell_constants_require_opt_in():
    with pytest.raises(ValueError, match="set per_grid_cell"):
        NormalizationConfig(means={"a": _cell_values(0)}, stds={"a": 1.0})


@pytest.mark.parametrize("fraction", [-0.1, 1.0])
def test_per_grid_cell_min_std_fraction_must_be_a_fraction(fraction):
    with pytest.raises(ValueError, match="per_grid_cell_min_std_fraction"):
        NormalizationConfig(
            means={"a": 0.0}, stds={"a": 1.0}, per_grid_cell_min_std_fraction=fraction
        )


@pytest.mark.parametrize("fraction", [0.01, 0.5])
def test_per_grid_cell_stds_are_floored(tmp_path, fraction):
    """A near-constant cell's std is raised to a fraction of the mean std."""
    stds = torch.full((N_LAT, N_LON), 2.0)
    stds[0, 0] = 0.0
    stds[1, 1] = 1e-9
    config = _per_grid_cell_file_config(
        tmp_path,
        means={"a": torch.zeros(N_LAT, N_LON), "b": 0.0},
        stds={"a": stds, "b": 0.0},
        per_grid_cell=True,
        per_grid_cell_min_std_fraction=fraction,
    )
    config.load()
    floored = config.stds["a"]
    assert isinstance(floored, torch.Tensor)
    floor = fraction * float(stds.mean())
    assert float(floored[0, 0]) == pytest.approx(floor)
    assert float(floored[1, 1]) == pytest.approx(floor)
    assert float(floored[2, 2]) == 2.0
    # Scalar stds are not floored, as before.
    assert config.stds["b"] == 0.0


def test_per_grid_cell_stds_all_zero_are_rejected(tmp_path):
    config = _per_grid_cell_file_config(
        tmp_path,
        means={"a": torch.zeros(N_LAT, N_LON)},
        stds={"a": torch.zeros(N_LAT, N_LON)},
        per_grid_cell=True,
    )
    with pytest.raises(ValueError, match="zero in every grid cell"):
        config.build(["a"])


def test_per_grid_cell_constants_must_be_finite(tmp_path):
    means = torch.zeros(N_LAT, N_LON)
    means[0, 0] = float("nan")
    config = _per_grid_cell_file_config(
        tmp_path,
        means={"a": means},
        stds={"a": torch.ones(N_LAT, N_LON)},
        per_grid_cell=True,
    )
    with pytest.raises(ValueError, match="non-finite"):
        config.load()


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


def test_loss_normalizer_combines_per_grid_cell_network_and_residual():
    """Loss normalization accepts per-grid-cell constants from either the
    network or the residual config."""
    network_stds = _cell_values(0, low=1.0, high=2.0)
    residual_stds = _cell_values(1, low=0.1, high=0.2)
    config = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={"a": 0.0, "b": _cell_values(2)},
            stds={"a": 1.0, "b": network_stds},
            per_grid_cell=True,
        ),
        residual=NormalizationConfig(
            means={"a": _cell_values(3)}, stds={"a": residual_stds}, per_grid_cell=True
        ),
    )
    loss_normalizer = config.get_loss_normalizer(
        names=["a", "b"], residual_scaled_names=["a"]
    )
    torch.testing.assert_close(loss_normalizer.stds["a"], _on_device(residual_stds))
    torch.testing.assert_close(loss_normalizer.stds["b"], _on_device(network_stds))
    loss_normalizer.validate_spatial_shape((N_LAT, N_LON))


# Grouped per-grid-cell normalization: per-grid-cell constants per group, with
# groups by data source or by configuration. Both are the same mechanism with
# different labels.
SOURCE_GROUPS = ["c96", "era5"]
CONFIG_GROUPS = ["amip", "ramped", "som", "era5"]


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


@pytest.mark.parametrize(
    "labels", [SOURCE_GROUPS, CONFIG_GROUPS], ids=["source", "config"]
)
def test_grouped_per_grid_cell_normalizer_uses_group_cell_constants(labels):
    groups = {label: _per_grid_cell_group(seed) for seed, label in enumerate(labels)}
    grouped = _build_per_grid_cell_grouped(groups)
    assert grouped.is_per_grid_cell
    normalizer = grouped.bind(_one_sample_per_label(labels))
    x = _on_device(torch.randn(len(labels), N_LAT, N_LON))
    normalized = normalizer.normalize({"a": x})["a"]
    for i, label in enumerate(labels):
        mean = _on_device(groups[label].means["a"])
        std = _on_device(groups[label].stds["a"])
        torch.testing.assert_close(normalized[i], (x[i] - mean) / std)
    torch.testing.assert_close(normalizer.denormalize({"a": normalized})["a"], x)


def test_grouped_per_grid_cell_pinned_variable_uses_pooled_cell_constants():
    groups = {
        label: _per_grid_cell_group(seed) for seed, label in enumerate(SOURCE_GROUPS)
    }
    pooled = _per_grid_cell_pooled()
    grouped = _build_per_grid_cell_grouped(groups, ["pinned"], pooled=pooled)
    normalizer = grouped.bind(_one_sample_per_label(SOURCE_GROUPS))
    x = _on_device(torch.randn(2, N_LAT, N_LON))
    torch.testing.assert_close(
        normalizer.normalize({"pinned": x})["pinned"],
        (x - pooled.means["pinned"]) / pooled.stds["pinned"],
    )


def test_grouped_normalizer_mixes_scalar_and_per_grid_cell_groups():
    """A variable may be scalar in one group and per-grid-cell in another, on
    top of scalar pooled constants."""
    groups = {
        "c96": NormalizationConfig(
            means={"a": 5.0, "pinned": 0.0}, stds={"a": 2.0, "pinned": 1.0}
        ),
        "era5": _per_grid_cell_group(0),
    }
    grouped = _build_per_grid_cell_grouped(groups, pooled=_pooled_normalizer())
    assert grouped.is_per_grid_cell
    normalizer = grouped.bind(_one_sample_per_label(SOURCE_GROUPS))
    x = _on_device(torch.randn(2, N_LAT, N_LON))
    normalized = normalizer.normalize({"a": x})["a"]
    torch.testing.assert_close(normalized[0], (x[0] - 5.0) / 2.0)
    era5 = groups["era5"]
    torch.testing.assert_close(
        normalized[1],
        (x[1] - _on_device(era5.means["a"])) / _on_device(era5.stds["a"]),
    )


def test_grouped_per_grid_cell_rejects_group_constant_in_one_cell():
    """The check for a variable constant within a group applies per cell."""
    group = _per_grid_cell_group(0)
    stds = torch.as_tensor(group.stds["a"]).clone()
    stds[0, 0] = 1e-8
    groups = {
        "c96": dataclasses.replace(group, stds={**group.stds, "a": stds}),
        "era5": _per_grid_cell_group(1),
    }
    with pytest.raises(ValueError, match="pinned_variables"):
        _build_per_grid_cell_grouped(groups)


def test_grouped_validate_spatial_shape_checks_every_group():
    groups = {
        "c96": NormalizationConfig(
            means={"a": 5.0, "pinned": 0.0}, stds={"a": 2.0, "pinned": 1.0}
        ),
        "era5": _per_grid_cell_group(0),
    }
    grouped = _build_per_grid_cell_grouped(groups, pooled=_pooled_normalizer())
    grouped.validate_spatial_shape((N_LAT, N_LON))
    with pytest.raises(ValueError, match="does not match the data's spatial shape"):
        grouped.validate_spatial_shape((N_LON, N_LAT))

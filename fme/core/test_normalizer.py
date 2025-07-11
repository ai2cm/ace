import dataclasses
import pathlib
import tempfile

import dacite
import pytest
import torch

from fme.ace.testing.fv3gfs_data import get_scalar_dataset
from fme.core.device import move_tensordict_to_device
from fme.core.normalizer import (
    NetworkAndLossNormalizationConfig,
    NormalizationConfig,
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

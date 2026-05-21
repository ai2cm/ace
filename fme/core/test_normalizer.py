import dataclasses
import pathlib
import tempfile

import dacite
import pytest
import torch

from fme.ace.testing.fv3gfs_data import get_scalar_dataset
from fme.core.co2_temperature_offset import CO2TemperatureProfileConfig
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


def _make_shared_offset_normalizer(means, stds):
    return NormalizationConfig(
        means=means,
        stds=stds,
        experimental_use_shared_temperature_offset=True,
    ).build(list(means))


def test_shared_temperature_offset_matches_users_worked_example():
    # Worked example from the user: surface_temperature norm mean = 280K and
    # the sample's mean surface_temperature = 285K -> offset = -5K. For
    # air_temperature_4 with norm mean 250K and sample mean 245K, after adding
    # the offset and subtracting the air_temperature_4 mean the sample's
    # normalized values are -10 / std (= -2.5 with std=4).
    means = {"surface_temperature": 280.0, "air_temperature_4": 250.0}
    stds = {"surface_temperature": 2.0, "air_temperature_4": 4.0}
    normalizer = _make_shared_offset_normalizer(means, stds)
    tensors = move_tensordict_to_device(
        {
            "surface_temperature": torch.full((1, 4, 4), 285.0),
            "air_temperature_4": torch.full((1, 4, 4), 245.0),
        }
    )
    normalized = normalizer.normalize(tensors)
    # surface_temperature: (285 + (-5) - 280) / 2 = 0
    torch.testing.assert_close(
        normalized["surface_temperature"],
        torch.zeros_like(normalized["surface_temperature"]),
    )
    # air_temperature_4: (245 + (-5) - 250) / 4 = -2.5
    torch.testing.assert_close(
        normalized["air_temperature_4"],
        torch.full_like(normalized["air_temperature_4"], -2.5),
    )


def test_shared_temperature_offset_round_trips_through_denormalize():
    torch.manual_seed(0)
    means = {
        "surface_temperature": 288.0,
        "air_temperature_0": 220.0,
        "TMP850": 250.0,
    }
    stds = {"surface_temperature": 5.0, "air_temperature_0": 3.0, "TMP850": 4.0}
    normalizer = _make_shared_offset_normalizer(means, stds)
    tensors = move_tensordict_to_device(
        {k: torch.randn(2, 4, 4) + means[k] for k in means}
    )
    round_tripped = normalizer.denormalize(normalizer.normalize(tensors))
    for k in means:
        torch.testing.assert_close(round_tripped[k], tensors[k])


def test_shared_temperature_offset_preserves_horizontal_gradients():
    # The offset is spatially constant per sample, so within-field gradients
    # are unchanged (apart from the std rescaling that standard normalization
    # would have done anyway). With std=1 the gradient is exactly preserved.
    means = {"surface_temperature": 280.0}
    stds = {"surface_temperature": 1.0}
    normalizer = _make_shared_offset_normalizer(means, stds)
    tensors = move_tensordict_to_device(
        {"surface_temperature": torch.tensor([[285.0, 290.0, 275.0]])}
    )
    normalized = normalizer.normalize(tensors)
    raw_gradient = (
        tensors["surface_temperature"][0, 1] - tensors["surface_temperature"][0, 0]
    )
    normalized_gradient = (
        normalized["surface_temperature"][0, 1]
        - normalized["surface_temperature"][0, 0]
    )
    torch.testing.assert_close(normalized_gradient, raw_gradient)


def test_shared_temperature_offset_is_per_sample():
    # Different batch elements get different offsets, applied independently.
    means = {"surface_temperature": 280.0, "air_temperature_0": 220.0}
    stds = {"surface_temperature": 1.0, "air_temperature_0": 1.0}
    normalizer = _make_shared_offset_normalizer(means, stds)
    surface_t = torch.tensor([[[285.0]], [[270.0]]])  # two samples
    air_t_0 = torch.tensor([[[230.0]], [[230.0]]])
    tensors = move_tensordict_to_device(
        {"surface_temperature": surface_t, "air_temperature_0": air_t_0}
    )
    normalized = normalizer.normalize(tensors)
    # Sample 0: offset = 280 - 285 = -5; sample 1: offset = 280 - 270 = +10.
    # air_temperature_0 normalized = (raw + offset - 220) / 1
    expected = torch.tensor([[[230.0 + (-5.0) - 220.0]], [[230.0 + 10.0 - 220.0]]])
    torch.testing.assert_close(normalized["air_temperature_0"].cpu(), expected)


def test_shared_temperature_offset_raises_at_build_when_surface_temperature_missing():
    normalization = NormalizationConfig(
        means={"air_temperature_0": 220.0},
        stds={"air_temperature_0": 1.0},
        experimental_use_shared_temperature_offset=True,
    )
    with pytest.raises(ValueError, match="surface_temperature"):
        normalization.build(["air_temperature_0"])


def test_shared_temperature_offset_raises_at_normalize_when_input_missing():
    means = {"surface_temperature": 280.0, "air_temperature_0": 220.0}
    stds = {"surface_temperature": 1.0, "air_temperature_0": 1.0}
    normalizer = _make_shared_offset_normalizer(means, stds)
    tensors = move_tensordict_to_device({"air_temperature_0": torch.zeros(1, 4, 4)})
    with pytest.raises(ValueError, match="surface_temperature"):
        normalizer.normalize(tensors)


def test_shared_temperature_offset_raises_when_denormalize_called_first():
    means = {"surface_temperature": 280.0, "air_temperature_0": 220.0}
    stds = {"surface_temperature": 1.0, "air_temperature_0": 1.0}
    normalizer = _make_shared_offset_normalizer(means, stds)
    with pytest.raises(RuntimeError, match="normalize"):
        normalizer.denormalize(
            move_tensordict_to_device({"air_temperature_0": torch.zeros(1, 4, 4)})
        )


def test_network_normalizer_has_shared_offset_but_loss_normalizer_does_not():
    # When the flag is set on the network NormalizationConfig, only the
    # network normalizer carries the offset behavior. The loss normalizer
    # built from the combined config never has it -- offsets are unnecessary
    # for the loss and would not cancel cleanly across separate normalize()
    # calls on target vs. prediction.
    combined = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={"surface_temperature": 280.0, "air_temperature_0": 220.0},
            stds={"surface_temperature": 2.0, "air_temperature_0": 4.0},
            experimental_use_shared_temperature_offset=True,
        ),
        residual=NormalizationConfig(
            means={"air_temperature_0": 219.0},
            stds={"air_temperature_0": 5.0},
        ),
    )
    network_normalizer = combined.get_network_normalizer(
        ["surface_temperature", "air_temperature_0"]
    )
    loss_normalizer = combined.get_loss_normalizer(
        names=["surface_temperature", "air_temperature_0"],
        residual_scaled_names=["air_temperature_0"],
    )
    assert network_normalizer.use_shared_temperature_offset
    assert not loss_normalizer.use_shared_temperature_offset


def _make_co2_offset_normalizer(means, stds, profile=None):
    if profile is None:
        profile = CO2TemperatureProfileConfig(co2_reference_vmr=280e-6)
    return NormalizationConfig(
        means=means,
        stds=stds,
        experimental_co2_temperature_offset=profile,
    ).build(list(means))


def test_co2_temperature_offset_zero_at_reference_co2():
    # When CO2 equals the reference, log2(ratio) = 0 and the offset
    # should be exactly zero, leaving standard normalization untouched.
    profile = CO2TemperatureProfileConfig(co2_reference_vmr=280e-6)
    means = {"air_temperature_4": 250.0, "global_mean_co2": 0.0}
    stds = {"air_temperature_4": 5.0, "global_mean_co2": 1.0}
    normalizer = _make_co2_offset_normalizer(means, stds, profile=profile)
    tensors = move_tensordict_to_device(
        {
            "air_temperature_4": torch.full((1, 4, 4), 260.0),
            "global_mean_co2": torch.full((1, 4, 4), 280e-6),
        }
    )
    normalized = normalizer.normalize(tensors)
    # (260 - 0 - 250) / 5 = 2.0
    torch.testing.assert_close(
        normalized["air_temperature_4"],
        torch.full_like(normalized["air_temperature_4"], 2.0),
    )


def test_co2_temperature_offset_applies_per_doubling_response_per_field():
    # With CO2 doubled, surface_temperature should be detrended by
    # delta_t_surface_per_doubling before standard normalization. Verifies
    # both the value and the per-field pressure mapping.
    profile = CO2TemperatureProfileConfig(
        co2_reference_vmr=280e-6,
        delta_t_surface_per_doubling=3.0,
        delta_t_stratosphere_per_doubling=-10.0,
        tropopause_pressure_pa=2.0e4,
        stratosphere_top_pressure_pa=1.0e2,
    )
    means = {"surface_temperature": 290.0, "global_mean_co2": 0.0}
    stds = {"surface_temperature": 1.0, "global_mean_co2": 1.0}
    normalizer = _make_co2_offset_normalizer(means, stds, profile=profile)
    tensors = move_tensordict_to_device(
        {
            "surface_temperature": torch.full((1, 4, 4), 293.0),  # warmed by +3K
            "global_mean_co2": torch.full((1, 4, 4), 560e-6),  # 2x reference
        }
    )
    normalized = normalizer.normalize(tensors)
    # ΔT_CO2 at surface = +3K per doubling × log2(2) = +3K. Subtracted before
    # standard normalize: (293 - 3 - 290) / 1 = 0.
    torch.testing.assert_close(
        normalized["surface_temperature"],
        torch.zeros_like(normalized["surface_temperature"]),
    )


def test_co2_temperature_offset_round_trips_through_denormalize():
    torch.manual_seed(0)
    means = {
        "surface_temperature": 290.0,
        "air_temperature_0": 220.0,  # upper stratosphere
        "air_temperature_4": 250.0,
        "global_mean_co2": 0.0,
    }
    stds = {k: 1.0 + 2.0 * i for i, k in enumerate(means)}
    profile = CO2TemperatureProfileConfig(co2_reference_vmr=280e-6)
    normalizer = _make_co2_offset_normalizer(means, stds, profile=profile)
    tensors = move_tensordict_to_device(
        {
            "surface_temperature": 290.0 + torch.randn(2, 4, 4),
            "air_temperature_0": 220.0 + torch.randn(2, 4, 4),
            "air_temperature_4": 250.0 + torch.randn(2, 4, 4),
            "global_mean_co2": torch.full((2, 4, 4), 420e-6),  # 1.5x reference
        }
    )
    round_tripped = normalizer.denormalize(normalizer.normalize(tensors))
    for k in ("surface_temperature", "air_temperature_0", "air_temperature_4"):
        torch.testing.assert_close(round_tripped[k], tensors[k])


def test_co2_temperature_offset_uses_per_sample_co2():
    # Different batch elements have different CO2 values -> different
    # per-sample offsets applied.
    profile = CO2TemperatureProfileConfig(
        co2_reference_vmr=280e-6,
        delta_t_surface_per_doubling=3.0,
    )
    means = {"surface_temperature": 290.0, "global_mean_co2": 0.0}
    stds = {"surface_temperature": 1.0, "global_mean_co2": 1.0}
    normalizer = _make_co2_offset_normalizer(means, stds, profile=profile)
    surface_t = torch.tensor([[[293.0]], [[290.0]]])  # both warmer / at norm
    co2 = torch.tensor([[[560e-6]], [[280e-6]]])  # 2x and 1x reference
    tensors = move_tensordict_to_device(
        {"surface_temperature": surface_t, "global_mean_co2": co2}
    )
    normalized = normalizer.normalize(tensors)
    # Sample 0: ΔT_CO2 = +3K, normalize = (293 - 3 - 290)/1 = 0
    # Sample 1: ΔT_CO2 = 0K, normalize = (290 - 0 - 290)/1 = 0
    expected = torch.zeros((2, 1, 1))
    torch.testing.assert_close(normalized["surface_temperature"].cpu(), expected)


def test_co2_temperature_offset_raises_when_global_mean_co2_missing():
    profile = CO2TemperatureProfileConfig(co2_reference_vmr=280e-6)
    means = {"surface_temperature": 290.0, "global_mean_co2": 0.0}
    stds = {"surface_temperature": 1.0, "global_mean_co2": 1.0}
    normalizer = _make_co2_offset_normalizer(means, stds, profile=profile)
    tensors = move_tensordict_to_device(
        {"surface_temperature": torch.full((1, 4, 4), 290.0)}
    )
    with pytest.raises(ValueError, match="global_mean_co2"):
        normalizer.normalize(tensors)


def test_co2_temperature_offset_raises_when_denormalize_called_first():
    profile = CO2TemperatureProfileConfig(co2_reference_vmr=280e-6)
    means = {"surface_temperature": 290.0, "global_mean_co2": 0.0}
    stds = {"surface_temperature": 1.0, "global_mean_co2": 1.0}
    normalizer = _make_co2_offset_normalizer(means, stds, profile=profile)
    with pytest.raises(RuntimeError, match="normalize"):
        normalizer.denormalize(
            move_tensordict_to_device({"surface_temperature": torch.zeros(1, 4, 4)})
        )


def test_loss_normalizer_strips_co2_temperature_offset():
    # When the CO2 offset is set on the network config, the loss normalizer
    # returned by NetworkAndLossNormalizationConfig must not carry it.
    combined = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={"surface_temperature": 290.0, "global_mean_co2": 0.0},
            stds={"surface_temperature": 1.0, "global_mean_co2": 1.0},
            experimental_co2_temperature_offset=CO2TemperatureProfileConfig(),
        ),
    )
    loss_normalizer = combined.get_loss_normalizer(
        names=["surface_temperature", "global_mean_co2"],
        residual_scaled_names=[],
    )
    assert loss_normalizer.co2_temperature_offset is None


def test_required_forcing_names_lists_global_mean_co2_when_enabled():
    no_co2 = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(means={"a": 1.0}, stds={"a": 1.0}),
    )
    assert no_co2.required_forcing_names == []
    with_co2 = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={"surface_temperature": 290.0, "global_mean_co2": 0.0},
            stds={"surface_temperature": 1.0, "global_mean_co2": 1.0},
            experimental_co2_temperature_offset=CO2TemperatureProfileConfig(),
        ),
    )
    assert with_co2.required_forcing_names == ["global_mean_co2"]


def test_co2_and_shared_temperature_offset_combine_correctly():
    # When both are enabled, CO2 is subtracted first, then the shared
    # offset is computed from the CO2-detrended surface_temperature.
    # End-to-end the round-trip through normalize/denormalize must be exact.
    torch.manual_seed(0)
    means = {
        "surface_temperature": 290.0,
        "air_temperature_4": 250.0,
        "global_mean_co2": 0.0,
    }
    stds = {
        "surface_temperature": 1.5,
        "air_temperature_4": 3.0,
        "global_mean_co2": 1.0,
    }
    normalizer = NormalizationConfig(
        means=means,
        stds=stds,
        experimental_use_shared_temperature_offset=True,
        experimental_co2_temperature_offset=CO2TemperatureProfileConfig(
            co2_reference_vmr=280e-6
        ),
    ).build(list(means))
    tensors = move_tensordict_to_device(
        {
            "surface_temperature": 292.0 + torch.randn(2, 3, 3),
            "air_temperature_4": 251.0 + torch.randn(2, 3, 3),
            "global_mean_co2": torch.full((2, 3, 3), 420e-6),
        }
    )
    round_tripped = normalizer.denormalize(normalizer.normalize(tensors))
    for k in ("surface_temperature", "air_temperature_4"):
        torch.testing.assert_close(round_tripped[k], tensors[k])


def test_loss_normalizer_strips_shared_offset_in_fallback_case():
    # Even with no residual and no loss config -- where get_loss_normalizer
    # falls back to building the network normalizer -- the loss normalizer
    # must not carry the offset behavior.
    combined = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={"surface_temperature": 280.0, "air_temperature_0": 220.0},
            stds={"surface_temperature": 2.0, "air_temperature_0": 4.0},
            experimental_use_shared_temperature_offset=True,
        ),
    )
    loss_normalizer = combined.get_loss_normalizer(
        names=["surface_temperature", "air_temperature_0"],
        residual_scaled_names=[],
    )
    assert not loss_normalizer.use_shared_temperature_offset


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

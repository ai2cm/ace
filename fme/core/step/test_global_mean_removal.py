import dataclasses

import dacite
import pytest
import torch

from fme.core.device import move_tensordict_to_device
from fme.core.humidity import bolton_saturation_vapor_pressure
from fme.core.step.global_mean_removal import (
    GlobalMeanRemovalConfigUnion,
    PerChannelGlobalMeanRemovalConfig,
    SharedGlobalMeanRemovalConfig,
)


def _build_normalizer_for(means, stds):
    from fme.core.normalizer import NormalizationConfig

    return NormalizationConfig(means=means, stds=stds).build(list(means))


# ── SharedGlobalMeanRemoval ─────────────────────────────────────────────


def _make_shared(
    means, stds, append_as_input=False, qsat_scaled_names=None, field_names=None
):
    normalizer = _build_normalizer_for(means, stds)
    config = SharedGlobalMeanRemovalConfig(
        reference_field="surface_temperature",
        field_names=list(means.keys()) if field_names is None else field_names,
        append_as_input=append_as_input,
        qsat_scaled_names=qsat_scaled_names or [],
    )
    return config.build(normalizer, list(means.keys()))


def test_shared_worked_example():
    means = {"surface_temperature": 280.0, "air_temperature_4": 250.0}
    stds = {"surface_temperature": 2.0, "air_temperature_4": 4.0}
    transform = _make_shared(means, stds)
    tensors = move_tensordict_to_device(
        {
            "surface_temperature": torch.full((1, 4, 4), 285.0),
            "air_temperature_4": torch.full((1, 4, 4), 245.0),
        }
    )
    result = transform.forward_transform(tensors, None)
    # offset = 280 - 285 = -5; shifted = 285 + (-5) = 280
    torch.testing.assert_close(
        result["surface_temperature"],
        torch.full_like(result["surface_temperature"], 280.0),
    )
    # shifted air_temperature_4 = 245 + (-5) = 240
    torch.testing.assert_close(
        result["air_temperature_4"],
        torch.full_like(result["air_temperature_4"], 240.0),
    )


def test_shared_round_trip():
    torch.manual_seed(0)
    means = {"surface_temperature": 288.0, "air_temperature_0": 220.0}
    stds = {"surface_temperature": 5.0, "air_temperature_0": 3.0}
    transform = _make_shared(means, stds)
    tensors = move_tensordict_to_device(
        {k: torch.randn(2, 4, 4) + means[k] for k in means}
    )
    result = transform.forward_transform(tensors, None)
    restored = transform.inverse_transform(result)
    for k in means:
        torch.testing.assert_close(restored[k], tensors[k])


def test_shared_preserves_horizontal_gradients():
    means = {"surface_temperature": 280.0}
    stds = {"surface_temperature": 1.0}
    transform = _make_shared(means, stds)
    tensors = move_tensordict_to_device(
        {"surface_temperature": torch.tensor([[285.0, 290.0, 275.0]])}
    )
    result = transform.forward_transform(tensors, None)
    raw_grad = (
        tensors["surface_temperature"][0, 1] - tensors["surface_temperature"][0, 0]
    )
    shifted_grad = (
        result["surface_temperature"][0, 1] - result["surface_temperature"][0, 0]
    )
    torch.testing.assert_close(shifted_grad, raw_grad)


def test_shared_is_per_sample():
    means = {"surface_temperature": 280.0, "air_temperature_0": 220.0}
    stds = {"surface_temperature": 1.0, "air_temperature_0": 1.0}
    transform = _make_shared(means, stds)
    surface_t = torch.tensor([[[285.0]], [[270.0]]])
    air_t_0 = torch.tensor([[[230.0]], [[230.0]]])
    tensors = move_tensordict_to_device(
        {"surface_temperature": surface_t, "air_temperature_0": air_t_0}
    )
    result = transform.forward_transform(tensors, None)
    # sample 0: offset = 280 - 285 = -5; sample 1: offset = 280 - 270 = +10
    expected = torch.tensor([[[230.0 + (-5.0)]], [[230.0 + 10.0]]])
    torch.testing.assert_close(result["air_temperature_0"].cpu(), expected)


def test_shared_no_extra_channels():
    means = {"surface_temperature": 280.0}
    stds = {"surface_temperature": 1.0}
    transform = _make_shared(means, stds)
    assert transform.n_extra_input_channels == 0
    assert transform.extra_channel_names == []
    tensors = move_tensordict_to_device(
        {"surface_temperature": torch.full((2, 4, 4), 285.0)}
    )
    transform.forward_transform(tensors, None)
    assert transform.extras_normalized() == {}


def test_shared_extra_channels():
    means = {"surface_temperature": 280.0}
    stds = {"surface_temperature": 5.0}
    transform = _make_shared(means, stds, append_as_input=True)
    assert transform.n_extra_input_channels == 1
    [extra_name] = transform.extra_channel_names
    tensors = move_tensordict_to_device(
        {"surface_temperature": torch.full((2, 4, 4), 285.0)}
    )
    transform.forward_transform(tensors, None)
    extras = transform.extras_normalized()
    assert list(extras) == [extra_name]
    extra = extras[extra_name]
    assert extra.shape == (2, 4, 4)
    # sample_mean = 285, normalized = (285 - 280) / 5 = 1.0
    torch.testing.assert_close(extra, torch.full_like(extra, 1.0))


def test_shared_raises_on_masked_reference():
    means = {"surface_temperature": 280.0}
    stds = {"surface_temperature": 1.0}
    transform = _make_shared(means, stds)
    tensors = move_tensordict_to_device(
        {"surface_temperature": torch.full((2, 4, 4), 285.0)}
    )
    data_mask = {"surface_temperature": torch.tensor([True, False])}
    with pytest.raises(ValueError, match="masked"):
        transform.forward_transform(tensors, data_mask)


def test_shared_raises_on_missing_reference():
    means = {"surface_temperature": 280.0, "air_temperature_0": 220.0}
    stds = {"surface_temperature": 1.0, "air_temperature_0": 1.0}
    transform = _make_shared(means, stds)
    tensors = move_tensordict_to_device(
        {"air_temperature_0": torch.full((2, 4, 4), 225.0)}
    )
    with pytest.raises(ValueError, match="not present"):
        transform.forward_transform(tensors, None)


def test_shared_inverse_before_forward_raises():
    means = {"surface_temperature": 280.0}
    stds = {"surface_temperature": 1.0}
    transform = _make_shared(means, stds)
    with pytest.raises(RuntimeError, match="forward_transform"):
        transform.inverse_transform({"surface_temperature": torch.zeros(1, 4, 4)})


# ── SharedGlobalMeanRemoval qsat_scaled_names ───────────────────────────


def test_shared_qsat_scaled_worked_example():
    means = {"surface_temperature": 280.0, "specific_humidity_4": 0.005}
    stds = {"surface_temperature": 2.0, "specific_humidity_4": 0.001}
    transform = _make_shared(
        means, stds, field_names=[], qsat_scaled_names=["specific_humidity_4"]
    )
    tensors = move_tensordict_to_device(
        {
            "surface_temperature": torch.full((1, 4, 4), 290.0),
            "specific_humidity_4": torch.full((1, 4, 4), 0.006),
        }
    )
    result = transform.forward_transform(tensors, None)
    expected_factor = (
        bolton_saturation_vapor_pressure(torch.tensor(280.0))
        / bolton_saturation_vapor_pressure(torch.tensor(290.0))
    ).item()
    # reference is unchanged (only listed in qsat_scaled_names? no, it's not listed)
    torch.testing.assert_close(
        result["surface_temperature"],
        torch.full_like(result["surface_temperature"], 290.0),
    )
    torch.testing.assert_close(
        result["specific_humidity_4"].cpu(),
        torch.full((1, 4, 4), 0.006 * expected_factor),
    )


def test_shared_qsat_scaled_round_trip():
    torch.manual_seed(0)
    means = {"surface_temperature": 288.0, "specific_humidity_0": 0.01}
    stds = {"surface_temperature": 5.0, "specific_humidity_0": 0.002}
    transform = _make_shared(
        means, stds, field_names=[], qsat_scaled_names=["specific_humidity_0"]
    )
    tensors = move_tensordict_to_device(
        {
            "surface_temperature": torch.randn(2, 4, 4) + 290.0,
            "specific_humidity_0": torch.rand(2, 4, 4) * 0.01 + 0.005,
        }
    )
    result = transform.forward_transform(tensors, None)
    restored = transform.inverse_transform(result)
    for k in tensors:
        torch.testing.assert_close(restored[k], tensors[k])


def test_shared_qsat_scaled_is_per_sample():
    means = {"surface_temperature": 280.0, "q": 0.005}
    stds = {"surface_temperature": 1.0, "q": 0.001}
    transform = _make_shared(means, stds, field_names=[], qsat_scaled_names=["q"])
    surface_t = torch.tensor([[[285.0]], [[270.0]]])
    q = torch.tensor([[[0.01]], [[0.01]]])
    tensors = move_tensordict_to_device({"surface_temperature": surface_t, "q": q})
    result = transform.forward_transform(tensors, None)
    factor_warm = (
        bolton_saturation_vapor_pressure(torch.tensor(280.0))
        / bolton_saturation_vapor_pressure(torch.tensor(285.0))
    ).item()
    factor_cold = (
        bolton_saturation_vapor_pressure(torch.tensor(280.0))
        / bolton_saturation_vapor_pressure(torch.tensor(270.0))
    ).item()
    expected = torch.tensor([[[0.01 * factor_warm]], [[0.01 * factor_cold]]])
    torch.testing.assert_close(result["q"].cpu(), expected)


def test_shared_qsat_scaled_does_not_apply_offset_to_humidity():
    """Humidities listed only in qsat_scaled_names must not be shifted by
    the temperature offset — only multiplicatively scaled."""
    means = {"surface_temperature": 280.0, "q": 0.005}
    stds = {"surface_temperature": 1.0, "q": 0.001}
    # field_names is empty: no fields get the additive offset
    transform = _make_shared(means, stds, field_names=[], qsat_scaled_names=["q"])
    tensors = move_tensordict_to_device(
        {
            # sample_mean equals climatology, so factor = 1
            "surface_temperature": torch.full((1, 4, 4), 280.0),
            "q": torch.full((1, 4, 4), 0.01),
        }
    )
    result = transform.forward_transform(tensors, None)
    torch.testing.assert_close(result["q"].cpu(), torch.full((1, 4, 4), 0.01))


def test_shared_qsat_scaled_output_only_field_unscaled_by_inverse():
    """A field listed in qsat_scaled_names that only appears in the
    output is not scaled by forward_transform (it isn't an input) but
    *is* unscaled by inverse_transform, so the network learns to
    produce it in the scaled space."""
    means = {"surface_temperature": 280.0, "q_out": 0.005}
    stds = {"surface_temperature": 1.0, "q_out": 0.001}
    transform = _make_shared(means, stds, field_names=[], qsat_scaled_names=["q_out"])
    inputs = move_tensordict_to_device(
        {"surface_temperature": torch.full((1, 4, 4), 290.0)}
    )
    transform.forward_transform(inputs, None)
    output = move_tensordict_to_device({"q_out": torch.full((1, 4, 4), 0.01)})
    restored = transform.inverse_transform(output)
    factor = (
        bolton_saturation_vapor_pressure(torch.tensor(280.0))
        / bolton_saturation_vapor_pressure(torch.tensor(290.0))
    ).item()
    torch.testing.assert_close(
        restored["q_out"].cpu(), torch.full((1, 4, 4), 0.01 / factor)
    )


def test_shared_qsat_scaled_default_empty_is_noop():
    """Default empty qsat_scaled_names must not change behavior vs. the
    pre-existing shared transform."""
    means = {"surface_temperature": 280.0, "q": 0.005}
    stds = {"surface_temperature": 1.0, "q": 0.001}
    transform = _make_shared(
        means, stds, field_names=["surface_temperature"], qsat_scaled_names=[]
    )
    tensors = move_tensordict_to_device(
        {
            "surface_temperature": torch.full((1, 4, 4), 285.0),
            "q": torch.full((1, 4, 4), 0.01),
        }
    )
    result = transform.forward_transform(tensors, None)
    # q is in neither field_names nor qsat_scaled_names — unchanged.
    torch.testing.assert_close(result["q"].cpu(), torch.full((1, 4, 4), 0.01))


# ── PerChannelGlobalMeanRemoval ─────────────────────────────────────────


def _make_per_channel(means, stds, field_names=None, append_as_input=False):
    normalizer = _build_normalizer_for(means, stds)
    config = PerChannelGlobalMeanRemovalConfig(
        field_names=field_names,
        append_as_input=append_as_input,
    )
    return config.build(normalizer, list(means.keys()))


def test_per_channel_removes_correct_means():
    means = {"a": 10.0, "b": 20.0}
    stds = {"a": 1.0, "b": 1.0}
    transform = _make_per_channel(means, stds)
    tensors = move_tensordict_to_device(
        {
            "a": torch.tensor([[[12.0, 14.0]]]),  # mean = 13
            "b": torch.tensor([[[22.0, 28.0]]]),  # mean = 25
        }
    )
    result = transform.forward_transform(tensors, None)
    # a: shift = 10 - 13 = -3; result = [12 - 3, 14 - 3] = [9, 11]
    torch.testing.assert_close(result["a"].cpu(), torch.tensor([[[9.0, 11.0]]]))
    # b: shift = 20 - 25 = -5; result = [22 - 5, 28 - 5] = [17, 23]
    torch.testing.assert_close(result["b"].cpu(), torch.tensor([[[17.0, 23.0]]]))


def test_per_channel_round_trip():
    torch.manual_seed(0)
    means = {"a": 10.0, "b": 20.0}
    stds = {"a": 2.0, "b": 3.0}
    transform = _make_per_channel(means, stds)
    tensors = move_tensordict_to_device(
        {k: torch.randn(3, 8, 8) + means[k] for k in means}
    )
    result = transform.forward_transform(tensors, None)
    restored = transform.inverse_transform(result)
    for k in means:
        torch.testing.assert_close(restored[k], tensors[k])


def test_per_channel_is_per_sample():
    means = {"a": 0.0}
    stds = {"a": 1.0}
    transform = _make_per_channel(means, stds)
    tensors = move_tensordict_to_device(
        {"a": torch.tensor([[[10.0, 20.0]], [[100.0, 200.0]]])}
    )
    result = transform.forward_transform(tensors, None)
    # clim_mean=0, so shift per sample is -sample_mean
    # sample 0: mean=15 → shift=-15 → [10-15, 20-15] = [-5, 5]
    # sample 1: mean=150 → shift=-150 → [100-150, 200-150] = [-50, 50]
    torch.testing.assert_close(
        result["a"].cpu(), torch.tensor([[[-5.0, 5.0]], [[-50.0, 50.0]]])
    )


def test_per_channel_no_extra_channels():
    means = {"a": 0.0}
    stds = {"a": 1.0}
    transform = _make_per_channel(means, stds)
    assert transform.n_extra_input_channels == 0
    assert transform.extra_channel_names == []
    tensors = move_tensordict_to_device({"a": torch.full((2, 4, 4), 5.0)})
    transform.forward_transform(tensors, None)
    assert transform.extras_normalized() == {}


def test_per_channel_extra_channels():
    means = {"a": 10.0, "b": 20.0}
    stds = {"a": 2.0, "b": 5.0}
    transform = _make_per_channel(means, stds, append_as_input=True)
    assert transform.n_extra_input_channels == 2
    a_extra, b_extra = transform.extra_channel_names
    tensors = move_tensordict_to_device(
        {
            "a": torch.full((1, 4, 4), 14.0),  # mean=14
            "b": torch.full((1, 4, 4), 25.0),  # mean=25
        }
    )
    transform.forward_transform(tensors, None)
    extras = transform.extras_normalized()
    assert list(extras) == [a_extra, b_extra]
    assert extras[a_extra].shape == (1, 4, 4)
    # a: (14 - 10) / 2 = 2.0
    torch.testing.assert_close(extras[a_extra].cpu(), torch.full((1, 4, 4), 2.0))
    # b: (25 - 20) / 5 = 1.0
    torch.testing.assert_close(extras[b_extra].cpu(), torch.full((1, 4, 4), 1.0))


def test_per_channel_with_field_names_subset():
    means = {"a": 10.0, "b": 20.0}
    stds = {"a": 2.0, "b": 5.0}
    transform = _make_per_channel(means, stds, field_names=["a"])
    tensors = move_tensordict_to_device(
        {"a": torch.full((1, 4, 4), 14.0), "b": torch.full((1, 4, 4), 25.0)}
    )
    result = transform.forward_transform(tensors, None)
    # a: shift = 10 - 14 = -4; result = 14 - 4 = 10 (== clim_mean)
    torch.testing.assert_close(result["a"].cpu(), torch.full((1, 4, 4), 10.0))
    # b was NOT shifted
    torch.testing.assert_close(result["b"].cpu(), torch.full((1, 4, 4), 25.0))


def test_per_channel_masked_no_shift():
    means = {"a": 10.0}
    stds = {"a": 2.0}
    transform = _make_per_channel(means, stds, append_as_input=True)
    [extra_name] = transform.extra_channel_names
    tensors = move_tensordict_to_device(
        {"a": torch.tensor([[[14.0, 14.0]], [[14.0, 14.0]]])}
    )
    data_mask = move_tensordict_to_device({"a": torch.tensor([True, False])})
    result = transform.forward_transform(tensors, data_mask)
    # sample 0 (unmasked): shift = 10 - 14 = -4; result = 14 - 4 = 10
    torch.testing.assert_close(result["a"][0].cpu(), torch.full((1, 2), 10.0))
    # sample 1 (masked): no shift, unchanged
    torch.testing.assert_close(result["a"][1].cpu(), torch.full((1, 2), 14.0))

    extra = transform.extras_normalized()[extra_name]
    # sample 0: -shift/std = -(-4)/2 = 2.0
    assert extra[0, 0, 0].item() == pytest.approx(2.0)
    # sample 1 (masked): no shift, extra = 0
    assert extra[1, 0, 0].item() == pytest.approx(0.0)


def test_per_channel_inverse_before_forward_raises():
    means = {"a": 0.0}
    stds = {"a": 1.0}
    transform = _make_per_channel(means, stds)
    with pytest.raises(RuntimeError, match="forward_transform"):
        transform.inverse_transform({"a": torch.zeros(1, 4, 4)})


def test_per_channel_post_normalization_mean_is_near_zero():
    """Regression: per-channel must shift toward each field's climatology,
    not to zero in physical space.  After the shift, the normalizer is
    applied; the post-normalization spatial mean must be ≈ 0 so the
    network does not see a large constant bias (a previous implementation
    shifted to zero in physical space, producing a post-normalization
    bias of ``-clim_mean / clim_std`` ≈ -19 for absolute temperatures).
    """
    torch.manual_seed(0)
    # Realistic climatology: surface temperature ~288 K, std ~15 K.
    means = {"surface_temperature": 288.0, "air_temperature_0": 220.0}
    stds = {"surface_temperature": 15.0, "air_temperature_0": 10.0}
    normalizer = _build_normalizer_for(means, stds)
    transform = _make_per_channel(means, stds)

    tensors = move_tensordict_to_device(
        {
            "surface_temperature": torch.randn(2, 8, 16) * 12.0 + 295.0,
            "air_temperature_0": torch.randn(2, 8, 16) * 8.0 + 215.0,
        }
    )
    shifted = transform.forward_transform(tensors, None)
    normalized = normalizer.normalize(shifted)
    for name in means:
        sample_means = normalized[name].mean(dim=tuple(range(1, normalized[name].ndim)))
        torch.testing.assert_close(
            sample_means.cpu(),
            torch.zeros_like(sample_means.cpu()),
            atol=1e-5,
            rtol=0.0,
        )


def test_per_channel_post_normalization_mean_near_zero_when_masked():
    """Masked samples are skipped (shift = 0), but unmasked samples must
    still land near zero in normalized space."""
    means = {"a": 100.0}
    stds = {"a": 5.0}
    normalizer = _build_normalizer_for(means, stds)
    transform = _make_per_channel(means, stds)

    # Two samples, both with physical mean 110; one is masked.
    tensors = move_tensordict_to_device({"a": torch.full((2, 4, 4), 110.0)})
    data_mask = move_tensordict_to_device({"a": torch.tensor([True, False])})
    shifted = transform.forward_transform(tensors, data_mask)
    normalized = normalizer.normalize(shifted)
    # Unmasked sample: spatial mean ≈ 0 in normalized space.
    assert normalized["a"][0].mean().abs().item() < 1e-5
    # Masked sample: not shifted, so it retains (110 - 100) / 5 = 2.0
    # — this is fine because _apply_input_mask zeros the channel later.
    assert normalized["a"][1].mean().item() == pytest.approx(2.0)


# ── Config validation ───────────────────────────────────────────────────


def test_shared_config_validates_reference_field():
    config = SharedGlobalMeanRemovalConfig(reference_field="missing")
    with pytest.raises(ValueError, match="reference_field"):
        config.validate_names(["a", "b"], ["a", "b"])


def test_shared_config_warns_on_unknown_field_names(caplog):
    config = SharedGlobalMeanRemovalConfig(
        reference_field="a", field_names=["a", "missing"]
    )
    config.validate_names(["a", "b"], ["a", "b"])
    assert "will have no effect" in caplog.text


def test_shared_config_warns_on_unknown_qsat_scaled_names(caplog):
    config = SharedGlobalMeanRemovalConfig(
        reference_field="a", field_names=["a"], qsat_scaled_names=["missing_q"]
    )
    config.validate_names(["a", "b"], ["a", "b"])
    assert "missing_q" in caplog.text
    assert "will have no effect" in caplog.text


def test_per_channel_config_warns_on_unknown_field_names(caplog):
    config = PerChannelGlobalMeanRemovalConfig(field_names=["missing"])
    config.validate_names(["a", "b"], ["a", "b"])
    assert "will have no effect" in caplog.text


def test_per_channel_config_raises_on_output_only_field():
    config = PerChannelGlobalMeanRemovalConfig(field_names=["c"])
    with pytest.raises(ValueError, match="field_name"):
        config.validate_names(["a", "b"], ["a", "b", "c"])


def test_per_channel_config_none_field_names_means_all():
    means = {"a": 0.0, "b": 0.0}
    stds = {"a": 1.0, "b": 1.0}
    normalizer = _build_normalizer_for(means, stds)
    config = PerChannelGlobalMeanRemovalConfig(field_names=None)
    config.validate_names(["a", "b"], ["a", "b"])
    assert config.build(normalizer, ["a", "b"]).n_extra_input_channels == 0
    config_with_input = PerChannelGlobalMeanRemovalConfig(
        field_names=None, append_as_input=True
    )
    assert config_with_input.build(normalizer, ["a", "b"]).n_extra_input_channels == 2


# ── Dacite union serialization ──────────────────────────────────────────


def test_shared_config_round_trips_through_dacite():
    config = SharedGlobalMeanRemovalConfig(
        reference_field="surface_temperature",
        field_names=["surface_temperature", "air_temperature_0"],
        append_as_input=True,
        qsat_scaled_names=["specific_humidity_0"],
    )
    data = dataclasses.asdict(config)
    restored = dacite.from_dict(
        SharedGlobalMeanRemovalConfig, data, dacite.Config(strict=True)
    )
    assert config == restored


def test_per_channel_config_round_trips_through_dacite():
    config = PerChannelGlobalMeanRemovalConfig(
        field_names=["a", "b"],
        append_as_input=True,
    )
    data = dataclasses.asdict(config)
    restored = dacite.from_dict(
        PerChannelGlobalMeanRemovalConfig, data, dacite.Config(strict=True)
    )
    assert config == restored


def test_union_resolved_by_kind_via_dacite():
    @dataclasses.dataclass
    class Container:
        removal: GlobalMeanRemovalConfigUnion | None = None

    shared_data = {
        "removal": {
            "kind": "shared",
            "reference_field": "surface_temperature",
            "field_names": ["surface_temperature"],
            "append_as_input": False,
        }
    }
    result = dacite.from_dict(Container, shared_data, dacite.Config(strict=True))
    assert isinstance(result.removal, SharedGlobalMeanRemovalConfig)

    per_ch_data = {
        "removal": {
            "kind": "per_channel",
            "field_names": ["a"],
            "append_as_input": True,
        }
    }
    result2 = dacite.from_dict(Container, per_ch_data, dacite.Config(strict=True))
    assert isinstance(result2.removal, PerChannelGlobalMeanRemovalConfig)

    none_data: dict = {}
    result3 = dacite.from_dict(Container, none_data, dacite.Config(strict=True))
    assert result3.removal is None

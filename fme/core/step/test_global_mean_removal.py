import dataclasses

import dacite
import pytest
import torch

from fme.core.device import move_tensordict_to_device
from fme.core.step.global_mean_removal import (
    GlobalMeanRemovalConfigUnion,
    PerChannelGlobalMeanRemovalConfig,
    SharedGlobalMeanRemovalConfig,
)


def _build_normalizer_for(means, stds):
    from fme.core.normalizer import NormalizationConfig

    return NormalizationConfig(means=means, stds=stds).build(list(means))


# ── SharedGlobalMeanRemoval ─────────────────────────────────────────────


def _make_shared(means, stds, append_as_input=False):
    normalizer = _build_normalizer_for(means, stds)
    config = SharedGlobalMeanRemovalConfig(
        reference_field="surface_temperature",
        field_names=list(means.keys()),
        append_as_input=append_as_input,
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
    tensors = move_tensordict_to_device(
        {"surface_temperature": torch.full((2, 4, 4), 285.0)}
    )
    transform.forward_transform(tensors, None)
    assert transform.get_extra_channels() is None


def test_shared_extra_channels():
    means = {"surface_temperature": 280.0}
    stds = {"surface_temperature": 5.0}
    transform = _make_shared(means, stds, append_as_input=True)
    assert transform.n_extra_input_channels == 1
    tensors = move_tensordict_to_device(
        {"surface_temperature": torch.full((2, 4, 4), 285.0)}
    )
    transform.forward_transform(tensors, None)
    extra = transform.get_extra_channels()
    assert extra is not None
    assert extra.shape == (2, 1, 4, 4)
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
    # a: [12 - 13, 14 - 13] = [-1, 1]
    torch.testing.assert_close(result["a"].cpu(), torch.tensor([[[-1.0, 1.0]]]))
    # b: [22 - 25, 28 - 25] = [-3, 3]
    torch.testing.assert_close(result["b"].cpu(), torch.tensor([[[-3.0, 3.0]]]))


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
    # sample 0: mean=15, sample 1: mean=150
    torch.testing.assert_close(
        result["a"].cpu(), torch.tensor([[[-5.0, 5.0]], [[-50.0, 50.0]]])
    )


def test_per_channel_no_extra_channels():
    means = {"a": 0.0}
    stds = {"a": 1.0}
    transform = _make_per_channel(means, stds)
    assert transform.n_extra_input_channels == 0
    tensors = move_tensordict_to_device({"a": torch.full((2, 4, 4), 5.0)})
    transform.forward_transform(tensors, None)
    assert transform.get_extra_channels() is None


def test_per_channel_extra_channels():
    means = {"a": 10.0, "b": 20.0}
    stds = {"a": 2.0, "b": 5.0}
    transform = _make_per_channel(means, stds, append_as_input=True)
    assert transform.n_extra_input_channels == 2
    tensors = move_tensordict_to_device(
        {
            "a": torch.full((1, 4, 4), 14.0),  # mean=14
            "b": torch.full((1, 4, 4), 25.0),  # mean=25
        }
    )
    transform.forward_transform(tensors, None)
    extra = transform.get_extra_channels()
    assert extra is not None
    assert extra.shape == (1, 2, 4, 4)
    # a: (14 - 10) / 2 = 2.0
    torch.testing.assert_close(extra[:, 0].cpu(), torch.full((1, 4, 4), 2.0))
    # b: (25 - 20) / 5 = 1.0
    torch.testing.assert_close(extra[:, 1].cpu(), torch.full((1, 4, 4), 1.0))


def test_per_channel_with_field_names_subset():
    means = {"a": 10.0, "b": 20.0}
    stds = {"a": 2.0, "b": 5.0}
    transform = _make_per_channel(means, stds, field_names=["a"])
    tensors = move_tensordict_to_device(
        {"a": torch.full((1, 4, 4), 14.0), "b": torch.full((1, 4, 4), 25.0)}
    )
    result = transform.forward_transform(tensors, None)
    # a was shifted
    torch.testing.assert_close(result["a"].cpu(), torch.full((1, 4, 4), 0.0))
    # b was NOT shifted
    torch.testing.assert_close(result["b"].cpu(), torch.full((1, 4, 4), 25.0))


def test_per_channel_masked_uses_zero():
    means = {"a": 10.0}
    stds = {"a": 2.0}
    transform = _make_per_channel(means, stds, append_as_input=True)
    tensors = move_tensordict_to_device(
        {"a": torch.tensor([[[14.0, 14.0]], [[14.0, 14.0]]])}
    )
    data_mask = move_tensordict_to_device({"a": torch.tensor([True, False])})
    result = transform.forward_transform(tensors, data_mask)
    # sample 0 (unmasked): mean=14, shifted to 0
    torch.testing.assert_close(result["a"][0].cpu(), torch.zeros(1, 2))
    # sample 1 (masked): mean→0, so no shift (14 - 0 = 14)
    torch.testing.assert_close(result["a"][1].cpu(), torch.full((1, 2), 14.0))

    extra = transform.get_extra_channels()
    assert extra is not None
    # sample 0: (14 - 10) / 2 = 2.0
    assert extra[0, 0, 0, 0].item() == pytest.approx(2.0)
    # sample 1 (masked): (0 - 10) / 2 = -5.0
    assert extra[1, 0, 0, 0].item() == pytest.approx(-5.0)


def test_per_channel_inverse_before_forward_raises():
    means = {"a": 0.0}
    stds = {"a": 1.0}
    transform = _make_per_channel(means, stds)
    with pytest.raises(RuntimeError, match="forward_transform"):
        transform.inverse_transform({"a": torch.zeros(1, 4, 4)})


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


def test_per_channel_config_warns_on_unknown_field_names(caplog):
    config = PerChannelGlobalMeanRemovalConfig(field_names=["missing"])
    config.validate_names(["a", "b"], ["a", "b"])
    assert "will have no effect" in caplog.text


def test_per_channel_config_raises_on_output_only_field():
    config = PerChannelGlobalMeanRemovalConfig(field_names=["c"])
    with pytest.raises(ValueError, match="field_name"):
        config.validate_names(["a", "b"], ["a", "b", "c"])


def test_per_channel_config_none_field_names_means_all():
    config = PerChannelGlobalMeanRemovalConfig(field_names=None)
    config.validate_names(["a", "b"], ["a", "b"])
    assert config.get_n_extra_input_channels(["a", "b"]) == 0
    config_with_input = PerChannelGlobalMeanRemovalConfig(
        field_names=None, append_as_input=True
    )
    assert config_with_input.get_n_extra_input_channels(["a", "b"]) == 2


# ── Dacite union serialization ──────────────────────────────────────────


def test_shared_config_round_trips_through_dacite():
    config = SharedGlobalMeanRemovalConfig(
        reference_field="surface_temperature",
        field_names=["surface_temperature", "air_temperature_0"],
        append_as_input=True,
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

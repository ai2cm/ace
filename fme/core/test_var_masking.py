import pytest
import torch

from fme.core.var_masking import VariableMaskingConfig


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"default_rate": -0.1}, "default_rate"),
        ({"default_rate": 1.1}, "default_rate"),
        ({"rates": {"T": -0.1}}, "'T'"),
        ({"rates": {"T": 1.5}}, "'T'"),
    ],
)
def test_variable_masking_config_invalid_rate_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        VariableMaskingConfig(**kwargs)


def test_sample_masks_rate_zero_returns_empty():
    config = VariableMaskingConfig(default_rate=0.0, rates={"T": 0.0})
    masks = config.sample_masks(["T", "q"], batch_size=8)
    assert masks == {}


def test_sample_masks_rate_one_always_masks():
    config = VariableMaskingConfig(rates={"T": 1.0})
    masks = config.sample_masks(["T", "q"], batch_size=4)
    assert "T" in masks
    assert not masks["T"].any()  # all False = all masked
    assert "q" not in masks


def test_sample_masks_prefix_groups_levels_together():
    """Variables sharing a prefix key must receive the same mask."""
    config = VariableMaskingConfig(rates={"air_temperature_": 1.0})
    names = ["air_temperature_0", "air_temperature_1", "air_temperature_2"]
    masks = config.sample_masks(names, batch_size=4)
    assert set(masks.keys()) == set(names)
    assert (masks["air_temperature_0"] == masks["air_temperature_1"]).all()
    assert (masks["air_temperature_0"] == masks["air_temperature_2"]).all()


def test_sample_masks_zero_rate_variable_never_masked():
    config = VariableMaskingConfig(rates={"T": 1.0, "SST": 0.0})
    masks = config.sample_masks(["T", "SST", "q"], batch_size=4)
    assert "T" in masks
    assert "SST" not in masks


def test_sample_masks_device_placement():
    config = VariableMaskingConfig(rates={"T": 1.0})
    masks = config.sample_masks(["T"], batch_size=4, device=torch.device("cpu"))
    assert masks["T"].device == torch.device("cpu")

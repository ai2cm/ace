import pytest
import torch

from fme.core.var_masking import UniformVariableMaskingConfig, VariableMaskingConfig


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


# --- UniformVariableMaskingConfig tests ---


def test_uniform_invalid_min_vars_negative():
    with pytest.raises(ValueError, match="min_vars"):
        UniformVariableMaskingConfig(min_vars=-1)


def test_uniform_invalid_max_vars_negative():
    with pytest.raises(ValueError, match="max_vars"):
        UniformVariableMaskingConfig(max_vars=-1)


def test_uniform_invalid_min_gt_max():
    with pytest.raises(ValueError, match="min_vars"):
        UniformVariableMaskingConfig(min_vars=3, max_vars=1)


def test_uniform_min_max_strings_accepted():
    config = UniformVariableMaskingConfig(min_vars="min", max_vars="max")
    assert config.min_vars == "min"
    assert config.max_vars == "max"


def test_uniform_min_vars_exceeds_eligible_raises():
    config = UniformVariableMaskingConfig(min_vars=5)
    with pytest.raises(ValueError, match="min_vars"):
        config.sample_masks(["a", "b"], batch_size=4)


def test_uniform_max_vars_exceeds_eligible_raises():
    config = UniformVariableMaskingConfig(max_vars=5)
    with pytest.raises(ValueError, match="max_vars"):
        config.sample_masks(["a", "b"], batch_size=4)


def test_uniform_ignore_vars_excluded():
    config = UniformVariableMaskingConfig(min_vars=1, max_vars=2, ignore_vars=["SST"])
    names = ["T", "q", "SST"]
    masks = config.sample_masks(names, batch_size=32)
    assert "SST" not in masks


def test_uniform_mask_count_in_range():
    config = UniformVariableMaskingConfig(min_vars=1, max_vars=2)
    names = ["a", "b", "c", "d"]
    masks = config.sample_masks(names, batch_size=100)
    for i in range(100):
        n_masked = sum(
            1 for name in names if name in masks and not masks[name][i].item()
        )
        assert 1 <= n_masked <= 2


def test_uniform_min_max_zero_returns_empty():
    config = UniformVariableMaskingConfig(min_vars=0, max_vars=0)
    masks = config.sample_masks(["a", "b"], batch_size=4)
    assert masks == {}


def test_uniform_max_vars_max_string():
    names = ["a", "b", "c"]
    config = UniformVariableMaskingConfig(min_vars=len(names), max_vars="max")
    masks = config.sample_masks(names, batch_size=8)
    # all samples must have all eligible vars masked
    for name in names:
        assert name in masks
        assert not masks[name].any()


def test_uniform_device_placement():
    config = UniformVariableMaskingConfig(min_vars=1, max_vars=1)
    masks = config.sample_masks(["a", "b"], batch_size=4, device=torch.device("cpu"))
    for tensor in masks.values():
        assert tensor.device == torch.device("cpu")

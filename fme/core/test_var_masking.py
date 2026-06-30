import pytest
import torch

from fme.core.var_masking import (
    PerVariableMaskingConfig,
    UniformMaskingConfig,
    VariableMaskingConfig,
)


def test_per_variable_masking_config_rate_validation():
    with pytest.raises(ValueError, match="rate"):
        PerVariableMaskingConfig(rate=-0.1)
    with pytest.raises(ValueError, match="rate"):
        PerVariableMaskingConfig(rate=1.1)


def test_uniform_masking_config_validation():
    with pytest.raises(ValueError, match="max_masked_vars"):
        UniformMaskingConfig(max_masked_vars=-1)


def test_uniform_mask_shape_and_dtype():
    config = UniformMaskingConfig(max_masked_vars=3)
    mask = config.sample_mask(n_channels=10, batch_size=4, device=torch.device("cpu"))
    assert mask.shape == (4, 10)
    assert mask.dtype == torch.bool


def test_uniform_mask_counts_in_range():
    config = UniformMaskingConfig(max_masked_vars=4)
    n_channels = 8
    batch_size = 64
    mask = config.sample_mask(n_channels, batch_size, torch.device("cpu"))
    # number of masked (False) variables per sample should be in [0, 4]
    n_masked = (~mask).sum(dim=1)
    assert (n_masked >= 0).all()
    assert (n_masked <= 4).all()


def test_uniform_mask_minimum_masked_count_is_zero():
    """A window may now have no masked variables (min count dropped to 0)."""
    config = UniformMaskingConfig(max_masked_vars=2)
    mask = config.sample_mask(n_channels=5, batch_size=512, device=torch.device("cpu"))
    n_masked = (~mask).sum(dim=1)
    assert (n_masked == 0).any(), "some samples should have zero masked variables"


def test_uniform_mask_defaults_max():
    n_channels = 5
    batch_size = 16
    config = UniformMaskingConfig()
    mask = config.sample_mask(n_channels, batch_size, torch.device("cpu"))
    n_masked = (~mask).sum(dim=1)
    assert (n_masked >= 0).all()
    assert (n_masked <= n_channels).all()


def test_per_variable_mask_shape_and_dtype():
    config = PerVariableMaskingConfig(rate=0.5)
    mask = config.sample_mask(n_channels=10, batch_size=8, device=torch.device("cpu"))
    assert mask.shape == (8, 10)
    assert mask.dtype == torch.bool


def test_per_variable_mask_rate_zero_keeps_all():
    config = PerVariableMaskingConfig(rate=0.0)
    mask = config.sample_mask(n_channels=20, batch_size=32, device=torch.device("cpu"))
    assert mask.all()


def test_per_variable_mask_rate_one_drops_all():
    config = PerVariableMaskingConfig(rate=1.0)
    mask = config.sample_mask(n_channels=20, batch_size=32, device=torch.device("cpu"))
    assert not mask.any()


def test_variable_masking_config_is_union_of_sub_configs():
    assert VariableMaskingConfig == UniformMaskingConfig | PerVariableMaskingConfig

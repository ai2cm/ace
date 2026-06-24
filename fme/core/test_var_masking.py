import pytest
import torch

from fme.core.var_masking import (
    CO2_NAME,
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


def test_uniform_same_mask_per_batch_rows_identical():
    config = UniformMaskingConfig(max_masked_vars=3, same_mask_per_batch=True)
    mask = config.sample_mask(n_channels=10, batch_size=8, device=torch.device("cpu"))
    assert mask.shape == (8, 10)
    assert mask.dtype == torch.bool
    assert (mask == mask[0]).all(), "all rows should be identical"


def test_uniform_same_mask_per_batch_counts_identical():
    config = UniformMaskingConfig(max_masked_vars=4, same_mask_per_batch=True)
    mask = config.sample_mask(n_channels=8, batch_size=64, device=torch.device("cpu"))
    n_masked = (~mask).sum(dim=1)
    assert (n_masked == n_masked[0]).all(), "masked count identical across rows"


def test_per_variable_same_mask_per_batch_rows_identical():
    config = PerVariableMaskingConfig(rate=0.5, same_mask_per_batch=True)
    mask = config.sample_mask(n_channels=10, batch_size=8, device=torch.device("cpu"))
    assert mask.shape == (8, 10)
    assert mask.dtype == torch.bool
    assert (mask == mask[0]).all(), "all rows should be identical"


def test_variable_masking_config_is_union_of_sub_configs():
    assert VariableMaskingConfig == UniformMaskingConfig | PerVariableMaskingConfig


@pytest.mark.parametrize(
    "make_config",
    [
        lambda co2_rate: UniformMaskingConfig(co2_rate=co2_rate),
        lambda co2_rate: PerVariableMaskingConfig(co2_rate=co2_rate),
    ],
)
def test_co2_rate_validation(make_config):
    with pytest.raises(ValueError, match="co2_rate"):
        make_config(-0.1)
    with pytest.raises(ValueError, match="co2_rate"):
        make_config(1.1)


@pytest.mark.parametrize(
    "config",
    [
        UniformMaskingConfig(max_masked_vars=0, co2_rate=0.3),
        PerVariableMaskingConfig(rate=0.0, co2_rate=0.3),
    ],
)
def test_co2_override_keep_rate(config):
    n_channels = 4
    co2_idx = 2
    channel_names = ["a", "b", CO2_NAME, "c"]
    mask = config.sample_mask(
        n_channels=n_channels,
        batch_size=4096,
        device=torch.device("cpu"),
        channel_names=channel_names,
    )
    # base config keeps all non-CO2 channels (rate/count is zero)
    other = mask[:, [i for i in range(n_channels) if i != co2_idx]]
    assert other.all(), "non-CO2 channels should all be kept"
    # CO2 column keep-rate matches 1 - co2_rate
    keep_rate = mask[:, co2_idx].float().mean().item()
    assert abs(keep_rate - 0.7) < 0.05


@pytest.mark.parametrize(
    "config",
    [
        UniformMaskingConfig(co2_rate=0.3),
        PerVariableMaskingConfig(co2_rate=0.3),
    ],
)
def test_co2_rate_set_but_co2_absent_raises(config):
    with pytest.raises(ValueError, match=CO2_NAME):
        config.sample_mask(
            n_channels=3,
            batch_size=2,
            device=torch.device("cpu"),
            channel_names=["a", "b", "c"],
        )
    with pytest.raises(ValueError, match=CO2_NAME):
        config.sample_mask(
            n_channels=3,
            batch_size=2,
            device=torch.device("cpu"),
            channel_names=None,
        )


@pytest.mark.parametrize(
    "config",
    [
        UniformMaskingConfig(max_masked_vars=0, co2_rate=0.5, same_mask_per_batch=True),
        PerVariableMaskingConfig(rate=0.0, co2_rate=0.5, same_mask_per_batch=True),
    ],
)
def test_co2_override_shared_when_same_mask_per_batch(config):
    channel_names = ["a", CO2_NAME, "b", "c"]
    mask = config.sample_mask(
        n_channels=4,
        batch_size=8,
        device=torch.device("cpu"),
        channel_names=channel_names,
    )
    co2_col = mask[:, channel_names.index(CO2_NAME)]
    assert (
        co2_col == co2_col[0]
    ).all(), "same_mask_per_batch must share the CO2 decision across the batch"


@pytest.mark.parametrize(
    "config",
    [
        UniformMaskingConfig(co2_rate=0.3),
        UniformMaskingConfig(co2_rate=None),
        PerVariableMaskingConfig(co2_rate=0.3),
        PerVariableMaskingConfig(co2_rate=None),
    ],
)
def test_validate_names(config):
    if config.co2_rate is None:
        config.validate_names(["a", "b"])
    else:
        with pytest.raises(ValueError, match=CO2_NAME):
            config.validate_names(["a", "b"])
        config.validate_names(["a", CO2_NAME])

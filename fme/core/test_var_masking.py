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
    with pytest.raises(ValueError, match="min_vars"):
        UniformMaskingConfig(min_vars=-1)
    with pytest.raises(ValueError, match="max_vars"):
        UniformMaskingConfig(max_vars=-1)
    with pytest.raises(ValueError, match="min_vars"):
        UniformMaskingConfig(min_vars=5, max_vars=2)


def test_uniform_mask_shape_and_dtype():
    config = UniformMaskingConfig(min_vars=1, max_vars=3)
    mask = config.sample_mask(n_channels=10, batch_size=4, device=torch.device("cpu"))
    assert mask.shape == (4, 10)
    assert mask.dtype == torch.bool


def test_uniform_mask_counts_in_range():
    config = UniformMaskingConfig(min_vars=2, max_vars=4)
    n_channels = 8
    batch_size = 64
    mask = config.sample_mask(n_channels, batch_size, torch.device("cpu"))
    # number of masked (False) channels per sample should be in [2, 4]
    n_masked = (~mask).sum(dim=1)
    assert (n_masked >= 2).all()
    assert (n_masked <= 4).all()


def test_uniform_mask_defaults_min_max():
    n_channels = 5
    batch_size = 16
    config = UniformMaskingConfig()
    mask = config.sample_mask(n_channels, batch_size, torch.device("cpu"))
    n_masked = (~mask).sum(dim=1)
    assert (n_masked >= 1).all()
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


def test_uniform_mask_ensemble_members_share_mask():
    n_samples, n_ensemble, n_channels = 6, 3, 8
    config = UniformMaskingConfig(min_vars=1, max_vars=3)
    mask = config.sample_mask(
        n_channels,
        batch_size=n_samples * n_ensemble,
        device=torch.device("cpu"),
        n_ensemble=n_ensemble,
    )
    assert mask.shape == (n_samples * n_ensemble, n_channels)
    grouped = mask.view(n_samples, n_ensemble, n_channels)
    assert (
        grouped == grouped[:, :1, :]
    ).all(), "All ensemble members of a base sample must share the same mask"


def test_per_variable_mask_ensemble_members_share_mask():
    n_samples, n_ensemble, n_channels = 6, 3, 8
    config = PerVariableMaskingConfig(rate=0.5)
    mask = config.sample_mask(
        n_channels,
        batch_size=n_samples * n_ensemble,
        device=torch.device("cpu"),
        n_ensemble=n_ensemble,
    )
    assert mask.shape == (n_samples * n_ensemble, n_channels)
    grouped = mask.view(n_samples, n_ensemble, n_channels)
    assert (
        grouped == grouped[:, :1, :]
    ).all(), "All ensemble members of a base sample must share the same mask"


def test_sample_mask_raises_on_indivisible_batch():
    config = UniformMaskingConfig()
    with pytest.raises(ValueError, match="divisible"):
        config.sample_mask(
            n_channels=4, batch_size=7, device=torch.device("cpu"), n_ensemble=3
        )


def test_variable_masking_config_is_union_of_sub_configs():
    assert VariableMaskingConfig == UniformMaskingConfig | PerVariableMaskingConfig


@pytest.mark.parametrize(
    "make_config",
    [
        lambda co2_rate: UniformMaskingConfig(co2_rate=co2_rate),
        lambda co2_rate: PerVariableMaskingConfig(co2_rate=co2_rate),
    ],
    ids=["uniform", "per_variable"],
)
def test_co2_rate_validation(make_config):
    with pytest.raises(ValueError, match="co2_rate"):
        make_config(-0.1)
    with pytest.raises(ValueError, match="co2_rate"):
        make_config(1.1)
    # None and valid rates are accepted
    make_config(None)
    make_config(0.5)


@pytest.mark.parametrize(
    "config",
    [
        UniformMaskingConfig(min_vars=0, max_vars=0, co2_rate=0.3),
        PerVariableMaskingConfig(rate=0.0, co2_rate=0.3),
    ],
    ids=["uniform", "per_variable"],
)
def test_co2_override_keep_rate(config):
    n_channels, batch_size = 4, 20000
    co2_idx = 2
    channel_names = ["a", "b", CO2_NAME, "c"]
    mask = config.sample_mask(
        n_channels,
        batch_size,
        torch.device("cpu"),
        channel_names=channel_names,
    )
    # base config keeps all non-CO2 channels (rate/count is zero)
    other = mask[:, [i for i in range(n_channels) if i != co2_idx]]
    assert other.all()
    # CO2 column keep-rate matches 1 - co2_rate
    keep_rate = mask[:, co2_idx].float().mean().item()
    assert abs(keep_rate - (1.0 - 0.3)) < 0.02


@pytest.mark.parametrize(
    "config",
    [
        UniformMaskingConfig(co2_rate=0.3),
        PerVariableMaskingConfig(co2_rate=0.3),
    ],
    ids=["uniform", "per_variable"],
)
def test_co2_rate_set_but_co2_absent_raises(config):
    with pytest.raises(ValueError, match=CO2_NAME):
        config.sample_mask(
            n_channels=3,
            batch_size=4,
            device=torch.device("cpu"),
            channel_names=["a", "b", "c"],
        )
    with pytest.raises(ValueError, match=CO2_NAME):
        config.sample_mask(n_channels=3, batch_size=4, device=torch.device("cpu"))


@pytest.mark.parametrize(
    "config",
    [
        UniformMaskingConfig(min_vars=0, max_vars=0, co2_rate=0.5),
        PerVariableMaskingConfig(rate=0.0, co2_rate=0.5),
    ],
    ids=["uniform", "per_variable"],
)
def test_co2_override_shared_across_ensemble(config):
    n_samples, n_ensemble, n_channels = 8, 3, 4
    channel_names = ["a", CO2_NAME, "b", "c"]
    mask = config.sample_mask(
        n_channels,
        batch_size=n_samples * n_ensemble,
        device=torch.device("cpu"),
        n_ensemble=n_ensemble,
        channel_names=channel_names,
    )
    grouped = mask.view(n_samples, n_ensemble, n_channels)
    assert (
        grouped == grouped[:, :1, :]
    ).all(), "Ensemble members of a base sample must share the CO2 decision"


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
        config.validate_names(["a", "b"])  # no-op
    else:
        with pytest.raises(ValueError, match=CO2_NAME):
            config.validate_names(["a", "b"])
        config.validate_names(["a", CO2_NAME])

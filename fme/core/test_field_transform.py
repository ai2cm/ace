import dataclasses

import pytest
import torch

from fme.core.field_transform import (
    GaussianRankTransformConfig,
    Log1pTransformConfig,
    LogitTransformConfig,
    transform_config_from_dict,
)
from fme.core.normalizer import (
    NormalizationConfig,
    StandardNormalizer,
    _combine_normalizers,
)

SWE_RANGE = torch.tensor([0.0, 0.5, 16.0, 432.0, 1365.0, 8063.0, 10000.0])


def test_log1p_round_trip():
    transform = Log1pTransformConfig().build()
    x = SWE_RANGE
    torch.testing.assert_close(transform.inverse(transform.forward(x)), x)


def test_log1p_clamps_negative_input():
    transform = Log1pTransformConfig().build()
    assert transform.forward(torch.tensor([-0.2])).item() == 0.0


def test_logit_round_trip_and_bounds():
    config = LogitTransformConfig(epsilon=1e-4, scale=1.0)
    transform = config.build()
    p = torch.tensor([1e-4, 0.01, 0.5, 0.99, 1.0 - 1e-4])
    torch.testing.assert_close(transform.inverse(transform.forward(p)), p)
    z = torch.tensor([-100.0, 0.0, 100.0])
    decoded = transform.inverse(z)
    assert (decoded >= 0.0).all() and (decoded <= 1.0).all()


def test_logit_percent_scale():
    transform = LogitTransformConfig(scale=100.0).build()
    p = torch.tensor([1.0, 50.0, 99.0])
    torch.testing.assert_close(transform.inverse(transform.forward(p)), p)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(epsilon=0.0),
        dict(epsilon=0.5),
        dict(scale=0.0),
    ],
)
def test_logit_config_validation(kwargs):
    with pytest.raises(ValueError):
        LogitTransformConfig(**kwargs)


def _gaussian_rank_config() -> GaussianRankTransformConfig:
    return GaussianRankTransformConfig(
        x_knots=[0.0, 1.0, 16.0, 432.0, 10000.0],
        z_knots=[-1.1, 0.0, 0.7, 2.0, 3.5],
    )


def test_gaussian_rank_round_trip_and_monotonicity():
    transform = _gaussian_rank_config().build()
    x = torch.tensor([0.0, 0.5, 1.0, 100.0, 10000.0])
    z = transform.forward(x)
    assert (z[1:] > z[:-1]).all()
    torch.testing.assert_close(transform.inverse(z), x)


def test_gaussian_rank_clamps_out_of_range():
    transform = _gaussian_rank_config().build()
    assert transform.forward(torch.tensor([20000.0])).item() == 3.5
    assert transform.inverse(torch.tensor([-5.0])).item() == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(x_knots=[0.0, 1.0], z_knots=[0.0]),
        dict(x_knots=[0.0, 0.0], z_knots=[0.0, 1.0]),
        dict(x_knots=[0.0], z_knots=[0.0]),
        dict(x_knots=[0.0, 1.0], z_knots=[0.0, 1.0], table_path="foo.nc"),
        dict(),
    ],
)
def test_gaussian_rank_config_validation(kwargs):
    with pytest.raises(ValueError):
        GaussianRankTransformConfig(**kwargs)


@pytest.mark.parametrize(
    "config",
    [
        Log1pTransformConfig(),
        LogitTransformConfig(epsilon=1e-3, scale=100.0),
        _gaussian_rank_config(),
    ],
)
def test_config_dict_round_trip(config):
    rebuilt = transform_config_from_dict(dataclasses.asdict(config))
    assert rebuilt == config


def _normalizer_with_transforms() -> StandardNormalizer:
    return NormalizationConfig(
        means={"surface_snow_amount": 1.5, "TMP2m": 288.0},
        stds={"surface_snow_amount": 2.0, "TMP2m": 10.0},
        transforms={"surface_snow_amount": Log1pTransformConfig()},
    ).build(names=["surface_snow_amount", "TMP2m"])


def test_transformed_normalizer_round_trip():
    normalizer = _normalizer_with_transforms()
    data = {
        "surface_snow_amount": SWE_RANGE.clone(),
        "TMP2m": torch.tensor([250.0, 288.0, 310.0]),
    }
    denormalized = normalizer.denormalize(normalizer.normalize(data))
    torch.testing.assert_close(
        denormalized["surface_snow_amount"], data["surface_snow_amount"]
    )
    torch.testing.assert_close(denormalized["TMP2m"], data["TMP2m"])


def test_transformed_normalizer_normalizes_in_transform_space():
    normalizer = _normalizer_with_transforms()
    x = torch.tensor([16.0])
    z = normalizer.normalize({"surface_snow_amount": x})["surface_snow_amount"]
    expected = (torch.log1p(x) - 1.5) / 2.0
    torch.testing.assert_close(z, expected)


def test_transformed_normalizer_skips_transform_without_mean():
    normalizer = _normalizer_with_transforms()
    delta = torch.tensor([4.0])
    z = normalizer.normalize({"surface_snow_amount": delta}, apply_mean=False)
    torch.testing.assert_close(z["surface_snow_amount"], delta / 2.0)


def test_transformed_normalizer_state_round_trip():
    normalizer = _normalizer_with_transforms()
    restored = StandardNormalizer.from_state(normalizer.get_state())
    data = {"surface_snow_amount": SWE_RANGE.clone()}
    torch.testing.assert_close(
        restored.normalize(data)["surface_snow_amount"],
        normalizer.normalize(data)["surface_snow_amount"],
    )
    assert restored.transform_configs == normalizer.transform_configs


def test_transformed_normalizer_config_round_trip():
    normalizer = _normalizer_with_transforms()
    rebuilt = normalizer.get_normalization_config().build(
        names=["surface_snow_amount", "TMP2m"]
    )
    data = {"surface_snow_amount": SWE_RANGE.clone()}
    torch.testing.assert_close(
        rebuilt.normalize(data)["surface_snow_amount"],
        normalizer.normalize(data)["surface_snow_amount"],
    )


def test_combine_normalizers_preserves_transforms():
    network = _normalizer_with_transforms()
    residual = NormalizationConfig(
        means={"surface_snow_amount": 0.0},
        stds={"surface_snow_amount": 0.5},
        transforms={"surface_snow_amount": Log1pTransformConfig()},
    ).build(names=["surface_snow_amount"])
    combined = _combine_normalizers(network, residual)
    x = torch.tensor([16.0])
    z = combined.normalize({"surface_snow_amount": x})["surface_snow_amount"]
    torch.testing.assert_close(z, torch.log1p(x) / 0.5)
    assert combined.stds["TMP2m"] == network.stds["TMP2m"]

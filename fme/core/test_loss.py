import pytest
import torch

from fme.core import metrics
from fme.core.device import get_device
from fme.core.gridded_ops import GriddedOperations, LatLonOperations
from fme.core.loss import (
    AreaWeightedMSELoss,
    CRPSLoss,
    EnergyScoreLoss,
    FiniteDifferenceCRPSLoss,
    GlobalMeanLoss,
    LossConfig,
    LossOutput,
    StepLossConfig,
    VariableWeightingLoss,
    WeightedMappingLoss,
    _build_channel_mask,
    _construct_weight_tensor,
    _reduce_to_per_channel,
)
from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer


@pytest.mark.parametrize("global_mean_type", [None, "LpLoss"])
def test_loss_builds_and_runs(global_mean_type):
    config = LossConfig(global_mean_type=global_mean_type)
    area = torch.randn(10, 1, device=get_device()).broadcast_to(size=(10, 10))
    loss = config.build(
        reduction="mean",
        gridded_operations=LatLonOperations(area),
    )
    x = torch.randn(10, 10, 10, 10, 10, device=get_device())
    y = torch.randn(10, 10, 10, 10, 10, device=get_device())
    result = loss(x, y)
    assert isinstance(result, torch.Tensor)


def test_spectral_energy_score(very_fast_only: bool):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    torch.manual_seed(0)
    DEVICE = get_device()
    n_lat, n_lon = 16, 32
    pred = torch.rand(10000, 2, n_lat, n_lon, device=DEVICE)
    target = torch.rand(10000, 2, n_lat, n_lon, device=DEVICE)
    sht = LatLonOperations(torch.ones((n_lat, n_lon), device=DEVICE)).get_real_sht()
    spectral_energy_score_loss = EnergyScoreLoss(sht=sht)
    crps_loss = CRPSLoss(alpha=0.95)
    score = spectral_energy_score_loss(pred, target)
    crps = crps_loss(pred, target)

    n_lat2, n_lon2 = 32, 64
    pred = torch.rand(10000, 2, n_lat2, n_lon2, device=DEVICE)
    target = torch.rand(10000, 2, n_lat2, n_lon2, device=DEVICE)
    sht = LatLonOperations(torch.ones((n_lat2, n_lon2), device=DEVICE)).get_real_sht()
    spectral_energy_score_loss = EnergyScoreLoss(sht=sht)
    larger_domain_score = spectral_energy_score_loss(pred, target)
    torch.testing.assert_close(larger_domain_score, score, rtol=0.05, atol=0.0)
    torch.testing.assert_close(score, crps, rtol=0.5, atol=0.0)


def test_loss_of_zeros_is_variance():
    torch.manual_seed(0)
    config = LossConfig(global_mean_type=None)
    loss = config.build(
        reduction="mean",
        gridded_operations=LatLonOperations(torch.ones(10, 10)),
    )
    x = torch.zeros(10, 10, 10, 10, 10, device=get_device())
    y = torch.randn(10, 10, 10, 10, 10, device=get_device())
    result = loss(x, y)
    assert isinstance(result, torch.Tensor)
    if str(get_device()).startswith("cuda"):
        tol = {"rtol": 1e-4, "atol": 1e-4}
    elif str(get_device()).startswith("mps"):
        tol = {"rtol": 1e-3, "atol": 1e-3}
    else:
        tol = {}
    torch.testing.assert_close(result.mean(), y.var(), **tol)


@pytest.mark.parametrize("global_mean_weight", [0.0, 1.0, 5.0])
def test_loss_of_zeros_is_one_plus_global_mean_weight(global_mean_weight: float):
    torch.manual_seed(0)
    config = LossConfig(
        global_mean_type="LpLoss", global_mean_weight=global_mean_weight
    )
    area = torch.randn(10, 1, device=get_device()).broadcast_to(size=(10, 10))
    loss = config.build(
        reduction="mean",
        gridded_operations=LatLonOperations(area),
    )
    x = torch.zeros(10, 10, 10, 10, 10, device=get_device())
    y = torch.randn(10, 10, 10, 10, 10, device=get_device())
    result = loss(x, y)
    assert isinstance(result, torch.Tensor)
    expected = torch.tensor(1.0 + global_mean_weight)
    tol = (
        {"atol": 0.015, "rtol": 0.01}
        if str(get_device()).startswith("cuda")
        else {"atol": 0.01, "rtol": 0.0}
    )
    torch.testing.assert_close(result.mean().cpu(), expected, **tol)


@pytest.mark.parametrize(
    "config",
    [
        LossConfig(type="AreaWeightedMSE"),
        LossConfig(global_mean_type="LpLoss"),
        LossConfig(
            type="EnsembleLoss",
            kwargs={
                "energy_score_weight": 1.0,
                "crps_weight": 0.0,
                "finite_difference_crps_weight": 0.05,
            },
        ),
    ],
)
def test_loss_fails_when_gridded_operations_not_provided(
    config: LossConfig,
):
    with pytest.raises(ValueError):
        config.build(reduction="mean", gridded_operations=None)


def test_global_mean_loss():
    torch.manual_seed(0)
    area = torch.randn(10, 1, device=get_device()).broadcast_to(size=(10, 10))
    loss = GlobalMeanLoss(
        LatLonOperations(area).area_weighted_mean, loss=torch.nn.MSELoss()
    )
    x = torch.zeros(10, 10, 10, 10, 10, device=get_device())
    y = torch.randn(10, 10, 10, 10, 10, device=get_device())
    result = loss(x, y)
    assert result.shape == ()
    assert isinstance(result, torch.Tensor)

    def global_weighted_mean(tensor, area):
        return (tensor * area[None, None, None, :, :]).sum(dim=(-1, -2)) / area.sum()

    mse = torch.nn.MSELoss()
    expected = mse(global_weighted_mean(x, area), global_weighted_mean(y, area))
    torch.testing.assert_close(result, expected)


def test_area_weighted_mse():
    torch.manual_seed(0)
    x = torch.rand(10, 10).to(get_device())
    target = torch.rand(10, 10).to(get_device())
    area = torch.rand(10, 1, device=get_device()).broadcast_to(size=(10, 10))
    area_weighted_mse = AreaWeightedMSELoss(LatLonOperations(area).area_weighted_mean)
    result = area_weighted_mse(x, target)
    expected = metrics.weighted_mean(
        torch.nn.MSELoss(reduction="none")(x, target), weights=area, dim=(-2, -1)
    ).mean()
    torch.testing.assert_close(result, expected)


def test__construct_weight_tensor():
    out_names = ["a", "b", "c"]
    weights = {"a": 0.5, "c": 3.0}
    n_samples = 10
    nlat, nlon = 8, 4
    gen_data = torch.rand(n_samples, len(out_names), nlat, nlon, device=get_device())
    weight_tensor = _construct_weight_tensor(
        weights, out_names, n_dim=4, channel_dim=-3
    )
    weighted_gen_data = gen_data * weight_tensor
    assert weight_tensor.shape == (1, len(out_names), 1, 1)
    assert weighted_gen_data.shape == gen_data.shape
    for i, name in enumerate(out_names):
        if name in weights:
            assert torch.allclose(
                weighted_gen_data[:, i], weights[name] * gen_data[:, i]
            )
        else:
            assert torch.allclose(weighted_gen_data[:, i], gen_data[:, i])


@pytest.mark.parametrize("mean", [0.0, 1.0])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_WeightedMappingLoss(mean, scale):
    loss = torch.nn.MSELoss(reduction="none")
    n_channels = 5
    packer = Packer([f"var_{i}" for i in range(n_channels)])
    out_names = [f"var_{i}" for i in range(n_channels)]
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(mean) for name in out_names},
        stds={name: torch.as_tensor(scale) for name in out_names},
    )
    mapping_loss = WeightedMappingLoss(
        loss,
        weights={},
        out_names=out_names,
        normalizer=normalizer,
    )
    x = torch.randn(15, n_channels, 10, 10).to(get_device(), dtype=torch.float)
    y = torch.randn(15, n_channels, 10, 10).to(get_device(), dtype=torch.float)
    x_mapping = {name: x[:, i, :, :] for i, name in enumerate(packer.names)}
    y_mapping = {name: y[:, i, :, :] for i, name in enumerate(packer.names)}
    result = mapping_loss(x_mapping, y_mapping)
    assert isinstance(result, LossOutput)
    expected_scalar = torch.nn.MSELoss()(x, y) / scale**2
    torch.testing.assert_close(result.total(), expected_scalar)
    channel_losses = result.get_channel_losses()
    assert set(channel_losses.keys()) == set(out_names)
    torch.testing.assert_close(sum(channel_losses.values()), expected_scalar)


def test_VariableWeightingLoss():
    weights = torch.tensor(
        [
            4.0,
            1.0,
        ]
    ).to(get_device())
    mse_loss = torch.nn.MSELoss()
    weighted_loss = VariableWeightingLoss(weights=weights, loss=mse_loss)

    x = torch.tensor([[1.0, 2.0]]).to(get_device())
    y = torch.tensor([[2, 2.5]]).to(get_device())

    weighted_result = weighted_loss(x, y)
    assert weighted_result == ((16 + 0.25) / 2.0)


def test_StepLossConfig_no_weights():
    loss_config = LossConfig()
    n_channels = 5
    out_names = [f"var_{i}" for i in range(n_channels)]
    channel_dim = -3
    area = torch.ones(1, 1)  # area not used by this config
    gridded_operations: GriddedOperations = LatLonOperations(area)
    mapping_loss_config = StepLossConfig(sqrt_loss_step_decay_constant=0.0)
    loss = loss_config.build(reduction="mean", gridded_operations=gridded_operations)
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(0.0) for name in out_names},
        stds={name: torch.as_tensor(1.0) for name in out_names},
    )
    mapping_loss = mapping_loss_config.build(
        gridded_operations,
        out_names=out_names,
        channel_dim=channel_dim,
        normalizer=normalizer,
    )
    packer = Packer(out_names)

    x_mapping = {name: torch.randn(4, 5, 5).to(get_device()) for name in out_names}
    y_mapping = {name: torch.randn(4, 5, 5).to(get_device()) for name in out_names}
    x = packer.pack(x_mapping, axis=channel_dim)
    y = packer.pack(y_mapping, axis=channel_dim)

    expected = loss(x, y)
    result_step0 = mapping_loss(x_mapping, y_mapping, step=0)
    result_step1 = mapping_loss(x_mapping, y_mapping, step=1)
    assert isinstance(result_step0, LossOutput)
    torch.testing.assert_close(expected, result_step0.total())
    torch.testing.assert_close(expected, result_step1.total())


def test_StepLossConfig_weights():
    out_names = ["var_0", "var_1"]
    channel_dim = -3
    area = torch.ones(1, 1)  # area not used by this config
    mapping_loss_config = StepLossConfig(
        type="MSE", weights={"var_0": 4.0, "var_1": 1.0}
    )
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(0.0) for name in out_names},
        stds={name: torch.as_tensor(1.0) for name in out_names},
    )

    gridded_operations: GriddedOperations = LatLonOperations(area)

    mapping_loss = mapping_loss_config.build(
        gridded_operations,
        out_names=out_names,
        channel_dim=channel_dim,
        normalizer=normalizer,
    )

    x0 = torch.ones(4, 5, 5).to(get_device())
    x1 = 2.0 * x0
    y0 = 2.0 * x0
    y1 = 2.5 * x0

    x_mapping = {"var_0": x0, "var_1": x1}
    y_mapping = {"var_0": y0, "var_1": y1}

    result = mapping_loss(x_mapping, y_mapping, step=0)
    assert isinstance(result, LossOutput)
    expected = torch.tensor((16 + 0.25) / 2.0, device=get_device())
    torch.testing.assert_close(result.total(), expected)
    channel_losses = result.get_channel_losses()
    assert set(channel_losses.keys()) == set(out_names)
    torch.testing.assert_close(sum(channel_losses.values()), expected)


@pytest.mark.parametrize("sqrt_loss_step_decay_constant", [0.0, 0.1, 1.0])
def test_StepLossConfig_with_step_loss_decay(sqrt_loss_step_decay_constant):
    out_names = ["var_0", "var_1"]
    channel_dim = -3
    area = torch.ones(1, 1)  # area not used by this config
    mapping_loss_config = StepLossConfig(
        type="MSE",
        weights={"var_0": 4.0, "var_1": 1.0},
        sqrt_loss_step_decay_constant=sqrt_loss_step_decay_constant,
    )
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(0.0) for name in out_names},
        stds={name: torch.as_tensor(1.0) for name in out_names},
    )

    gridded_operations: GriddedOperations = LatLonOperations(area)

    mapping_loss = mapping_loss_config.build(
        gridded_operations,
        out_names=out_names,
        channel_dim=channel_dim,
        normalizer=normalizer,
    )

    x0 = torch.ones(4, 5, 5).to(get_device())
    x1 = 2.0 * x0
    y0 = 2.0 * x0
    y1 = 2.5 * x0

    x_mapping = {"var_0": x0, "var_1": x1}
    y_mapping = {"var_0": y0, "var_1": y1}

    torch.testing.assert_close(
        mapping_loss(x_mapping, y_mapping, step=0).total(),
        mapping_loss(x_mapping, y_mapping, step=1).total()
        * (1 + sqrt_loss_step_decay_constant) ** 0.5,
    )
    torch.testing.assert_close(
        mapping_loss(x_mapping, y_mapping, step=0).total(),
        mapping_loss(x_mapping, y_mapping, step=2).total()
        * (1 + sqrt_loss_step_decay_constant * 2) ** 0.5,
    )


@pytest.mark.parametrize("mean", [0.0, 1.0])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_WeightedMappingLoss_with_ensemble_dim(mean, scale):
    loss = torch.nn.MSELoss(reduction="none")
    n_channels = 5
    n_ensemble = 3
    packer = Packer([f"var_{i}" for i in range(n_channels)])
    out_names = [f"var_{i}" for i in range(n_channels)]
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(mean) for name in out_names},
        stds={name: torch.as_tensor(scale) for name in out_names},
    )
    mapping_loss = WeightedMappingLoss(
        loss,
        weights={},
        out_names=out_names,
        normalizer=normalizer,
    )
    x = torch.randn(15, n_ensemble, n_channels, 10, 10).to(
        get_device(), dtype=torch.float
    )
    y = torch.randn(15, 1, n_channels, 10, 10).to(get_device(), dtype=torch.float)
    x_mapping = {name: x[:, :, i, :, :] for i, name in enumerate(packer.names)}
    y_mapping = {name: y[:, :, i, :, :] for i, name in enumerate(packer.names)}
    result = mapping_loss(x_mapping, y_mapping)
    assert isinstance(result, LossOutput)
    expected_scalar = torch.nn.MSELoss()(x, y) / scale**2
    torch.testing.assert_close(result.total(), expected_scalar)
    channel_losses = result.get_channel_losses()
    assert set(channel_losses.keys()) == set(out_names)


def test_WeightedMappingLoss_with_target_nans():
    loss = torch.nn.MSELoss(reduction="none")
    n_channels = 5
    packer = Packer([f"var_{i}" for i in range(n_channels)])
    out_names = [f"var_{i}" for i in range(n_channels)]
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(0.0) for name in out_names},
        stds={name: torch.as_tensor(1.0) for name in out_names},
    )
    mapping_loss = WeightedMappingLoss(
        loss,
        weights={},
        out_names=out_names,
        normalizer=normalizer,
    )
    x = torch.randn(15, n_channels, 10, 10).to(get_device(), dtype=torch.float)
    y = torch.randn(15, n_channels, 10, 10).to(get_device(), dtype=torch.float)
    x_mapping = {name: x[:, i, :, :].clone() for i, name in enumerate(packer.names)}
    y_mapping = {name: y[:, i, :, :].clone() for i, name in enumerate(packer.names)}
    y_mapping[packer.names[0]][:, :, 0] = float("nan")
    x[:, 0, :, 0] = 0.0
    y[:, 0, :, 0] = 0.0
    result = mapping_loss(x_mapping, y_mapping)
    assert isinstance(result, LossOutput)
    mse = torch.nn.MSELoss()
    expected = mse(x, y)
    torch.testing.assert_close(result.total(), expected)
    channel_losses = result.get_channel_losses()
    assert set(channel_losses.keys()) == set(out_names)


def test_reduce_to_per_channel():
    """Unit test for _reduce_to_per_channel covering scalar, 1D, and N-D inputs."""
    n_c = 3
    channel_dim = 1

    elementwise = torch.randn(4, n_c, 8, 8, device=get_device())
    result = _reduce_to_per_channel(elementwise, channel_dim, n_c)
    assert result.shape == (n_c,)
    torch.testing.assert_close(result.sum(), elementwise.mean())

    scalar = torch.tensor(6.0, device=get_device())
    result_scalar = _reduce_to_per_channel(scalar, channel_dim, n_c)
    assert result_scalar.shape == (n_c,)
    torch.testing.assert_close(result_scalar.sum(), scalar)

    one_d = torch.randn(n_c, device=get_device())
    result_1d = _reduce_to_per_channel(one_d, 0, n_c)
    assert result_1d.shape == (n_c,)
    torch.testing.assert_close(result_1d.sum(), one_d.sum() / n_c)


def test_finite_difference_crps_zero_for_spatially_constant():
    """Data varying by channel but constant spatially should give zero loss.

    Catches bugs where finite differences are taken along the channel dim
    instead of spatial dims.
    """
    torch.manual_seed(0)
    gen = torch.randn(4, 2, 3, 1, 1).expand(4, 2, 3, 8, 16).clone()
    target = torch.randn(4, 1, 3, 1, 1).expand(4, 1, 3, 8, 16).clone()
    loss = FiniteDifferenceCRPSLoss(alpha=1.0, levels=1)
    result = loss(gen, target)
    torch.testing.assert_close(result, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_finite_difference_crps_positive_for_different_inputs():
    torch.manual_seed(0)
    gen = torch.randn(4, 2, 3, 8, 16)
    target = torch.randn(4, 1, 3, 8, 16)
    loss = FiniteDifferenceCRPSLoss(alpha=1.0, levels=1)
    result = loss(gen, target)
    assert result > 0


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_finite_difference_crps_multi_level(levels: int):
    torch.manual_seed(0)
    gen = torch.randn(4, 2, 3, 16, 32)
    target = torch.randn(4, 1, 3, 16, 32)
    loss = FiniteDifferenceCRPSLoss(alpha=1.0, levels=levels)
    result = loss(gen, target)
    assert result.shape == ()
    assert result > 0


def test_finite_difference_crps_rejects_zero_levels():
    with pytest.raises(ValueError, match="levels must be at least 1"):
        FiniteDifferenceCRPSLoss(alpha=1.0, levels=0)


def test_build_channel_mask():
    data_mask = {
        "a": torch.tensor([True, True, False]),
        "b": torch.tensor([True, False, True]),
    }
    names = ["a", "b"]
    mask = _build_channel_mask(
        data_mask,
        names,
        loss_ndim=4,
        channel_dim=1,
        device=torch.device("cpu"),
        batch_size=3,
    )
    assert mask.shape == (3, 2, 1, 1)
    expected = torch.tensor(
        [
            [[1.0], [1.0]],
            [[1.0], [0.0]],
            [[0.0], [1.0]],
        ]
    ).unsqueeze(-1)
    torch.testing.assert_close(mask, expected)


def test_build_channel_mask_missing_name_defaults_to_present():
    data_mask = {"a": torch.tensor([True, False])}
    names = ["a", "b"]
    mask = _build_channel_mask(
        data_mask,
        names,
        loss_ndim=4,
        channel_dim=1,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert mask[0, 0, 0, 0] == 1.0
    assert mask[1, 0, 0, 0] == 0.0
    assert mask[0, 1, 0, 0] == 1.0
    assert mask[1, 1, 0, 0] == 1.0


def test_loss_output_total_with_mask():
    loss = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    mask = torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]])
    output = LossOutput(loss=loss, channel_dim=1, channel_names=["a", "b"], mask=mask)
    expected = (1 + 2 + 3 + 4 + 5 + 6) / 6.0
    torch.testing.assert_close(output.total(), torch.tensor(expected))


def test_loss_output_total_without_mask_unchanged():
    loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output = LossOutput(loss=loss, channel_dim=1, channel_names=["a", "b"])
    torch.testing.assert_close(output.total(), loss.mean())


def test_loss_output_channel_losses_sum_to_total_with_mask():
    torch.manual_seed(42)
    loss = torch.randn(4, 3, 5, 5).abs()
    mask = torch.ones(4, 3, 1, 1)
    mask[2, 1, 0, 0] = 0.0
    mask[3, 0, 0, 0] = 0.0
    output = LossOutput(
        loss=loss * mask.expand_as(loss),
        channel_dim=1,
        channel_names=["a", "b", "c"],
        mask=mask,
    )
    channel_losses = output.get_channel_losses()
    torch.testing.assert_close(
        sum(channel_losses.values()), output.total(), atol=1e-5, rtol=1e-5
    )


def test_reduce_to_per_channel_with_mask():
    n_c = 3
    channel_dim = 1
    elementwise = torch.randn(4, n_c, 8, 8)
    mask = torch.ones(4, n_c, 1, 1)
    mask[0, 2, 0, 0] = 0.0
    expanded = mask.expand_as(elementwise)
    masked_loss = elementwise * expanded
    result = _reduce_to_per_channel(masked_loss, channel_dim, n_c, mask=mask)
    assert result.shape == (n_c,)
    expected_total = masked_loss.sum() / expanded.sum()
    torch.testing.assert_close(result.sum(), expected_total, atol=1e-5, rtol=1e-5)


def test_weighted_mapping_loss_with_data_mask():
    out_names = ["var_0", "var_1"]
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(0.0) for name in out_names},
        stds={name: torch.as_tensor(1.0) for name in out_names},
    )
    loss = torch.nn.MSELoss(reduction="none")
    mapping_loss = WeightedMappingLoss(
        loss,
        weights={},
        out_names=out_names,
        normalizer=normalizer,
    )
    x_val = torch.ones(4, 5, 5).to(get_device())
    y_val = torch.zeros(4, 5, 5).to(get_device())
    x_mapping = {"var_0": x_val, "var_1": x_val * 2}
    y_mapping = {"var_0": y_val, "var_1": y_val}
    result_no_mask = mapping_loss(x_mapping, y_mapping)
    expected_no_mask = (1.0 + 4.0) / 2.0
    torch.testing.assert_close(
        result_no_mask.total(), torch.tensor(expected_no_mask, device=get_device())
    )

    data_mask = {
        "var_0": torch.tensor([True, True, True, True]),
        "var_1": torch.tensor([True, True, False, False]),
    }
    result_masked = mapping_loss(x_mapping, y_mapping, data_mask=data_mask)
    n_unmasked = 4 * 25 + 2 * 25
    expected_masked = (4 * 25 * 1.0 + 2 * 25 * 4.0) / n_unmasked
    torch.testing.assert_close(
        result_masked.total(),
        torch.tensor(expected_masked, device=get_device()),
        atol=1e-5,
        rtol=1e-5,
    )


def test_step_loss_forwards_data_mask():
    out_names = ["var_0", "var_1"]
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(0.0) for name in out_names},
        stds={name: torch.as_tensor(1.0) for name in out_names},
    )
    area = torch.ones(1, 1)
    gridded_operations: GriddedOperations = LatLonOperations(area)
    config = StepLossConfig(type="MSE")
    step_loss = config.build(
        gridded_operations,
        out_names=out_names,
        channel_dim=-3,
        normalizer=normalizer,
    )
    x_mapping = {name: torch.ones(4, 5, 5).to(get_device()) for name in out_names}
    y_mapping = {name: torch.zeros(4, 5, 5).to(get_device()) for name in out_names}
    data_mask = {
        "var_0": torch.tensor([True, True, True, True]),
        "var_1": torch.tensor([False, False, False, False]),
    }
    result = step_loss(x_mapping, y_mapping, step=0, data_mask=data_mask)
    torch.testing.assert_close(result.total(), torch.tensor(1.0, device=get_device()))

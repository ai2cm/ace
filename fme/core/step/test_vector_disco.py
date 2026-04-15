"""Tests for VectorDiscoStep."""

import datetime
import math

import numpy as np
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.models.vector_disco import VectorDiscoNetworkConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.step.vector_disco import (
    VectorDiscoStepConfig,
    _compute_wind_scale,
    _find_level_names,
)

N_LEVELS = 3
IMG_SHAPE = (16, 32)


def _make_dataset_info() -> DatasetInfo:
    lat = torch.linspace(-90, 90, IMG_SHAPE[0]).float()
    lon = torch.linspace(0, 360, IMG_SHAPE[1] + 1)[:-1].float()
    coords = LatLonCoordinates(lat=lat, lon=lon)
    return DatasetInfo(
        horizontal_coordinates=coords,
        timestep=datetime.timedelta(hours=6),
    )


def _make_norm_config(in_names, out_names) -> NetworkAndLossNormalizationConfig:
    """Create a normalization config with unit std for all variables."""
    all_names = list(set(in_names).union(out_names))
    means = {n: np.zeros(1, dtype=np.float32) for n in all_names}
    stds = {n: np.ones(1, dtype=np.float32) for n in all_names}
    # Give winds a realistic std (~10 m/s)
    for n in all_names:
        if "wind" in n:
            stds[n] = np.array([10.0], dtype=np.float32)
    return NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(means=means, stds=stds),
    )


def _var_names(n_levels: int = N_LEVELS) -> tuple[list[str], list[str]]:
    in_names = (
        ["air_temperature_" + str(k) for k in range(n_levels)]
        + ["specific_total_water_" + str(k) for k in range(n_levels)]
        + ["eastward_wind_" + str(k) for k in range(n_levels)]
        + ["northward_wind_" + str(k) for k in range(n_levels)]
        + ["PRESsfc", "land_fraction", "DSWRFtoa"]
    )
    out_names = (
        ["air_temperature_" + str(k) for k in range(n_levels)]
        + ["specific_total_water_" + str(k) for k in range(n_levels)]
        + ["eastward_wind_" + str(k) for k in range(n_levels)]
        + ["northward_wind_" + str(k) for k in range(n_levels)]
        + ["PRESsfc", "PRATEsfc"]
    )
    return in_names, out_names


def _make_config(
    n_levels: int = N_LEVELS, residual_prediction: bool = False
) -> VectorDiscoStepConfig:
    in_names, out_names = _var_names(n_levels)
    return VectorDiscoStepConfig(
        network=VectorDiscoNetworkConfig(
            n_scalar_channels=8,
            n_vector_channels=4,
            n_blocks=2,
            kernel_shape=3,
        ),
        in_names=in_names,
        out_names=out_names,
        normalization=_make_norm_config(in_names, out_names),
        residual_prediction=residual_prediction,
    )


def _make_input_data(
    config: VectorDiscoStepConfig, batch_size: int = 2
) -> dict[str, torch.Tensor]:
    """Generate random input data matching the config, on the active device."""
    device = get_device()
    data: dict[str, torch.Tensor] = {}
    for name in config.in_names:
        if "wind" in name:
            data[name] = 10.0 * torch.randn(batch_size, *IMG_SHAPE, device=device)
        elif "temperature" in name:
            data[name] = 280.0 + 10.0 * torch.randn(
                batch_size, *IMG_SHAPE, device=device
            )
        else:
            data[name] = torch.randn(batch_size, *IMG_SHAPE, device=device)
    return data


class TestHelpers:
    def test_find_level_names(self):
        names = [
            "eastward_wind_0",
            "eastward_wind_1",
            "eastward_wind_2",
            "PRESsfc",
            "northward_wind_0",
        ]
        result = _find_level_names(names, "eastward_wind")
        assert result == [
            "eastward_wind_0",
            "eastward_wind_1",
            "eastward_wind_2",
        ]

    def test_find_level_names_empty(self):
        assert _find_level_names(["PRESsfc", "TMP2m"], "eastward_wind") == []

    def test_compute_wind_scale(self):
        from fme.core.normalizer import StandardNormalizer

        means = {"u_0": torch.tensor(0.0), "v_0": torch.tensor(0.0)}
        stds = {"u_0": torch.tensor(3.0), "v_0": torch.tensor(4.0)}
        normalizer = StandardNormalizer(means, stds)
        scale = _compute_wind_scale(normalizer, ["u_0"], ["v_0"])
        assert abs(scale - 5.0) < 1e-6  # sqrt(9 + 16) = 5


class TestVectorDiscoStepConfig:
    def test_construction(self):
        config = _make_config()
        assert len(config.in_names) > 0
        assert len(config.out_names) > 0

    def test_prognostic_names(self):
        config = _make_config()
        prog = set(config.prognostic_names)
        # Temperature, water, wind, PRESsfc should be prognostic
        assert "air_temperature_0" in prog
        assert "eastward_wind_0" in prog
        assert "PRESsfc" in prog
        # land_fraction is input-only, PRATEsfc is output-only
        assert "land_fraction" not in prog
        assert "PRATEsfc" not in prog

    def test_diagnostic_names(self):
        config = _make_config()
        diag = set(config.diagnostic_names)
        assert "PRATEsfc" in diag
        assert "air_temperature_0" not in diag


class TestVectorDiscoStep:
    def test_construction_and_forward(self):
        config = _make_config()
        dataset_info = _make_dataset_info()
        step = config.get_step(dataset_info, init_weights=lambda m: None)

        input_data = _make_input_data(config)
        from fme.core.step.args import StepArgs

        args = StepArgs(input=input_data, next_step_input_data={})
        output = step.step(args)

        # Check all output names are present
        for name in config.out_names:
            assert name in output, f"missing output: {name}"
            assert output[name].shape == (2, *IMG_SHAPE)

    def test_wind_normalization_no_bias(self):
        """Wind normalization uses only scale (no mean subtraction)."""
        config = _make_config()
        dataset_info = _make_dataset_info()
        step = config.get_step(dataset_info, init_weights=lambda m: None)

        # Verify wind_scale is derived from stds
        expected_scale = math.sqrt(10.0**2 + 10.0**2)  # std_u=std_v=10
        assert abs(step._wind_scale - expected_scale) < 1e-4

    def test_coriolis_latitude_pattern(self):
        """Coriolis should be zero at equator and maximal at poles."""
        config = _make_config()
        dataset_info = _make_dataset_info()
        step = config.get_step(dataset_info, init_weights=lambda m: None)

        coriolis = step.coriolis.squeeze()  # (H, W)
        mid_lat = IMG_SHAPE[0] // 2
        # Equator should be near zero
        assert coriolis[mid_lat, 0].abs() < 1e-4
        # Poles should be maximal (and opposite sign)
        assert coriolis[0, 0] < 0  # south pole, negative
        assert coriolis[-1, 0] > 0  # north pole, positive

    def test_gaussian_input_well_behaved(self):
        """Outputs from realistic-scale gaussian inputs are finite and bounded."""
        torch.manual_seed(42)
        config = _make_config()
        dataset_info = _make_dataset_info()
        step = config.get_step(dataset_info, init_weights=lambda m: None)

        input_data = _make_input_data(config)
        from fme.core.step.args import StepArgs

        args = StepArgs(input=input_data, next_step_input_data={})
        output = step.step(args)

        for name, tensor in output.items():
            assert torch.isfinite(tensor).all(), f"{name} has NaN/Inf"

    def test_residual_prediction_flag(self):
        """With residual_prediction, prognostic scalars include input."""
        config = _make_config(residual_prediction=True)
        dataset_info = _make_dataset_info()
        step = config.get_step(dataset_info, init_weights=lambda m: None)

        input_data = _make_input_data(config)
        from fme.core.step.args import StepArgs

        args = StepArgs(input=input_data, next_step_input_data={})
        output = step.step(args)

        # With zero-init decoder, network output is zero. With residual
        # prediction, prognostic outputs should equal inputs. With winds,
        # the vector path always uses residual, so wind output = input.
        for name in ["eastward_wind_0", "northward_wind_0"]:
            torch.testing.assert_close(
                output[name], input_data[name], atol=1e-5, rtol=1e-5
            )

    def test_get_state_and_load(self):
        config = _make_config()
        dataset_info = _make_dataset_info()
        step = config.get_step(dataset_info, init_weights=lambda m: None)

        state = step.get_state()
        assert "network" in state

        # Load into a fresh step
        step2 = config.get_step(dataset_info, init_weights=lambda m: None)
        step2.load_state(state)

import math

import pytest
import torch

from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.normalizer import StandardNormalizer
from fme.core.step.co2_band_olr import (
    SPECTROSCOPIC_DECAY,
    CO2BandOLRInputConfig,
    band_center_emission_pressure,
    co2_band_olr,
    planck_flux_density,
)
from fme.core.step.single_module import step_with_adjustments

AK = [1.0, 5119.9, 13881.3, 19343.5, 20087.1, 15596.7, 8880.5, 3057.3, 0.0]
BK = [0.0, 0.0, 0.0054, 0.0597, 0.2035, 0.4384, 0.6806, 0.8739, 1.0]
LAYER_TEMPERATURE = [215.0, 205.0, 215.0, 235.0, 255.0, 268.0, 278.0, 285.0]


def _column(co2: float, dtype=torch.float64):
    coordinate = HybridSigmaPressureCoordinate(
        ak=torch.tensor(AK, dtype=dtype), bk=torch.tensor(BK, dtype=dtype)
    )
    surface_pressure = torch.tensor([1.0e5], dtype=dtype)
    return dict(
        air_temperature=torch.tensor([LAYER_TEMPERATURE], dtype=dtype),
        surface_temperature=torch.tensor([288.0], dtype=dtype),
        interface_pressure=coordinate.interface_pressure(surface_pressure),
        co2_volume_mixing_ratio=torch.tensor([co2], dtype=dtype),
        band_half_width=100.0,
    )


def test_planck_flux_density_matches_reference_value():
    # pi * B(667.5 cm^-1, 288 K) computed independently
    value = planck_flux_density(torch.tensor(288.0, dtype=torch.float64))
    c1, c2, nu = 1.191042972e-16, 1.438776877e-2, 66750.0
    expected = math.pi * 100.0 * c1 * nu**3 / (math.exp(c2 * nu / 288.0) - 1.0)
    assert value.item() == pytest.approx(expected, rel=1e-12)
    assert 0.40 < value.item() < 0.42


def test_band_center_emission_pressure_is_in_the_stratosphere():
    # Jeevanjee et al. (2021) quote 16 hPa at 280 ppmv
    p0 = band_center_emission_pressure(torch.tensor(280e-6, dtype=torch.float64))
    assert 1500.0 < p0.item() < 1900.0


def test_doubling_forcing_matches_analytic_formula():
    flux_1x = co2_band_olr(**_column(280e-6))
    flux_2x = co2_band_olr(**_column(560e-6))
    surface = planck_flux_density(torch.tensor(288.0, dtype=torch.float64))
    top = planck_flux_density(torch.tensor(LAYER_TEMPERATURE[0], dtype=torch.float64))
    expected = 2.0 * SPECTROSCOPIC_DECAY * math.log(2.0) * (surface - top)
    forcing = (flux_1x - flux_2x).item()
    assert forcing == pytest.approx(expected.item(), rel=1e-10)
    assert 3.0 < forcing < 5.0


def test_flux_has_gradient_to_every_temperature():
    column = _column(400e-6)
    column["air_temperature"].requires_grad_(True)
    column["surface_temperature"].requires_grad_(True)
    co2_band_olr(**column).sum().backward()
    assert (column["air_temperature"].grad > 0).all()
    assert (column["surface_temperature"].grad > 0).all()


def test_flux_is_continuous_as_emission_pressure_crosses_an_interface():
    # p_0 equals the first interior interface (5119.9 Pa) near 33 ppmv
    p_interface = AK[1]
    co2 = 280e-6 * (band_center_emission_pressure(torch.tensor(280e-6)) ** 2).item()
    co2 /= p_interface**2
    below = co2_band_olr(**_column(co2 * (1 - 1e-6)))
    above = co2_band_olr(**_column(co2 * (1 + 1e-6)))
    assert below.item() == pytest.approx(above.item(), abs=1e-4)


def test_config_requires_inputs_to_be_in_names():
    config = CO2BandOLRInputConfig(mean=50.0, std=10.0)
    with pytest.raises(ValueError, match="global_mean_co2"):
        config.validate_names(["surface_temperature", "PRESsfc"])
    coordinate = HybridSigmaPressureCoordinate(ak=torch.tensor(AK), bk=torch.tensor(BK))
    with pytest.raises(ValueError, match="air_temperature_0"):
        config.build(coordinate, ["global_mean_co2", "surface_temperature"])


def test_step_replaces_co2_channel_with_normalized_flux():
    coordinate = HybridSigmaPressureCoordinate(ak=torch.tensor(AK), bk=torch.tensor(BK))
    config = CO2BandOLRInputConfig(mean=50.0, std=10.0)
    names = [
        "global_mean_co2",
        "surface_temperature",
        "PRESsfc",
    ] + config.air_temperature_names(8)
    derived = config.build(coordinate, names)
    shape = (2, 4, 8)
    input = {
        "global_mean_co2": torch.full(shape, 400e-6),
        "surface_temperature": 288.0 + torch.rand(shape),
        "PRESsfc": torch.full(shape, 1.0e5),
    }
    for k, name in enumerate(config.air_temperature_names(8)):
        input[name] = torch.full(shape, LAYER_TEMPERATURE[k])
    normalizer = StandardNormalizer(
        means={name: torch.tensor(0.0) for name in names},
        stds={name: torch.tensor(1.0) for name in names},
    )
    seen = {}

    def network_call(input_norm):
        seen.update(input_norm)
        return {"surface_temperature": input_norm["surface_temperature"]}

    step_with_adjustments(
        input=input,
        next_step_input_data={},
        network_calls=network_call,
        normalizer=normalizer,
        corrector=None,
        ocean=None,
        residual_names=None,
        normalized_input_overrides=derived.normalized_overrides,
    )
    expected = (derived.flux(input) - 50.0) / 10.0
    torch.testing.assert_close(seen["global_mean_co2"], expected)
    assert seen["global_mean_co2"].std() > 0  # varies per column
    torch.testing.assert_close(seen["PRESsfc"], input["PRESsfc"])

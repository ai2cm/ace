import math

import pytest
import torch

from fme.core.co2_temperature_offset import CO2TemperatureProfileConfig


def test_per_doubling_at_surface_equals_surface_warming():
    cfg = CO2TemperatureProfileConfig(
        delta_t_surface_per_doubling=3.0,
        delta_t_stratosphere_per_doubling=-10.0,
        tropopause_pressure_pa=2.0e4,
        stratosphere_top_pressure_pa=1.0e2,
    )
    assert cfg.per_doubling_at(1.0e5) == pytest.approx(3.0)
    assert cfg.per_doubling_at(8.5e4) == pytest.approx(3.0)


def test_per_doubling_at_tropopause_is_continuous():
    cfg = CO2TemperatureProfileConfig(
        delta_t_surface_per_doubling=3.0,
        delta_t_stratosphere_per_doubling=-10.0,
        tropopause_pressure_pa=2.0e4,
        stratosphere_top_pressure_pa=1.0e2,
    )
    assert cfg.per_doubling_at(cfg.tropopause_pressure_pa) == pytest.approx(3.0)


def test_per_doubling_saturates_at_strat_top():
    cfg = CO2TemperatureProfileConfig(
        delta_t_surface_per_doubling=3.0,
        delta_t_stratosphere_per_doubling=-10.0,
        tropopause_pressure_pa=2.0e4,
        stratosphere_top_pressure_pa=1.0e2,
    )
    assert cfg.per_doubling_at(1.0e2) == pytest.approx(-10.0)
    assert cfg.per_doubling_at(10.0) == pytest.approx(-10.0)  # clipped


def test_per_doubling_log_linear_in_stratosphere():
    # In the stratosphere the per-doubling response is linear in log(p)
    # between ΔT_surf at p_t and ΔT_strat at p_top.
    cfg = CO2TemperatureProfileConfig(
        delta_t_surface_per_doubling=3.0,
        delta_t_stratosphere_per_doubling=-10.0,
        tropopause_pressure_pa=2.0e4,
        stratosphere_top_pressure_pa=1.0e2,
    )
    # log-midpoint of the stratospheric range
    p_mid = math.exp((math.log(2.0e4) + math.log(1.0e2)) / 2)
    assert cfg.per_doubling_at(p_mid) == pytest.approx((3.0 + -10.0) / 2)


def test_per_doubling_layer_0_value_is_reasonable():
    # Layer 0 (~25 hPa) should land somewhere around -2 K per doubling with
    # the default parameters -- documents the calibration choice.
    cfg = CO2TemperatureProfileConfig()  # defaults
    p_layer_0 = 2560.5
    assert cfg.per_doubling_at(p_layer_0) == pytest.approx(-2.04, abs=0.05)


def test_delta_t_scales_logarithmically_with_co2():
    cfg = CO2TemperatureProfileConfig(co2_reference_vmr=280e-6)
    co2 = torch.tensor([280e-6, 560e-6, 1120e-6])  # 1x, 2x, 4x reference
    dt = cfg.delta_t(co2, pressure_pa=1.0e5)
    # +0, +3 (one doubling), +6 (two doublings)
    torch.testing.assert_close(dt, torch.tensor([0.0, 3.0, 6.0]))


def test_delta_t_zero_at_reference():
    cfg = CO2TemperatureProfileConfig(co2_reference_vmr=280e-6)
    co2 = torch.tensor([280e-6])
    for p in (1.0e5, 5.0e4, 2.0e4, 2.5e3, 1.0e2):
        torch.testing.assert_close(cfg.delta_t(co2, pressure_pa=p), torch.tensor([0.0]))


def test_delta_t_negative_in_stratosphere_for_increased_co2():
    cfg = CO2TemperatureProfileConfig(co2_reference_vmr=280e-6)
    co2 = torch.tensor([560e-6])  # 2x
    # Layer 0 (~25 hPa): negative
    assert cfg.delta_t(co2, pressure_pa=2560.5).item() < 0
    # Surface: positive
    assert cfg.delta_t(co2, pressure_pa=1.0e5).item() > 0


def test_invalid_reference_vmr_raises():
    with pytest.raises(ValueError, match="co2_reference_vmr"):
        CO2TemperatureProfileConfig(co2_reference_vmr=-1.0)


def test_invalid_pressure_ordering_raises():
    with pytest.raises(ValueError, match="tropopause_pressure_pa"):
        CO2TemperatureProfileConfig(
            tropopause_pressure_pa=1.0e2,
            stratosphere_top_pressure_pa=2.0e4,
        )

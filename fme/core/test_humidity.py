import torch

from fme.core.constants import RDGAS, RVGAS
from fme.core.humidity import (
    bolton_saturation_vapor_pressure,
    saturation_specific_humidity,
)


def test_bolton_at_freezing_point():
    # Bolton (1980) yields exactly 6.112 hPa at T_C = 0.
    result = bolton_saturation_vapor_pressure(torch.tensor(273.15))
    torch.testing.assert_close(result, torch.tensor(6.112))


def test_bolton_monotonic_in_temperature():
    temps = torch.linspace(250.0, 320.0, 10)
    es = bolton_saturation_vapor_pressure(temps)
    diffs = es[1:] - es[:-1]
    assert (diffs > 0).all()


def test_bolton_reference_value_at_300k():
    # Bolton (1980): T_C = 26.85; e_s ≈ 6.112 * exp(17.67 * 26.85 / 270.35)
    # ≈ 35.30 hPa
    result = bolton_saturation_vapor_pressure(torch.tensor(300.0))
    torch.testing.assert_close(result, torch.tensor(35.345211), rtol=1e-5, atol=1e-4)


def test_bolton_preserves_shape():
    t = torch.full((2, 3, 4), 285.0)
    es = bolton_saturation_vapor_pressure(t)
    assert es.shape == t.shape


def test_saturation_specific_humidity_worked_example():
    # At T = 300 K, e_s ≈ 35.345 hPa = 3534.5 Pa. At p = 1000 hPa = 1e5 Pa,
    # q_sat = eps * e_s / (p - (1 - eps) * e_s).
    t = torch.tensor(300.0)
    p = torch.tensor(1.0e5)
    eps = RDGAS / RVGAS
    e_s = bolton_saturation_vapor_pressure(t) * 100.0
    expected = eps * e_s / (p - (1.0 - eps) * e_s)
    result = saturation_specific_humidity(t, p)
    torch.testing.assert_close(result, expected)
    # sanity: a warm near-surface value is a small positive mixing ratio
    assert 0.01 < result.item() < 0.04


def test_saturation_specific_humidity_increases_with_temperature():
    p = torch.full((5,), 1.0e5)
    t = torch.linspace(250.0, 305.0, 5)
    qsat = saturation_specific_humidity(t, p)
    assert (qsat[1:] - qsat[:-1] > 0).all()


def test_saturation_specific_humidity_decreases_with_pressure():
    t = torch.full((5,), 290.0)
    p = torch.linspace(3.0e4, 1.0e5, 5)
    qsat = saturation_specific_humidity(t, p)
    # at fixed temperature, higher pressure means lower saturation mixing ratio
    assert (qsat[1:] - qsat[:-1] < 0).all()


def test_saturation_specific_humidity_broadcasts():
    t = torch.full((2, 3, 4), 285.0)
    p = torch.full((2, 3, 4), 9.0e4)
    qsat = saturation_specific_humidity(t, p)
    assert qsat.shape == (2, 3, 4)

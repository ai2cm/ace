import torch

from fme.core.humidity import bolton_saturation_vapor_pressure


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

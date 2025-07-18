"""In this test module, we test the torch spherical harmonics module."""

import pytest
import torch
import torch_harmonics as harmonics

from fme.core.device import get_device


@pytest.mark.parametrize("nlat, nlon", [(6, 12)])
@pytest.mark.parametrize("grid", ["equiangular", "legendre-gauss"])
@pytest.mark.parametrize("constant", [1.0, 0.42])
def test_constant_field(nlat, nlon, grid, constant):
    """Tests that the SHT of a constant field has a single non-zero wavenumber,
    the first one.
    """
    constant_field = torch.tensor(constant, device=get_device()).repeat(nlat, nlon)

    sht = harmonics.RealSHT(nlat, nlon, grid=grid).to(get_device())

    coeffs = sht(constant_field).ravel()
    zero = torch.zeros(1, dtype=torch.complex64, device=get_device())

    assert not torch.isclose(coeffs[0], zero)
    assert torch.all(torch.isclose(zero, coeffs[1:], atol=1e-6))


def _roundtrip(field, grid):
    nlat, nlon = field.shape
    sht = harmonics.RealSHT(nlat, nlon, grid=grid).to(get_device())
    isht = harmonics.InverseRealSHT(nlat, nlon, grid=grid).to(get_device())
    return isht(sht(field))


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("nlat, nlon", [(6, 12)])
def test_roundtrip(nlat, nlon, seed, grid="legendre-gauss"):
    """Tests that the SHT and ISHT are inverses of each other."""
    torch.manual_seed(seed)
    random_field = torch.randn(nlat, nlon, device=get_device())
    proj = _roundtrip(random_field, grid)
    assert torch.all(torch.isclose(proj, _roundtrip(proj, grid), atol=1e-6))

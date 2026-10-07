import pytest
import torch

from fme.core.ocean_eos import (
    G_EARTH,
    RHO_0,
    _density_anomaly,
    boussinesq_pressure,
    interface_to_center_depth,
    wright97_anomaly,
)

# MOM_EOS.F90::EOS_unit_tests check values, T=25 degC, p=1e7 Pa; the S=15 value
# is the (commented-out) WRIGHT_REDUCED test outside the fit range.
T_CHK, P_CHK = 25.0, 1.0e7
CHECK_VALUES = [(35.0, 1027.54303596346), (15.0, 1012.625699301455)]
RTOL = {
    torch.float64: 1000 * torch.finfo(torch.float64).eps,
    torch.float32: 32 * torch.finfo(torch.float32).eps,
}


def _grid(dtype, requires_grad=False):
    S, T, p = torch.meshgrid(
        torch.linspace(28.0, 38.0, 11, dtype=dtype),
        torch.linspace(-2.0, 30.0, 17, dtype=dtype),
        torch.linspace(0.0, 5.0e7, 11, dtype=dtype),
        indexing="ij",
    )
    if requires_grad:
        for x in (S, T, p):
            x.requires_grad_(True)
    return S, T, p


@pytest.mark.parametrize("rho_ref", [RHO_0, 1000.0])
@pytest.mark.parametrize("S, rho_chk", CHECK_VALUES)
def test_mom6_check_values(S, rho_chk, rho_ref):
    f64 = torch.float64
    a = wright97_anomaly(
        torch.tensor(S, dtype=f64),
        torch.tensor(T_CHK, dtype=f64),
        torch.tensor(P_CHK, dtype=f64),
        rho_ref,
    )
    assert a.dtype == f64
    assert abs(float(a) + rho_ref - rho_chk) / rho_chk < RTOL[f64]


def test_float32_vs_float64():
    S, T, p = _grid(torch.float64)
    a64 = wright97_anomaly(S, T, p)
    a32 = wright97_anomaly(S.float(), T.float(), p.float())
    assert a32.dtype == torch.float32
    assert ((a32.double() - a64).abs() <= RTOL[torch.float32] * (a64 + RHO_0)).all()


def test_monotonic_in_fit_range():
    S, T, p = _grid(torch.float64, requires_grad=True)
    dS, dT, dp = torch.autograd.grad(wright97_anomaly(S, T, p).sum(), (S, T, p))
    assert (dS > 0).all()
    assert (dT < 0).all()
    assert (dp > 0).all()


def test_pressure_helpers():
    idepth = torch.tensor([0.0, 10.0, 30.0], dtype=torch.float64)
    z = interface_to_center_depth(idepth)
    torch.testing.assert_close(z, torch.tensor([5.0, 20.0], dtype=torch.float64))
    torch.testing.assert_close(boussinesq_pressure(z), RHO_0 * G_EARTH * z)


def test_density_anomaly_values_and_mask():
    idepth = torch.tensor([0.0, 10.0, 1000.0], dtype=torch.float64)
    mask = torch.ones(3, 4, 2)
    mask[0] = 0.0  # land
    mask[1, :, 1] = 0.0  # below the sea floor at level 1
    g = torch.Generator().manual_seed(0)
    thetao = 2.0 + 20.0 * torch.rand(2, 3, 4, 2, generator=g, dtype=torch.float64)
    so = 34.0 + 2.0 * torch.rand(2, 3, 4, 2, generator=g, dtype=torch.float64)
    thetao[0, 2, 3, 0] = torch.nan
    rho = _density_anomaly(thetao, so, idepth, mask)
    assert rho.shape == thetao.shape
    for k in range(2):
        p = boussinesq_pressure(interface_to_center_depth(idepth))[k]
        expected = wright97_anomaly(so[..., k], thetao[..., k], p)
        valid = (mask[..., k] > 0) & expected.isfinite()
        torch.testing.assert_close(rho[..., k][valid], expected[valid])
        assert rho[..., k][~valid].isnan().all()
    assert rho[:, 0].isnan().all()
    assert rho[:, 1, :, 1].isnan().all()
    assert rho[:, 1, :, 0].isfinite().all()
    assert rho[0, 2, 3, 0].isnan()

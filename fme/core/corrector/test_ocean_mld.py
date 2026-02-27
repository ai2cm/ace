"""Unit tests for ocean_mld helper functions."""

import pytest
import torch

from fme import get_device
from fme.core.constants import DENSITY_OF_WATER_CM4, SPECIFIC_HEAT_OF_WATER_CM4
from fme.core.corrector.ocean_mld import (
    apply_geothermal_bottom_correction,
    compute_mld,
    compute_mld_active_thickness,
    compute_mld_soft,
    jmd95_potential_density,
)
from fme.core.ocean_data import OceanData

DEVICE = get_device()


def test_jmd95_potential_density_pure_water():
    S = torch.tensor([0.0], device=DEVICE)
    theta = torch.tensor([0.0], device=DEVICE)
    rho = jmd95_potential_density(S, theta)
    torch.testing.assert_close(rho, torch.tensor([999.842594], device=DEVICE))


def test_jmd95_potential_density_seawater():
    S = torch.tensor([35.0], device=DEVICE)
    theta = torch.tensor([20.0], device=DEVICE)
    rho = jmd95_potential_density(S, theta)
    assert 1024.0 < rho.item() < 1026.0


def test_jmd95_potential_density_cold_seawater_denser():
    S = torch.tensor([35.0, 35.0], device=DEVICE)
    theta = torch.tensor([20.0, 5.0], device=DEVICE)
    rho = jmd95_potential_density(S, theta)
    assert rho[1] > rho[0]


def test_compute_mld_sharp_pycnocline():
    nsamples, ny, nx, nz = 1, 2, 2, 4
    idepth = torch.tensor([0.0, 10.0, 50.0, 200.0, 1000.0], device=DEVICE)
    mask = torch.ones(nsamples, ny, nx, nz, device=DEVICE)
    deptho = torch.full((nsamples, ny, nx), 1000.0, device=DEVICE)

    S = torch.full((nsamples, ny, nx, nz), 35.0, device=DEVICE)
    theta = torch.zeros(nsamples, ny, nx, nz, device=DEVICE)
    theta[..., 0] = 20.0
    theta[..., 1] = 20.0
    theta[..., 2] = 5.0
    theta[..., 3] = 5.0
    density = jmd95_potential_density(S, theta)

    mld = compute_mld(density, idepth, deptho, mask)
    assert mld.shape == (nsamples, ny, nx)
    # MLD should be between layer-1 center (30 m) and layer-2 center (125 m)
    assert (mld > 30).all() and (mld < 50).all()


def test_compute_mld_wellmixed_falls_back_to_deptho():
    nsamples, ny, nx, nz = 1, 2, 2, 4
    idepth = torch.tensor([0.0, 10.0, 50.0, 200.0, 1000.0], device=DEVICE)
    mask = torch.ones(nsamples, ny, nx, nz, device=DEVICE)
    deptho = torch.full((nsamples, ny, nx), 500.0, device=DEVICE)

    S = torch.full((nsamples, ny, nx, nz), 35.0, device=DEVICE)
    theta = torch.full((nsamples, ny, nx, nz), 20.0, device=DEVICE)
    density = jmd95_potential_density(S, theta)

    mld = compute_mld(density, idepth, deptho, mask)
    torch.testing.assert_close(mld, deptho)


def test_compute_mld_active_thickness():
    idepth = torch.tensor([0.0, 10.0, 50.0, 200.0, 1000.0], device=DEVICE)
    nsamples, ny, nx, nz = 1, 1, 1, 4
    mask = torch.ones(nsamples, ny, nx, nz, device=DEVICE)
    mld_2d = torch.full((nsamples, ny, nx), 30.0, device=DEVICE)

    h = compute_mld_active_thickness(mld_2d, idepth, mask)
    assert h.shape == (nsamples, ny, nx, nz)
    # Layer 0 (0-10): min(10, 30) - 0 = 10
    torch.testing.assert_close(h[..., 0], torch.tensor([[[10.0]]], device=DEVICE))
    # Layer 1 (10-50): min(50, 30) - 10 = 20
    torch.testing.assert_close(h[..., 1], torch.tensor([[[20.0]]], device=DEVICE))
    # Layer 2 (50-200): min(200, 30) - 50 = 0
    torch.testing.assert_close(h[..., 2], torch.tensor([[[0.0]]], device=DEVICE))
    # Layer 3 (200-1000): 0
    torch.testing.assert_close(h[..., 3], torch.tensor([[[0.0]]], device=DEVICE))
    # Total active thickness = 30 (== MLD)
    torch.testing.assert_close(
        h.sum(dim=-1),
        torch.tensor([[[30.0]]], device=DEVICE),
    )


def test_compute_mld_active_thickness_masked():
    idepth = torch.tensor([0.0, 10.0, 50.0], device=DEVICE)
    nsamples, ny, nx, nz = 1, 2, 2, 2
    mask = torch.ones(nsamples, ny, nx, nz, device=DEVICE)
    mask[:, 0, 0, :] = 0.0
    mld_2d = torch.full((nsamples, ny, nx), 30.0, device=DEVICE)

    h = compute_mld_active_thickness(mld_2d, idepth, mask)
    assert (h[:, 0, 0, :] == 0.0).all()
    assert (h[:, 1, 1, :] > 0.0).all()


def test_apply_geothermal_bottom_correction_modifies_only_bottom():
    nsamples, ny, nx, nz = 1, 2, 2, 3
    idepth = torch.tensor([0.0, 10.0, 50.0, 200.0], device=DEVICE)
    mask = torch.ones(nsamples, ny, nx, nz, device=DEVICE)
    mask[:, 0, 0, 2] = 0.0  # column (0,0) has only 2 wet levels

    from fme.core.coordinates import DepthCoordinate

    depth_coord = DepthCoordinate(idepth, mask)

    gen_data = {
        f"thetao_{k}": torch.ones(nsamples, ny, nx, device=DEVICE) * 10.0
        for k in range(nz)
    }
    gen_data.update(
        {
            f"so_{k}": torch.ones(nsamples, ny, nx, device=DEVICE) * 35.0
            for k in range(nz)
        }
    )
    gen_before = {k: v.clone() for k, v in gen_data.items()}

    hfgeou = torch.ones(nsamples, ny, nx, device=DEVICE) * 0.1
    ssf = torch.ones(nsamples, ny, nx, device=DEVICE)
    forcing_data = {"hfgeou": hfgeou, "sea_surface_fraction": ssf}

    gen = OceanData(gen_data, depth_coord)
    forcing = OceanData(forcing_data)
    dt = 5 * 86400.0
    apply_geothermal_bottom_correction(gen, forcing, depth_coord, dt)

    dz = idepth.diff(dim=-1)
    expected_dT = (
        hfgeou * ssf * dt / (DENSITY_OF_WATER_CM4 * SPECIFIC_HEAT_OF_WATER_CM4)
    )

    # Column (0,0): bottom wet is level 1 (mask[...,2]=0), dz=40
    torch.testing.assert_close(
        gen.data["thetao_0"][:, 0, 0],
        gen_before["thetao_0"][:, 0, 0],
    )
    torch.testing.assert_close(
        gen.data["thetao_1"][:, 0, 0],
        gen_before["thetao_1"][:, 0, 0] + expected_dT[:, 0, 0] / dz[1],
    )

    # Column (1,1): bottom wet is level 2, dz=150
    torch.testing.assert_close(
        gen.data["thetao_0"][:, 1, 1],
        gen_before["thetao_0"][:, 1, 1],
    )
    torch.testing.assert_close(
        gen.data["thetao_1"][:, 1, 1],
        gen_before["thetao_1"][:, 1, 1],
    )
    torch.testing.assert_close(
        gen.data["thetao_2"][:, 1, 1],
        gen_before["thetao_2"][:, 1, 1] + expected_dT[:, 1, 1] / dz[2],
    )


# --- compute_mld_soft tests ---


def _make_pycnocline_density():
    """Sharp pycnocline between layers 1 and 2, reusable for hard and soft tests."""
    nsamples, ny, nx, nz = 1, 2, 2, 4
    idepth = torch.tensor([0.0, 10.0, 50.0, 200.0, 1000.0], device=DEVICE)
    mask = torch.ones(nsamples, ny, nx, nz, device=DEVICE)
    deptho = torch.full((nsamples, ny, nx), 1000.0, device=DEVICE)

    S = torch.full((nsamples, ny, nx, nz), 35.0, device=DEVICE)
    theta = torch.zeros(nsamples, ny, nx, nz, device=DEVICE)
    theta[..., 0] = 20.0
    theta[..., 1] = 20.0
    theta[..., 2] = 5.0
    theta[..., 3] = 5.0
    density = jmd95_potential_density(S, theta)
    return density, idepth, deptho, mask


def test_compute_mld_soft_sharp_pycnocline():
    density, idepth, deptho, mask = _make_pycnocline_density()
    tau = 1e-4
    mld_soft = compute_mld_soft(density, idepth, deptho, mask, tau=tau)
    mld_hard = compute_mld(density, idepth, deptho, mask)
    assert mld_soft.shape == mld_hard.shape
    torch.testing.assert_close(mld_soft, mld_hard, atol=1e-4, rtol=0.0)


def test_compute_mld_soft_wellmixed_falls_back_to_deptho():
    nsamples, ny, nx, nz = 1, 2, 2, 4
    idepth = torch.tensor([0.0, 10.0, 50.0, 200.0, 1000.0], device=DEVICE)
    mask = torch.ones(nsamples, ny, nx, nz, device=DEVICE)
    deptho = torch.full((nsamples, ny, nx), 500.0, device=DEVICE)

    S = torch.full((nsamples, ny, nx, nz), 35.0, device=DEVICE)
    theta = torch.full((nsamples, ny, nx, nz), 20.0, device=DEVICE)
    density = jmd95_potential_density(S, theta)

    tau = 1e-4
    mld = compute_mld_soft(density, idepth, deptho, mask, tau=tau)
    torch.testing.assert_close(mld, deptho, atol=1e-4, rtol=0.0)


def test_compute_mld_soft_masked_layers_ignored():
    nsamples, ny, nx, nz = 1, 2, 2, 4
    idepth = torch.tensor([0.0, 10.0, 50.0, 200.0, 1000.0], device=DEVICE)
    deptho = torch.full((nsamples, ny, nx), 1000.0, device=DEVICE)

    S = torch.full((nsamples, ny, nx, nz), 35.0, device=DEVICE)
    theta = torch.zeros(nsamples, ny, nx, nz, device=DEVICE)
    theta[..., 0] = 20.0
    theta[..., 1] = 20.0
    theta[..., 2] = 5.0
    theta[..., 3] = 5.0
    density = jmd95_potential_density(S, theta)

    mask_full = torch.ones(nsamples, ny, nx, nz, device=DEVICE)
    mask_partial = mask_full.clone()
    mask_partial[..., 2:] = 0.0  # mask out the dense layers

    tau = 1e-4
    mld_full = compute_mld_soft(density, idepth, deptho, mask_full, tau=tau)
    mld_partial = compute_mld_soft(density, idepth, deptho, mask_partial, tau=tau)

    assert (mld_full < 200).all()
    torch.testing.assert_close(mld_partial, deptho, atol=1e-4, rtol=0.0)


def test_compute_mld_soft_differentiable_across_layers():
    """Gradient flows through the soft MLD when the crossing is between layers."""
    nsamples, ny, nx, nz = 1, 1, 1, 4
    idepth = torch.tensor([0.0, 10.0, 50.0, 200.0, 1000.0], device=DEVICE)
    mask = torch.ones(nsamples, ny, nx, nz, device=DEVICE)
    deptho = torch.full((nsamples, ny, nx), 1000.0, device=DEVICE)

    S = torch.full((nsamples, ny, nx, nz), 35.0, device=DEVICE)
    theta = torch.zeros(nsamples, ny, nx, nz, device=DEVICE, requires_grad=True)
    with torch.no_grad():
        theta[..., 0] = 20.0
        theta[..., 1] = 20.0
        theta[..., 2] = 5.0
        theta[..., 3] = 5.0
    density = jmd95_potential_density(S, theta)

    tau = 0.005
    mld = compute_mld_soft(density, idepth, deptho, mask, tau=tau)
    mld.sum().backward()
    assert theta.grad is not None
    assert theta.grad.abs().sum() > 0


class TestOceanCorrectorConfigValidation:
    def test_soft_method_requires_threshold(self):
        from fme.core.corrector.ocean import (
            OceanCorrectorConfig,
            OceanHeatContentBudgetConfig,
        )

        with pytest.raises(ValueError, match="mld_soft_threshold must be set"):
            OceanCorrectorConfig(
                ocean_heat_content_correction=OceanHeatContentBudgetConfig(
                    method="mixed_layer_depth_soft"
                ),
            )

    def test_threshold_without_soft_method_raises(self):
        from fme.core.corrector.ocean import (
            OceanCorrectorConfig,
            OceanHeatContentBudgetConfig,
        )

        with pytest.raises(ValueError, match="no budget correction uses"):
            OceanCorrectorConfig(
                ocean_heat_content_correction=OceanHeatContentBudgetConfig(
                    method="mixed_layer_depth"
                ),
                mld_soft_threshold=0.005,
            )

    def test_soft_method_with_threshold_ok(self):
        from fme.core.corrector.ocean import (
            OceanCorrectorConfig,
            OceanHeatContentBudgetConfig,
        )

        config = OceanCorrectorConfig(
            ocean_heat_content_correction=OceanHeatContentBudgetConfig(
                method="mixed_layer_depth_soft"
            ),
            mld_soft_threshold=0.005,
        )
        assert config.mld_soft_threshold == 0.005

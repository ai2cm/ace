import math

import torch

from fme.core.shallow_water import ShallowWaterStepper

SHAPE = (32, 64)


def _make_stepper(**kw):
    return ShallowWaterStepper(shape=SHAPE, **kw)


def _gaussian_bump(nlat, nlon, lat0_deg=30.0, lon0_deg=180.0, sigma_deg=10.0):
    """Create a Gaussian height perturbation centered at (lat0, lon0)."""
    from fme.core.disco._quadrature import precompute_latitudes, precompute_longitudes

    colats = precompute_latitudes(nlat)[0].float()
    lons = precompute_longitudes(nlon).float()
    lat = (math.pi / 2.0 - colats).unsqueeze(1)
    lon = lons.unsqueeze(0)

    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    sigma = math.radians(sigma_deg)

    dlat = lat - lat0
    dlon = lon - lon0
    dist2 = dlat**2 + (dlon * torch.cos(lat)) ** 2
    bump = torch.exp(-dist2 / (2 * sigma**2))
    return bump.unsqueeze(0).unsqueeze(0)  # (1, 1, nlat, nlon)


DT = 0.01
H_AMPLITUDE = 0.01


class TestShallowWaterStepper:
    def test_resting_state_stays_at_rest(self):
        """Zero initial conditions produce zero tendencies."""
        stepper = _make_stepper()
        h = torch.zeros(1, 1, *SHAPE)
        uv = torch.zeros(1, 1, *SHAPE, 2)

        dh, duv = stepper.compute_tendencies(h, uv)
        assert dh.abs().max() < 1e-10
        assert duv.abs().max() < 1e-10

    def test_mass_conservation(self):
        """Total mass is conserved over multiple time steps."""
        stepper = _make_stepper()
        h = H_AMPLITUDE * _gaussian_bump(*SHAPE, sigma_deg=15.0)
        uv = torch.zeros(1, 1, *SHAPE, 2)

        mass_initial = stepper.total_mass(h).item()
        for _ in range(200):
            h, uv = stepper.step(h, uv, DT)
        mass_final = stepper.total_mass(h).item()

        assert abs(mass_final - mass_initial) / (abs(mass_initial) + 1e-30) < 0.01

    def test_energy_approximately_conserved(self):
        """Total energy doesn't grow over many time steps."""
        stepper = _make_stepper()
        h = H_AMPLITUDE * _gaussian_bump(*SHAPE, sigma_deg=15.0)
        uv = torch.zeros(1, 1, *SHAPE, 2)

        energy_initial = stepper.total_energy(h, uv).item()
        assert energy_initial > 0

        for _ in range(500):
            h, uv = stepper.step(h, uv, DT)

        energy_final = stepper.total_energy(h, uv).item()
        assert energy_final / energy_initial < 1.05
        assert energy_final / energy_initial > 0.5

    def test_perturbation_generates_motion(self):
        """A height perturbation should generate nonzero velocities."""
        stepper = _make_stepper()
        h = H_AMPLITUDE * _gaussian_bump(*SHAPE, sigma_deg=15.0)
        uv = torch.zeros(1, 1, *SHAPE, 2)

        for _ in range(50):
            h, uv = stepper.step(h, uv, DT)

        assert uv.abs().max() > 1e-6

    def test_output_shapes(self):
        """Step preserves tensor shapes."""
        stepper = _make_stepper()
        h = torch.zeros(2, 1, *SHAPE)
        uv = torch.zeros(2, 1, *SHAPE, 2)

        h_new, uv_new = stepper.step(h, uv, DT)
        assert h_new.shape == h.shape
        assert uv_new.shape == uv.shape

    def test_no_nans_after_many_steps(self):
        """Integration remains finite (no numerical blowup)."""
        stepper = _make_stepper()
        h = H_AMPLITUDE * _gaussian_bump(*SHAPE, sigma_deg=15.0)
        uv = torch.zeros(1, 1, *SHAPE, 2)

        for _ in range(1000):
            h, uv = stepper.step(h, uv, DT)

        assert torch.isfinite(h).all()
        assert torch.isfinite(uv).all()

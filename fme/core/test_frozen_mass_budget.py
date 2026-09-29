import numpy as np
import pytest
import torch

from fme.core.constants import (
    FREEZING_TEMPERATURE_KELVIN,
    LATENT_HEAT_OF_FREEZING,
    LATENT_HEAT_OF_VAPORIZATION,
    SPECIFIC_HEAT_OF_SEA_WATER_CM4,
)
from fme.core.corrector.ocean import _correct_hfds
from fme.core.frozen_mass_budget import (
    calving_residue,
    correct_hfds,
    frozen_mass,
    frozen_mass_energy_budget_residual,
    frozen_mass_flux_sum,
    ocean_net_surface_energy_flux,
    target_frozen_mass_energy_budget_residual,
)

ATMOS_NAMES = [
    "DSWRFsfc",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "PRATEsfc",
    "total_frozen_precipitation_rate",
]


def _window(n_time=4, n_lat=3, n_lon=5, seed=0) -> dict[str, torch.Tensor]:
    """Synthetic window with land, NaN on land for ocean/ice fields, time on dim 0."""
    g = torch.Generator().manual_seed(seed)

    def r(scale=1.0, offset=0.0):
        return offset + scale * torch.rand(n_time, n_lat, n_lon, generator=g)

    land = torch.zeros(n_lat, n_lon)
    land[0, 0] = 1.0
    land[1, 2] = 0.4
    ssf = 1 - land
    land_t = land.expand(n_time, n_lat, n_lon).clone()
    ssf_t = ssf.expand(n_time, n_lat, n_lon).clone()
    data = {
        "land_fraction": land_t,
        "sea_surface_fraction": ssf_t,
        "ocean_sea_ice_fraction": r(),
        "sst": r(10.0, 270.0),
        "simass": r(900.0),
        "sisnmass": r(300.0),
        "hflso": r(20.0, -10.0),
        "evs": r(1e-5),
        "prsn": r(1e-5),
        "hfrunoffds": r(5.0),
        "hfds_total_area": r(100.0, -50.0),
        "DSWRFsfc": r(300.0),
        "USWRFsfc": r(50.0),
        "DLWRFsfc": r(300.0),
        "ULWRFsfc": r(350.0),
        "LHTFLsfc": r(100.0),
        "SHTFLsfc": r(50.0),
        "PRATEsfc": r(1e-4),
        "total_frozen_precipitation_rate": r(1e-5),
    }
    for name in ["simass", "sisnmass", "hflso", "evs", "prsn", "hfrunoffds", "sst"]:
        data[name][:, 0, 0] = float("nan")
    data["hfds_total_area"][:, 0, 0] = float("nan")
    return data


def _numpy_residual(d: dict[str, np.ndarray], dt: float) -> np.ndarray:
    """Independent recomputation of the target residual."""
    lf, lv = LATENT_HEAT_OF_FREEZING, LATENT_HEAT_OF_VAPORIZATION
    d = {k: np.nan_to_num(v.astype(np.float64)) for k, v in d.items()}
    ssf = d["sea_surface_fraction"]
    m = ssf * (d["simass"] + d["sisnmass"])
    cr = -d["hflso"] + lv * d["evs"] - lf * d["prsn"]
    f_top = (
        d["DSWRFsfc"]
        - d["USWRFsfc"]
        + d["DLWRFsfc"]
        - d["ULWRFsfc"]
        - d["LHTFLsfc"]
        - d["SHTFLsfc"]
    )
    atmos = f_top - lf * d["total_frozen_precipitation_rate"]
    r = np.zeros_like(m)
    for k in range(1, m.shape[0]):
        sea_ice = d["ocean_sea_ice_fraction"][k - 1] * (1 - d["land_fraction"][k - 1])
        ocean_fraction = 1 - d["land_fraction"][k - 1] - sea_ice
        mass_heat = (
            SPECIFIC_HEAT_OF_SEA_WATER_CM4
            * (
                d["PRATEsfc"][k]
                + d["total_frozen_precipitation_rate"][k]
                - d["LHTFLsfc"][k] / lv
            )
            * (d["sst"][k - 1] - FREEZING_TEMPERATURE_KELVIN)
        )
        net = (atmos[k] + mass_heat + d["hfrunoffds"][k] - cr[k]) * ssf[k]
        hfds_c = net * ocean_fraction + d["hfds_total_area"][k] * (1 - ocean_fraction)
        s = -ssf[k] * atmos[k] + hfds_c - ssf[k] * d["hfrunoffds"][k] + ssf[k] * cr[k]
        r[k] = s - lf * (m[k] - m[k - 1]) / dt
    return r


def test_frozen_mass():
    d = _window()
    result = frozen_mass(d["simass"], d["sisnmass"], d["sea_surface_fraction"])
    expected = d["sea_surface_fraction"] * torch.nan_to_num(d["simass"] + d["sisnmass"])
    torch.testing.assert_close(result, expected)
    assert torch.isfinite(result).all()
    assert (result[:, 0, 0] == 0).all()


def test_calving_residue():
    d = _window()
    result = calving_residue(d["hflso"], d["evs"], d["prsn"])
    expected = torch.nan_to_num(
        -d["hflso"]
        + LATENT_HEAT_OF_VAPORIZATION * d["evs"]
        - LATENT_HEAT_OF_FREEZING * d["prsn"]
    )
    torch.testing.assert_close(result, expected)


def test_correct_hfds_runoff_and_calving_before_ssf():
    d = _window()
    net = torch.rand_like(d["sst"])
    of = torch.rand_like(d["sst"])
    ssf = d["sea_surface_fraction"]
    gen = torch.rand_like(d["sst"])
    runoff, cr = torch.rand_like(net), torch.rand_like(net)
    result = correct_hfds(
        net, gen, of, "prescribed", ssf, hfrunoffds=runoff, calving_residue=cr
    )
    expected = (net + runoff - cr) * ssf * of + gen * (1 - of)
    torch.testing.assert_close(result, expected)
    with pytest.raises(ValueError):
        correct_hfds(net, gen, of, "prescribed", ssf, hfrunoffds=runoff)


def test_frozen_mass_flux_sum():
    d = _window()
    d = {k: torch.nan_to_num(v) for k, v in d.items()}
    hfds, runoff, cr = (torch.rand_like(d["sst"]) for _ in range(3))
    ssf = d["sea_surface_fraction"]
    f_top = (
        d["DSWRFsfc"]
        - d["USWRFsfc"]
        + d["DLWRFsfc"]
        - d["ULWRFsfc"]
        - d["LHTFLsfc"]
        - d["SHTFLsfc"]
    )
    expected = (
        ssf * (LATENT_HEAT_OF_FREEZING * d["total_frozen_precipitation_rate"] - f_top)
        + hfds
        - ssf * runoff
        + ssf * cr
    )
    torch.testing.assert_close(frozen_mass_flux_sum(d, hfds, runoff, cr), expected)


def test_frozen_mass_energy_budget_residual():
    s, m1, m0 = torch.tensor(2.0), torch.tensor(5.0), torch.tensor(3.0)
    result = frozen_mass_energy_budget_residual(s, m1, m0, 10.0)
    torch.testing.assert_close(result, s - LATENT_HEAT_OF_FREEZING * 2.0 / 10.0)


def test_target_residual_matches_numpy():
    d = _window()
    dt = 5 * 86400.0
    result = target_frozen_mass_energy_budget_residual(d, dt)
    assert result.dtype == d["simass"].dtype
    assert (result[0] == 0).all()
    expected = _numpy_residual({k: v.numpy() for k, v in d.items()}, dt)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-3)


def test_target_residual_hfds_matches_corrector():
    """With zero runoff and calving, the loader's corrected hfds_total_area at k
    is what the corrector's ``_correct_hfds`` returns for a step k-1 -> k."""
    d = _window()
    d["hfrunoffds"] = torch.zeros_like(d["sst"])
    d["hflso"] = torch.zeros_like(d["sst"])
    d["evs"] = torch.zeros_like(d["sst"])
    d["prsn"] = torch.zeros_like(d["sst"])
    d64 = {k: torch.nan_to_num(v.double()) for k, v in d.items()}
    dt = 5 * 86400.0
    r = target_frozen_mass_energy_budget_residual(d64, dt)
    k = 2
    input_data = {
        n: d64[n][k - 1] for n in ["sst", "land_fraction", "ocean_sea_ice_fraction"]
    }
    forcing = {n: d64[n][k] for n in ATMOS_NAMES + ["sea_surface_fraction"]}
    gen = {"hfds_total_area": d64["hfds_total_area"][k]}
    hfds_c = _correct_hfds(input_data, gen, forcing, "prescribed")["hfds_total_area"]
    m = frozen_mass(d64["simass"], d64["sisnmass"], d64["sea_surface_fraction"])
    zero = torch.zeros_like(hfds_c)
    s = frozen_mass_flux_sum(forcing, hfds_c, zero, zero)
    expected = frozen_mass_energy_budget_residual(s, m[k], m[k - 1], dt)
    torch.testing.assert_close(r[k], expected)


def test_ocean_net_surface_energy_flux_alias():
    from fme.core.corrector import ocean

    assert ocean._compute_ocean_net_surface_energy_flux is ocean_net_surface_energy_flux

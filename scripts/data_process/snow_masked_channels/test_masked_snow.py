import numpy as np
import pytest
from masked_snow import PARENTS, land_snow_fields


def _inputs():
    land_frac = np.array([[1.0, 0.5], [0.8, 0.2]])
    valid = np.array([[True, True], [True, False]])
    swe = np.array([[[100.0, 100.0], [100.0, 100.0]]])
    return land_frac, valid, swe


def test_era5_divides_by_land_fraction_and_clips_cover():
    land_frac, valid, swe = _inputs()
    scf = np.array([[[0.5, 0.5], [0.9, 1.0]]])
    out_swe, out_scf = land_snow_fields(swe, scf, PARENTS["era5"], land_frac, valid)
    np.testing.assert_allclose(out_swe[0, 0], [100.0, 200.0])
    np.testing.assert_allclose(out_swe[0, 1, 0], 125.0)
    np.testing.assert_allclose(out_scf[0, 0], [0.5, 1.0])
    np.testing.assert_allclose(out_scf[0, 1, 0], 1.0)


def test_cm4_rescales_cover_only():
    land_frac, valid, swe = _inputs()
    scf = np.array([[[50.0, 100.0], [25.0, 0.0]]])
    out_swe, out_scf = land_snow_fields(swe, scf, PARENTS["cm4"], land_frac, valid)
    np.testing.assert_allclose(out_swe[0, 0], [100.0, 100.0])
    np.testing.assert_allclose(out_scf[0, 0], [0.5, 1.0])
    np.testing.assert_allclose(out_scf[0, 1, 0], 0.25)


@pytest.mark.parametrize("dataset", ["era5", "cm4"])
def test_nan_outside_mask_and_float32(dataset):
    land_frac, valid, swe = _inputs()
    scf = np.zeros_like(swe)
    out_swe, out_scf = land_snow_fields(swe, scf, PARENTS[dataset], land_frac, valid)
    assert np.isnan(out_swe[0, 1, 1]) and np.isnan(out_scf[0, 1, 1])
    assert np.isfinite(out_swe[0][valid]).all()
    assert out_swe.dtype == np.float32 and out_scf.dtype == np.float32


def test_era5_rejects_mask_cells_with_low_land_fraction():
    land_frac, _, swe = _inputs()
    valid = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        land_snow_fields(swe, np.zeros_like(swe), PARENTS["era5"], land_frac, valid)

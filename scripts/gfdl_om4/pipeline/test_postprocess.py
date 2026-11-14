import numpy as np
import pytest
import xarray as xr

from pipeline.postprocess import ChunkContext, kelvin_sst_postprocess


@pytest.mark.parametrize("celsius_name", ["SST", "tos"])
def test_kelvin_sst_output_name_follows_source(celsius_name):
    spec = kelvin_sst_postprocess(celsius_sst=celsius_name)
    ds = xr.Dataset({celsius_name: ("x", np.array([-1.8, 0.0, 28.5]))})
    context = ChunkContext(
        ocean_fraction=xr.ones_like(ds[celsius_name]), store="gs://toy"
    )
    out = spec.fn(ds, context)
    expected_name = f"{celsius_name}_degK"
    assert spec.adds == (expected_name,)
    np.testing.assert_allclose(out[expected_name], ds[celsius_name] + 273.15)
    assert out[expected_name].attrs["units"] == "K"

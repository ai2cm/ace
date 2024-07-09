import xarray as xr
from combine_stats import get_combined_stats


def test_get_combined_stats():
    import numpy as np

    arr1 = xr.DataArray(np.random.rand(10, 10), dims=["x", "y"])
    arr2 = xr.DataArray(np.random.rand(5, 10) * 2 + 5, dims=["x", "y"])
    full_field_datasets = [
        xr.Dataset(
            {"arr": arr1.std()},
            attrs={"input_samples": 100},
        ),
        xr.Dataset(
            {"arr": arr2.std()},
            attrs={"input_samples": 50},
        ),
    ]
    centering_datasets = [
        xr.Dataset(
            {"arr": arr1.mean()},
            attrs={"input_samples": 100},
        ),
        xr.Dataset(
            {"arr": arr2.mean()},
            attrs={"input_samples": 50},
        ),
    ]
    samples = xr.DataArray([100, 50], dims=["run"])
    average = get_combined_stats(full_field_datasets, centering_datasets, samples)
    combined_arr = np.concatenate([arr1.values.flatten(), arr2.values.flatten()])
    assert np.allclose(average["arr"], np.std(combined_arr))

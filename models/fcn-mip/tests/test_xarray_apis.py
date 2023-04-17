"""
xarray is confusing so let's test it to gain understanding
"""
import xarray
import numpy as np
import pytest


def test_xarray_var_reference():
    ds = xarray.DataArray(np.ones((10, 10)), dims=("x", "y"))
    ds["x"] = np.arange(10)
    ds["y"] = np.arange(10)
    datasets = ds.to_dataset(name="wind")
    datasets["var"] = datasets["wind"]
    assert isinstance(datasets["var"], xarray.DataArray)


def test_xarray_loop():
    ds = xarray.DataArray(np.ones((10, 10)), dims=("x", "y"))
    ds["x"] = np.arange(10)
    ds["y"] = np.arange(10)
    datasets = ds.to_dataset(name="wind")

    assert list(datasets) == ["wind"]
    assert set(datasets.variables) == {"x", "y", "wind"}


def test_xarray_var():
    ds = xarray.DataArray(np.ones((10, 10)), dims=("x", "y"))
    ds["x"] = np.arange(10)
    ds["y"] = np.arange(10)
    datasets = ds.to_dataset(name="wind")

    with pytest.raises(AttributeError):
        datasets.variables["wind"].weighted

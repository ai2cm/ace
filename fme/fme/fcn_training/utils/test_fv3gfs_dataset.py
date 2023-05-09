import copy
import os
from typing import Any, Dict
import netCDF4
from .data_loader_fv3gfs import FV3GFSDataset
import pytest

TEST_PATH = "/traindata"
TEST_STATS_PATH = "/statsdata"
TEST_IN_NAMES = ["UGRD10m", "VGRD10m", "PRMSL", "TCWV"]
TEST_OUT_NAMES = ["UGRD10m", "VGRD10m", "TMP850", "TCWV"]
TEST_PARAMS: Dict[str, Any] = {
    "n_history": 0,
    "in_channels": [0, 1, 4, 19],
    "out_channels": [0, 1, 5, 19],
    "two_step_training": False,
    "dt": 1,
    "n_history": 0,
    "normalization": "zscore",
    "global_means_path": os.path.join(TEST_STATS_PATH, "fv3gfs-mean.nc"),
    "global_stds_path": os.path.join(TEST_STATS_PATH, "fv3gfs-stddev.nc"),
    "time_means_path": os.path.join(TEST_STATS_PATH, "fv3gfs-time-mean.nc"),
}
TEST_PARAMS_SPECIFY_BY_NAME = copy.copy(TEST_PARAMS)
TEST_PARAMS_SPECIFY_BY_NAME.pop("in_channels")
TEST_PARAMS_SPECIFY_BY_NAME.pop("out_channels")
TEST_PARAMS_SPECIFY_BY_NAME["in_names"] = TEST_IN_NAMES
TEST_PARAMS_SPECIFY_BY_NAME["out_names"] = TEST_OUT_NAMES


class DotDict:
    """Overrides the dot operator for accessing fields to be dict lookup.

    e.g. `dotdict.key == dotdict['key']  == dotdict.items['key']`
    """

    def __init__(self, items):
        self.items = items

    def __getitem__(self, key):
        return self.items[key]

    def __getattr__(self, attr):
        return self.items[attr]

    def __contains__(self, key):
        return key in self.items


@pytest.mark.skip(reason="Requires access to sample training data.")
@pytest.mark.parametrize("params", [TEST_PARAMS, TEST_PARAMS_SPECIFY_BY_NAME])
def test_FV3GFSDataset_init(params):
    dataset = FV3GFSDataset(DotDict(params), TEST_PATH, True)
    assert dataset.in_names == TEST_IN_NAMES
    assert dataset.out_names == TEST_OUT_NAMES


@pytest.mark.skip(reason="Requires access to sample training data.")
def test_FV3GFSDataset_len():
    dataset = FV3GFSDataset(DotDict(TEST_PARAMS), TEST_PATH, True)
    full_path = os.path.join(TEST_PATH, "*.nc")
    expected_length = len(netCDF4.MFDataset(full_path).variables["time"][:]) - 1
    assert len(dataset) == expected_length


@pytest.mark.skip(reason="Requires access to sample training data.")
def test_FV3GFSDataset_getitem():
    dataset = FV3GFSDataset(DotDict(TEST_PARAMS), TEST_PATH, True)
    output = dataset[150]
    assert len(output) == 2
    assert output[0].shape == (len(TEST_PARAMS["in_channels"]), 180, 360)
    assert output[1].shape == (len(TEST_PARAMS["out_channels"]), 180, 360)


@pytest.mark.skip(reason="Requires access to sample training data.")
def test_FV3GFSDataset_raises_value_error_if_names_and_channels_both_specified():
    params = copy.copy(TEST_PARAMS)
    params["in_names"] = ["foo", "bar"]
    with pytest.raises(ValueError):
        FV3GFSDataset(DotDict(params), TEST_PATH, True)

    params = copy.copy(TEST_PARAMS)
    params["out_names"] = ["foo", "bar"]
    with pytest.raises(ValueError):
        FV3GFSDataset(DotDict(params), TEST_PATH, True)

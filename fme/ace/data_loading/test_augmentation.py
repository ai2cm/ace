import random

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.data_loading.augmentation import RotateModifier
from fme.ace.data_loading.batch_data import BatchData


def rotate(data: torch.Tensor) -> torch.Tensor:
    return torch.flip(data, dims=[-2, -1])


def test_rotate_modifier_all_rotation():
    rotate_modifier = RotateModifier(
        rotate_probability=1.0, additional_directional_names=[]
    )
    n_lat = 8
    n_lon = 16
    batch = BatchData(
        data={
            "UGRD": torch.randn(1, 2, n_lat, n_lon),
            "VGRD": torch.randn(1, 2, n_lat, n_lon),
            "PS": torch.randn(1, 2, n_lat, n_lon),
        },
        time=xr.DataArray(np.zeros((1, 2)), dims=["sample", "time"]),
        horizontal_dims=["lat", "lon"],
    )
    rotated_batch = rotate_modifier(batch)
    assert rotated_batch.data["UGRD"].shape == (1, 2, n_lat, n_lon)
    assert torch.allclose(rotate(rotated_batch.data["UGRD"]), -1 * batch.data["UGRD"])
    assert torch.allclose(rotate(rotated_batch.data["VGRD"]), -1 * batch.data["VGRD"])
    assert torch.allclose(rotate(rotated_batch.data["PS"]), batch.data["PS"])


def test_rotate_modifier_no_rotation():
    rotate_modifier = RotateModifier(
        rotate_probability=0.0, additional_directional_names=[]
    )
    n_lat = 8
    n_lon = 16
    batch = BatchData(
        data={
            "UGRD": torch.randn(1, 2, n_lat, n_lon),
            "VGRD": torch.randn(1, 2, n_lat, n_lon),
            "PS": torch.randn(1, 2, n_lat, n_lon),
        },
        time=xr.DataArray(np.zeros((1, 2)), dims=["sample", "time"]),
        horizontal_dims=["lat", "lon"],
    )
    rotated_batch = rotate_modifier(batch)
    assert rotated_batch.data["UGRD"].shape == (1, 2, n_lat, n_lon)
    assert torch.allclose(rotated_batch.data["UGRD"], batch.data["UGRD"])
    assert torch.allclose(rotated_batch.data["VGRD"], batch.data["VGRD"])
    assert torch.allclose(rotated_batch.data["PS"], batch.data["PS"])


def test_rotate_modifier_random_rotation():
    random.seed(0)
    rotate_modifier = RotateModifier(
        rotate_probability=0.5, additional_directional_names=[]
    )
    n_lat = 8
    n_lon = 16
    batch = BatchData(
        data={
            "UGRD": torch.randn(40, 2, n_lat, n_lon),
            "VGRD": torch.randn(40, 2, n_lat, n_lon),
            "PS": torch.randn(40, 2, n_lat, n_lon),
        },
        time=xr.DataArray(np.zeros((40, 2)), dims=["sample", "time"]),
        horizontal_dims=["lat", "lon"],
    )
    rotated_batch = rotate_modifier(batch)
    assert rotated_batch.data.keys() == batch.data.keys()
    assert rotated_batch.data["UGRD"].shape == (40, 2, n_lat, n_lon)
    rotated = {}
    unrotated = {}
    for name in rotated_batch.data:
        unrotated[name] = np.all(
            torch.abs(batch.data[name] - rotated_batch.data[name]).cpu().numpy() < 1e-6,
            axis=(1, 2, 3),
        )
        if name in ("UGRD", "VGRD"):
            sign = -1
        else:
            sign = 1
        rotated[name] = np.all(
            torch.abs(sign * rotate(batch.data[name]) - rotated_batch.data[name])
            .cpu()
            .numpy()
            < 1e-6,
            axis=(1, 2, 3),
        )
        assert np.all(rotated[name] + unrotated[name] == 1), name
        assert np.sum(rotated[name]) > 0, name
        assert np.sum(unrotated[name]) > 0, name
    for name in ("VGRD", "PS"):
        assert np.all(rotated[name] == rotated["UGRD"]), name
        assert np.all(unrotated[name] == unrotated["UGRD"]), name


@pytest.mark.parametrize(
    "name, additional_directional_names, match_expected",
    [
        ("UGRD", [], True),
        ("VGRD", [], True),
        ("UGRD_10m", [], True),
        ("UGRD_10m", ["UGRD"], True),
        ("VGRD200", [], True),
        ("eastward_wind_3", [], True),
        ("UGRD10m", [], True),
        ("NWIND10m", [], False),
        ("NWIND10m", ["NWIND"], True),
    ],
)
def test_rotate_modifier_pattern(
    name: str, additional_directional_names: list[str], match_expected: bool
):
    rotate_modifier = RotateModifier(
        rotate_probability=1.0,
        additional_directional_names=additional_directional_names,
    )
    assert (rotate_modifier._pattern.match(name) is not None) == match_expected, name

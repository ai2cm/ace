import pathlib
import h5py

from datasets import era5


def test_open_34_vars(tmp_path: pathlib.Path):
    path = tmp_path / "1979.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("fields", shape=[1, 34, 721, 1440])

    ds = era5.open_34_vars(path)
    # ensure that data can be grabbed
    ds.mean().compute()

    assert set(ds.coords) == {"time", "channel", "lat", "lon"}

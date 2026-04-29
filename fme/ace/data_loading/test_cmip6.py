import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fme.ace.data_loading.cmip6 import Cmip6DataConfig
from fme.core.dataset.xarray import XarrayDataConfig


def _make_zarr(path, n_times=10, n_lat=4, n_lon=8, varnames=("tas", "pr")):
    """Create a minimal zarr store with the given variables."""
    times = xr.cftime_range("2000-01-01", periods=n_times, freq="D", calendar="noleap")
    ds = xr.Dataset(
        {
            name: xr.DataArray(
                np.random.randn(n_times, n_lat, n_lon).astype(np.float32),
                dims=["time", "lat", "lon"],
                attrs={"units": "K", "long_name": name},
            )
            for name in varnames
        },
        coords={
            "time": times,
            "lat": np.linspace(-90, 90, n_lat),
            "lon": np.linspace(0, 360, n_lon, endpoint=False),
        },
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_zarr(path)
    return ds


def _make_index(data_dir, rows):
    """Create a minimal index.csv in data_dir from a list of row dicts."""
    records = []
    for row in rows:
        record = {
            "source_id": row["source_id"],
            "experiment": row["experiment"],
            "variant_label": row["variant_label"],
            "variant_r": row.get("variant_r", 1),
            "variant_i": 1,
            "variant_p": 1,
            "variant_f": 1,
            "label": row.get("label", f"{row['source_id']}.p{row.get('variant_p', 1)}"),
            "output_zarr": os.path.join(
                data_dir,
                row["source_id"],
                row["experiment"],
                row["variant_label"],
                "data.zarr",
            ),
            "status": row.get("status", "ok"),
            "skip_reason": "",
        }
        records.append(record)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(data_dir, "index.csv"), index=False)
    return df


@pytest.fixture
def cmip6_data_dir(tmp_path):
    """Create a mock CMIP6 data directory with two models and two experiments."""
    data_dir = str(tmp_path / "cmip6")
    os.makedirs(data_dir)

    rows = [
        {
            "source_id": "ModelA",
            "experiment": "historical",
            "variant_label": "r1i1p1f1",
            "variant_r": 1,
        },
        {
            "source_id": "ModelA",
            "experiment": "ssp585",
            "variant_label": "r1i1p1f1",
            "variant_r": 1,
        },
        {
            "source_id": "ModelA",
            "experiment": "historical",
            "variant_label": "r2i1p1f1",
            "variant_r": 2,
        },
        {
            "source_id": "ModelB",
            "experiment": "historical",
            "variant_label": "r1i1p1f1",
            "variant_r": 1,
        },
        {
            "source_id": "ModelC",
            "experiment": "historical",
            "variant_label": "r1i1p1f1",
            "variant_r": 1,
            "status": "skipped",
        },
    ]
    _make_index(data_dir, rows)

    for row in rows:
        if row.get("status", "ok") == "ok":
            zarr_path = os.path.join(
                data_dir,
                str(row["source_id"]),
                str(row["experiment"]),
                str(row["variant_label"]),
                "data.zarr",
            )
            _make_zarr(zarr_path)

    return data_dir


def test_defaults_load_all_ok(cmip6_data_dir):
    config = Cmip6DataConfig(data_dir=cmip6_data_dir)
    assert config.zarr_engine_used is True
    labels = config.available_labels
    assert labels is not None
    assert "ModelA.p1" in labels
    assert "ModelB.p1" in labels


def test_filter_source_ids(cmip6_data_dir):
    config = Cmip6DataConfig(data_dir=cmip6_data_dir, source_ids=["ModelA"])
    labels = config.available_labels
    assert labels == {"ModelA.p1"}


def test_filter_experiments(cmip6_data_dir):
    config = Cmip6DataConfig(data_dir=cmip6_data_dir, experiments=["ssp585"])
    concat = config._get_concat_config()
    assert len(concat.concat) == 1
    assert "ModelA" in concat.concat[0].data_path


def test_filter_realizations(cmip6_data_dir):
    config = Cmip6DataConfig(data_dir=cmip6_data_dir, realizations=[2])
    concat = config._get_concat_config()
    assert len(concat.concat) == 1
    assert "r2i1p1f1" in concat.concat[0].data_path


def test_skipped_datasets_excluded(cmip6_data_dir):
    config = Cmip6DataConfig(
        data_dir=cmip6_data_dir,
        experiments=["historical"],
    )
    labels = config.available_labels
    assert labels is not None
    assert "ModelC.p1" not in labels


def test_no_matching_datasets_raises(cmip6_data_dir):
    config = Cmip6DataConfig(
        data_dir=cmip6_data_dir,
        source_ids=["NonexistentModel"],
    )
    with pytest.raises(ValueError, match="No datasets"):
        config.available_labels


def test_concat_config_structure(cmip6_data_dir):
    config = Cmip6DataConfig(data_dir=cmip6_data_dir)
    concat = config._get_concat_config()
    assert concat.strict is False
    for xarray_config in concat.concat:
        assert isinstance(xarray_config, XarrayDataConfig)
        assert xarray_config.engine == "zarr"
        assert xarray_config.file_pattern == "data.zarr"
        assert xarray_config.labels is not None
        assert len(xarray_config.labels) == 1


def test_dacite_loading(cmip6_data_dir):
    import dacite

    from fme.ace.data_loading.config import DataLoaderConfig

    config_dict = {
        "dataset": {
            "data_dir": cmip6_data_dir,
            "source_ids": ["ModelA"],
            "experiments": ["historical"],
        },
        "batch_size": 1,
    }
    config = dacite.from_dict(
        data_class=DataLoaderConfig,
        data=config_dict,
        config=dacite.Config(strict=True),
    )
    assert isinstance(config.dataset, Cmip6DataConfig)
    assert config.zarr_engine_used is True


def test_build(cmip6_data_dir):
    config = Cmip6DataConfig(
        data_dir=cmip6_data_dir,
        source_ids=["ModelA"],
        experiments=["historical"],
        realizations=[1],
    )
    from fme.core.dataset.schedule import IntSchedule

    dataset, properties = config.build(
        names=["tas", "pr"],
        n_timesteps=IntSchedule.from_constant(2),
    )
    assert len(dataset) > 0

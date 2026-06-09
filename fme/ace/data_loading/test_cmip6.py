import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fme.ace.data_loading.cmip6 import Cmip6DataConfig, Cmip6TimeKeep, Cmip6TimeMask
from fme.core.dataset.time import TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.typing_ import Slice


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


def test_netcdf4_engine(cmip6_data_dir):
    """Test that engine='netcdf4' produces the right file_pattern and engine."""
    config = Cmip6DataConfig(data_dir=cmip6_data_dir, engine="netcdf4")
    assert config.zarr_engine_used is False
    assert config._file_pattern == "data.*.nc"
    concat = config._get_concat_config()
    for xarray_config in concat.concat:
        assert xarray_config.engine == "netcdf4"
        assert xarray_config.file_pattern == "data.*.nc"


def test_invalid_engine():
    with pytest.raises(ValueError, match="engine must be"):
        Cmip6DataConfig(data_dir="/nonexistent", engine="h5netcdf")  # type: ignore[arg-type]


def _make_yearly_netcdfs(zarr_path: str, nc_dir: str):
    """Convert a zarr store to yearly netCDF files with 1-day chunks."""
    ds = xr.open_zarr(zarr_path)
    ds.load()
    os.makedirs(nc_dir, exist_ok=True)
    encoding = {
        name: {"chunksizes": (1,) + ds[name].shape[1:]}
        for name in ds.data_vars
        if "time" in ds[name].dims
    }
    for year, yearly_ds in ds.groupby("time.year"):
        yearly_ds.to_netcdf(os.path.join(nc_dir, f"data.{year}.nc"), encoding=encoding)
    ds.close()


@pytest.fixture
def cmip6_netcdf_dir(tmp_path, cmip6_data_dir):
    """Create a netCDF mirror of the zarr cmip6_data_dir with yearly files."""
    nc_dir = str(tmp_path / "cmip6-nc")
    os.makedirs(nc_dir)

    import shutil

    for f in os.listdir(cmip6_data_dir):
        src = os.path.join(cmip6_data_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(nc_dir, f))

    idx = pd.read_csv(os.path.join(cmip6_data_dir, "index.csv"))
    for _, row in idx.iterrows():
        if row["status"] != "ok":
            continue
        rel = os.path.join(row["source_id"], row["experiment"], row["variant_label"])
        zarr_path = os.path.join(cmip6_data_dir, rel, "data.zarr")
        nc_subdir = os.path.join(nc_dir, rel)
        _make_yearly_netcdfs(zarr_path, nc_subdir)

    return nc_dir


def test_netcdf4_build(cmip6_netcdf_dir):
    """End-to-end: build a dataset from yearly netCDF files."""
    from fme.core.dataset.schedule import IntSchedule

    config = Cmip6DataConfig(
        data_dir=cmip6_netcdf_dir,
        source_ids=["ModelA"],
        experiments=["historical"],
        realizations=[1],
        engine="netcdf4",
    )
    dataset, properties = config.build(
        names=["tas", "pr"],
        n_timesteps=IntSchedule.from_constant(2),
    )
    assert len(dataset) > 0
    sample = dataset[0]
    data, time, labels, epoch, _source_ids = sample
    assert "tas" in data
    assert "pr" in data


# ---------------------------------------------------------------------------
# exclude_variants: drop specific (source, experiment, variant) triples
# ---------------------------------------------------------------------------


def test_exclude_variants_drops_matching_rows(cmip6_data_dir):
    """``exclude_variants`` is subtractive: it drops a specific row
    after the include filters have run. Other variants of the same
    source remain."""
    config = Cmip6DataConfig(
        data_dir=cmip6_data_dir,
        exclude_variants=[["ModelA", "historical", "r2i1p1f1"]],
    )
    concat = config._get_concat_config()
    paths = [c.data_path for c in concat.concat]
    assert not any("r2i1p1f1" in p for p in paths)
    # The r1 variant of ModelA/historical and ModelA/ssp585 are still in.
    assert any("ModelA/historical/r1i1p1f1" in p for p in paths)
    assert any("ModelA/ssp585/r1i1p1f1" in p for p in paths)


def test_exclude_variants_unknown_triple_raises(cmip6_data_dir):
    """Listing a triple that's not in index.csv is almost certainly a
    typo; raise to catch it loudly rather than silently dropping
    nothing."""
    with pytest.raises(ValueError, match="not present in index.csv"):
        Cmip6DataConfig(
            data_dir=cmip6_data_dir,
            exclude_variants=[["NonexistentModel", "historical", "r1i1p1f1"]],
        )._get_concat_config()


def test_exclude_variants_invalid_triple_shape_raises(cmip6_data_dir):
    """Each exclude_variants entry must be a 3-element list. A
    2-element entry is a config bug — raise at __post_init__ so the
    failure is visible before any data load."""
    with pytest.raises(ValueError, match="triple"):
        Cmip6DataConfig(
            data_dir=cmip6_data_dir,
            exclude_variants=[["ModelA", "historical"]],
        )


# ---------------------------------------------------------------------------
# time_masks: split matched datasets into pre-mask + post-mask slices
# ---------------------------------------------------------------------------


def test_time_mask_splits_matched_dataset(cmip6_data_dir):
    """A matched (source, experiment) dataset becomes two
    XarrayDataConfig entries — one with stop_time=keep_before, one
    with start_time=keep_after — both labelled identically so
    downstream per-source handling treats them as the same dataset."""
    config = Cmip6DataConfig(
        data_dir=cmip6_data_dir,
        source_ids=["ModelA"],
        experiments=["historical"],
        time_masks=[
            Cmip6TimeMask(
                source_ids=["ModelA"],
                experiments=["historical"],
                keep_before="2000-01-04",
                keep_after="2000-01-08",
            )
        ],
    )
    concat = config._get_concat_config()
    # ModelA has two historical variants (r1, r2). Each gets split, so
    # 4 entries total.
    assert len(concat.concat) == 4
    # All four point at a ModelA/historical/r? directory.
    for entry in concat.concat:
        assert "ModelA/historical/" in entry.data_path
    # Each variant produces a pre-mask (stop_time set) and a post-mask
    # (start_time set) entry; labels match across the pair.
    pre = [
        e
        for e in concat.concat
        if isinstance(e.subset, TimeSlice) and e.subset.stop_time == "2000-01-04"
    ]
    post = [
        e
        for e in concat.concat
        if isinstance(e.subset, TimeSlice) and e.subset.start_time == "2000-01-08"
    ]
    assert len(pre) == 2
    assert len(post) == 2
    for entry in pre + post:
        assert isinstance(entry.subset, TimeSlice)
        assert entry.labels == ["ModelA.p1"]


def test_time_mask_skips_non_matching_dataset(cmip6_data_dir):
    """A dataset whose (source, experiment) doesn't match any mask
    keeps its single un-subsetted entry — masks are opt-in per
    (source, experiment) pair."""
    config = Cmip6DataConfig(
        data_dir=cmip6_data_dir,
        experiments=["historical", "ssp585"],
        time_masks=[
            Cmip6TimeMask(
                source_ids=["ModelA"],
                experiments=["historical"],
                keep_before="2000-01-04",
                keep_after="2000-01-08",
            )
        ],
    )
    concat = config._get_concat_config()
    # ModelA/historical r1 + r2: 2 datasets × 2 (pre/post) = 4.
    # ModelA/ssp585 r1: 1 dataset, no mask, 1 entry.
    # ModelB/historical r1: 1 dataset, no mask, 1 entry.
    # Total: 6.
    assert len(concat.concat) == 6
    ssp_entries = [e for e in concat.concat if "ssp585" in e.data_path]
    assert len(ssp_entries) == 1
    # Unmasked entry has a default empty Slice/TimeSlice with no
    # bounds set — gate the attribute access on the type so mypy is
    # happy.
    ssp_subset = ssp_entries[0].subset
    if isinstance(ssp_subset, TimeSlice):
        assert ssp_subset.start_time is None
        assert ssp_subset.stop_time is None


def test_time_mask_overlapping_configs_raise():
    """Two masks targeting the same (source, experiment) pair are a
    config bug — only one mask per pair is supported. Raise at
    __post_init__ so it fails before any data load."""
    with pytest.raises(ValueError, match="Multiple time_masks match"):
        Cmip6DataConfig(
            data_dir="/nonexistent",
            time_masks=[
                Cmip6TimeMask(
                    source_ids=["ModelA"],
                    experiments=["historical"],
                    keep_before="1970-12-31",
                    keep_after="1980-01-01",
                ),
                Cmip6TimeMask(
                    source_ids=["ModelA", "ModelB"],
                    experiments=["historical"],
                    keep_before="1990-12-31",
                    keep_after="2000-01-01",
                ),
            ],
        )


def test_time_mask_end_to_end_build(cmip6_netcdf_dir):
    """End-to-end smoke test: build a real dataset from a time-masked
    Cmip6DataConfig. The mask covers timesteps 4-7 of a 10-step
    fixture (2000-01-04 through 2000-01-07 in noleap), so keep_before
    2000-01-03 + keep_after 2000-01-08 leaves 6 timesteps total per
    variant (3 pre + 3 post)."""
    from fme.core.dataset.schedule import IntSchedule

    config = Cmip6DataConfig(
        data_dir=cmip6_netcdf_dir,
        source_ids=["ModelA"],
        experiments=["historical"],
        realizations=[1],
        engine="netcdf4",
        time_masks=[
            Cmip6TimeMask(
                source_ids=["ModelA"],
                experiments=["historical"],
                keep_before="2000-01-03",
                keep_after="2000-01-08",
            )
        ],
    )
    dataset, _properties = config.build(
        names=["tas", "pr"],
        n_timesteps=IntSchedule.from_constant(2),
    )
    # 6 surviving timesteps × n_timesteps=2 window → 5 samples
    # (overlapping windows), but the pre/post boundary means windows
    # crossing it are dropped — exact count depends on the concat
    # internals. The substantive assertion: dataset built without
    # error and is non-empty.
    assert len(dataset) > 0


# ---------------------------------------------------------------------------
# time_keeps: keep only a contiguous window of matched datasets
# (inverse of time_masks; for the eval side of temporal-interp holdouts)
# ---------------------------------------------------------------------------


def test_time_keep_restricts_matched_dataset_to_window(cmip6_data_dir):
    """``time_keeps`` produces one XarrayDataConfig per matched
    dataset, with ``TimeSlice(start_time=keep_start, stop_time=
    keep_end)``. Unmatched datasets keep their default un-subsetted
    entry."""
    config = Cmip6DataConfig(
        data_dir=cmip6_data_dir,
        experiments=["historical", "ssp585"],
        time_keeps=[
            Cmip6TimeKeep(
                source_ids=["ModelA"],
                experiments=["historical"],
                keep_start="2000-01-04",
                keep_end="2000-01-07",
            )
        ],
    )
    concat = config._get_concat_config()
    # ModelA/historical r1 + r2: 2 datasets, each becomes ONE windowed entry.
    # ModelA/ssp585 r1: 1 dataset, untouched (no time mask, no time keep).
    # ModelB/historical r1: 1 dataset, untouched.
    # Total: 4 entries (not split — unlike time_masks).
    assert len(concat.concat) == 4
    windowed = [
        e
        for e in concat.concat
        if isinstance(e.subset, TimeSlice) and e.subset.start_time == "2000-01-04"
    ]
    assert len(windowed) == 2
    for entry in windowed:
        assert isinstance(entry.subset, TimeSlice)
        assert entry.subset.stop_time == "2000-01-07"
        assert "ModelA/historical/" in entry.data_path
        assert entry.labels == ["ModelA.p1"]


def test_subsample_step_applied_to_plain_entries(cmip6_data_dir):
    """``subsample_step`` propagates to ``XarrayDataConfig.subset.step``
    for entries that would otherwise be un-subsetted."""
    config = Cmip6DataConfig(data_dir=cmip6_data_dir, subsample_step=7)
    concat = config._get_concat_config()
    for entry in concat.concat:
        assert isinstance(entry.subset, Slice)
        assert entry.subset.step == 7
        assert entry.subset.start is None
        assert entry.subset.stop is None


def test_subsample_step_applied_to_time_keep_window(cmip6_data_dir):
    """``subsample_step`` writes through to the ``TimeSlice.step`` on
    matched ``time_keeps`` entries."""
    config = Cmip6DataConfig(
        data_dir=cmip6_data_dir,
        experiments=["historical", "ssp585"],
        time_keeps=[
            Cmip6TimeKeep(
                source_ids=["ModelA"],
                experiments=["historical"],
                keep_start="2000-01-04",
                keep_end="2000-01-07",
            )
        ],
        subsample_step=3,
    )
    concat = config._get_concat_config()
    windowed = [
        e
        for e in concat.concat
        if isinstance(e.subset, TimeSlice) and e.subset.start_time == "2000-01-04"
    ]
    assert len(windowed) == 2
    for entry in windowed:
        assert isinstance(entry.subset, TimeSlice)
        assert entry.subset.step == 3
    # Untouched entries still get a Slice with the step.
    plain = [e for e in concat.concat if isinstance(e.subset, Slice)]
    for entry in plain:
        assert isinstance(entry.subset, Slice)
        assert entry.subset.step == 3


def test_subsample_step_applied_to_time_mask_split_entries(cmip6_data_dir):
    """``subsample_step`` writes through to both the pre-mask and the
    post-mask ``TimeSlice``s when a ``time_masks`` entry matches."""
    config = Cmip6DataConfig(
        data_dir=cmip6_data_dir,
        source_ids=["ModelA"],
        experiments=["historical"],
        time_masks=[
            Cmip6TimeMask(
                source_ids=["ModelA"],
                experiments=["historical"],
                keep_before="2000-01-04",
                keep_after="2000-01-08",
            )
        ],
        subsample_step=5,
    )
    concat = config._get_concat_config()
    for entry in concat.concat:
        assert isinstance(entry.subset, TimeSlice)
        assert entry.subset.step == 5


def test_subsample_step_rejects_invalid_values():
    with pytest.raises(ValueError, match="subsample_step must be >= 1"):
        Cmip6DataConfig(data_dir="/nonexistent", subsample_step=0)


def test_time_keep_and_time_mask_on_same_pair_raises():
    """A single (source, experiment) pair can't appear in both
    time_masks and time_keeps — the two are inverse operations and
    mixing them produces a contradictory slice list."""
    with pytest.raises(ValueError, match="both time_masks and time_keeps"):
        Cmip6DataConfig(
            data_dir="/nonexistent",
            time_masks=[
                Cmip6TimeMask(
                    source_ids=["ModelA"],
                    experiments=["historical"],
                    keep_before="1969-12-31",
                    keep_after="1990-01-01",
                )
            ],
            time_keeps=[
                Cmip6TimeKeep(
                    source_ids=["ModelA"],
                    experiments=["historical"],
                    keep_start="1970-07-01",
                    keep_end="1989-06-30",
                )
            ],
        )


def test_time_keep_overlapping_configs_raise():
    """Two keeps targeting the same (source, experiment) pair is a
    config bug — raise at __post_init__."""
    with pytest.raises(ValueError, match="Multiple time_keeps match"):
        Cmip6DataConfig(
            data_dir="/nonexistent",
            time_keeps=[
                Cmip6TimeKeep(
                    source_ids=["ModelA"],
                    experiments=["historical"],
                    keep_start="1970-01-01",
                    keep_end="1980-12-31",
                ),
                Cmip6TimeKeep(
                    source_ids=["ModelA", "ModelB"],
                    experiments=["historical"],
                    keep_start="1990-01-01",
                    keep_end="2000-12-31",
                ),
            ],
        )


def test_time_keep_end_to_end_build(cmip6_netcdf_dir):
    """End-to-end smoke: keep only timesteps 4-7 of a 10-step fixture
    via ``time_keeps``, build the dataset, and confirm it's non-empty.
    Complements the time_masks end-to-end test by exercising the
    opposite polarity."""
    from fme.core.dataset.schedule import IntSchedule

    config = Cmip6DataConfig(
        data_dir=cmip6_netcdf_dir,
        source_ids=["ModelA"],
        experiments=["historical"],
        realizations=[1],
        engine="netcdf4",
        time_keeps=[
            Cmip6TimeKeep(
                source_ids=["ModelA"],
                experiments=["historical"],
                keep_start="2000-01-04",
                keep_end="2000-01-07",
            )
        ],
    )
    dataset, _properties = config.build(
        names=["tas", "pr"],
        n_timesteps=IntSchedule.from_constant(2),
    )
    assert len(dataset) > 0

"""Tests for the translate multi-stream, paired-by-time data loader.

Data is synthesized on disk as tiny netCDF files via ``fme.ace.testing``; no
real dataset is read.
"""

import pathlib

import numpy as np
import pytest

from fme.ace.testing import DimSizes, save_nd_netcdf
from fme.core.coordinates import DimSize
from fme.core.dataset.schedule import IntMilestone, IntSchedule
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.rand import set_seed
from fme.core.typing_ import Slice
from fme.translate.data.config import StreamConfig, TranslateDataLoaderConfig
from fme.translate.data.getters import get_gridded_data
from fme.translate.data.requirements import (
    ObjectiveDataRequirements,
    StreamRequirements,
    TranslateDataRequirements,
)

NAMES = ["temp", "humidity"]
# 1°/2°/4°-shaped grids, scaled down to keep the synthetic files tiny.
SHAPES = {"era5_1deg": (16, 32), "era5_2deg": (8, 16), "era5_4deg": (4, 8)}


def _write_stream(
    path: pathlib.Path,
    img_shape: tuple[int, int],
    n_times: int,
    names: list[str] = NAMES,
    timestep_days: float = 1.0,
) -> XarrayDataConfig:
    """Write one stream's netCDF file and return the config that reads it."""
    path.mkdir(parents=True, exist_ok=True)
    save_nd_netcdf(
        path / "data.nc",
        dim_sizes=DimSizes(
            n_time=n_times,
            horizontal=[DimSize("lat", img_shape[0]), DimSize("lon", img_shape[1])],
            nz_interface=3,
        ),
        variable_names=names,
        timestep_days=timestep_days,
    )
    return XarrayDataConfig(data_path=str(path))


def _multi_resolution_config(
    tmp_path: pathlib.Path, n_times: int = 8, batch_size: int = 2
) -> TranslateDataLoaderConfig:
    """Three streams of the same state at three resolutions, identical times."""
    return TranslateDataLoaderConfig(
        streams=[
            StreamConfig(
                name=name,
                domain=name.replace("era5_", "atmos_"),
                dataset=_write_stream(tmp_path / name, shape, n_times),
            )
            for name, shape in SHAPES.items()
        ],
        batch_size=batch_size,
    )


def _requirements(
    *objectives: dict[str, list[str]], n_timesteps: int = 1
) -> TranslateDataRequirements:
    """Requirements for objectives each consuming the named streams."""
    return TranslateDataRequirements.from_objectives(
        [
            ObjectiveDataRequirements(
                streams={
                    stream: StreamRequirements(names=names, n_timesteps=n_timesteps)
                    for stream, names in objective.items()
                }
            )
            for objective in objectives
        ]
    )


def _start_times(batch, stream: str) -> np.ndarray:
    return batch[stream].time.isel(time=0).values


def test_every_batch_carries_all_paired_streams_at_matching_times(tmp_path):
    """The multi-resolution story: one objective over three resolutions."""
    config = _multi_resolution_config(tmp_path)
    data = get_gridded_data(
        config,
        _requirements({name: NAMES for name in SHAPES}),
        train=True,
    )
    assert data.batch_size == 2
    assert data.n_batches == 4  # 8 times, window of 1, batch of 2
    assert data.n_samples == 8

    batches = list(data.loader)
    assert len(batches) == data.n_batches
    for batch in batches:
        assert sorted(batch) == sorted(SHAPES)
        for name, shape in SHAPES.items():
            assert batch[name].data["temp"].shape == (2, 1, *shape)
        reference = _start_times(batch, "era5_1deg")
        for name in SHAPES:
            np.testing.assert_array_equal(_start_times(batch, name), reference)


def test_independently_sampled_streams_are_not_time_locked(tmp_path):
    """The transfer-learning story: no objective reads both streams."""
    config = TranslateDataLoaderConfig(
        streams=[
            StreamConfig(
                domain="era5", dataset=_write_stream(tmp_path / "a", (4, 8), 9)
            ),
            StreamConfig(
                domain="shield_c96", dataset=_write_stream(tmp_path / "b", (4, 8), 9)
            ),
        ],
        batch_size=1,
    )
    requirements = _requirements({"era5": NAMES}, {"shield_c96": NAMES})
    assert requirements.pairing_groups == [["era5"], ["shield_c96"]]

    set_seed(0)
    data = get_gridded_data(config, requirements, train=True)
    era5 = np.concatenate([_start_times(b, "era5") for b in data.loader])
    shield = np.concatenate([_start_times(b, "shield_c96") for b in data.loader])

    # Both groups cover the same times, but in different orders: they are
    # genuinely shuffled independently rather than index-paired.
    assert not np.array_equal(era5, shield)
    np.testing.assert_array_equal(np.sort(era5), np.sort(shield))


def test_misaligned_times_within_a_pairing_group_raise(tmp_path):
    config = TranslateDataLoaderConfig(
        streams=[
            StreamConfig(
                domain="fine", dataset=_write_stream(tmp_path / "f", (8, 16), 6)
            ),
            StreamConfig(
                domain="coarse",
                # starts one timestep later than the fine stream
                dataset=XarrayDataConfig(
                    data_path=str(_write_stream(tmp_path / "c", (4, 8), 6).data_path),
                    subset=Slice(start=1),
                ),
            ),
        ],
        batch_size=2,
    )
    with pytest.raises(ValueError, match="at sample index 0"):
        get_gridded_data(config, _requirements({"fine": NAMES, "coarse": NAMES}), True)


def test_misalignment_between_independent_streams_is_allowed(tmp_path):
    """Same two misaligned streams, but no objective consumes both."""
    config = TranslateDataLoaderConfig(
        streams=[
            StreamConfig(
                domain="fine", dataset=_write_stream(tmp_path / "f", (8, 16), 6)
            ),
            StreamConfig(
                domain="coarse",
                dataset=XarrayDataConfig(
                    data_path=str(_write_stream(tmp_path / "c", (4, 8), 6).data_path),
                    subset=Slice(start=1),
                ),
            ),
        ],
        batch_size=2,
    )
    data = get_gridded_data(
        config, _requirements({"fine": NAMES}, {"coarse": NAMES}), train=True
    )
    assert data.n_batches == 2  # the shorter (subset) group has 5 samples


def test_mismatched_timestep_within_a_pairing_group_raises(tmp_path):
    config = TranslateDataLoaderConfig(
        streams=[
            StreamConfig(
                domain="daily",
                dataset=_write_stream(tmp_path / "d", (4, 8), 8, timestep_days=1.0),
            ),
            StreamConfig(
                domain="two_daily",
                dataset=_write_stream(tmp_path / "t", (4, 8), 8, timestep_days=2.0),
            ),
        ],
        batch_size=2,
    )
    with pytest.raises(ValueError, match="must have the same timestep"):
        get_gridded_data(
            config, _requirements({"daily": NAMES, "two_daily": NAMES}), True
        )


def test_group_length_is_the_minimum_over_its_streams(tmp_path):
    """Objectives asking different window lengths leave different sample counts."""
    config = _multi_resolution_config(tmp_path, n_times=8, batch_size=1)
    requirements = TranslateDataRequirements.from_objectives(
        [
            ObjectiveDataRequirements(
                streams={
                    "era5_1deg": StreamRequirements(names=NAMES, n_timesteps=1),
                    "era5_2deg": StreamRequirements(names=NAMES, n_timesteps=1),
                    # a 4-step rollout leaves 8 - 4 + 1 = 5 start times
                    "era5_4deg": StreamRequirements(names=NAMES, n_timesteps=4),
                }
            )
        ]
    )
    data = get_gridded_data(config, requirements, train=False)
    assert data.n_batches == 5


def test_scheduled_window_growth_keeps_the_group_length(tmp_path):
    """A stream whose window grows over epochs keeps its start-time count.

    A group's length and its start-time alignment are computed once, when the
    ``PairedStreamDataset`` is built, so they are only correct if a stream's
    number of valid start times is fixed by its schedule's *longest* window
    rather than by the current epoch's.
    """
    config = _multi_resolution_config(tmp_path, n_times=8, batch_size=1)
    requirements = TranslateDataRequirements.from_objectives(
        [
            ObjectiveDataRequirements(
                streams={
                    "era5_1deg": StreamRequirements(names=NAMES, n_timesteps=1),
                    "era5_2deg": StreamRequirements(names=NAMES, n_timesteps=1),
                    "era5_4deg": StreamRequirements(
                        names=NAMES,
                        n_timesteps=IntSchedule(
                            start_value=1,
                            milestones=[IntMilestone(epoch=2, value=4)],
                        ),
                    ),
                }
            )
        ]
    )
    data = get_gridded_data(config, requirements, train=False)
    window_lengths = {}
    for epoch in [0, 1, 2, 5]:
        data.set_epoch(epoch)
        # 8 - 4 + 1, set by the longest scheduled window, at every epoch
        assert data.n_batches == 5
        assert len(list(data.loader)) == 5
        batch = next(iter(data.loader))
        window_lengths[epoch] = batch["era5_4deg"].data["temp"].shape[1]
    # the window itself does grow at the milestone; only the count is fixed
    assert window_lengths == {0: 1, 1: 1, 2: 4, 5: 4}


def test_validation_loader_is_not_shuffled(tmp_path):
    config = _multi_resolution_config(tmp_path, n_times=8, batch_size=1)
    data = get_gridded_data(
        config, _requirements({name: NAMES for name in SHAPES}), train=False
    )
    times = np.concatenate([_start_times(b, "era5_1deg") for b in data.loader])
    np.testing.assert_array_equal(times, np.sort(times))


def test_subset_loader_selects_a_batch_range(tmp_path):
    config = _multi_resolution_config(tmp_path, n_times=8, batch_size=2)
    data = get_gridded_data(
        config, _requirements({name: NAMES for name in SHAPES}), train=False
    )
    all_batches = list(data.loader)
    subset = list(data.subset_loader(start_batch=1, stop_batch=3))
    assert len(subset) == 2
    for offset, batch in enumerate(subset):
        np.testing.assert_array_equal(
            _start_times(batch, "era5_1deg"),
            _start_times(all_batches[offset + 1], "era5_1deg"),
        )


def test_set_epoch_reaches_every_stream(tmp_path):
    config = _multi_resolution_config(tmp_path, n_times=8, batch_size=2)
    data = get_gridded_data(
        config, _requirements({name: NAMES for name in SHAPES}), train=True
    )
    data.set_epoch(3)
    batch = next(iter(data.loader))
    assert batch.epoch == 3
    assert all(stream.epoch == 3 for stream in batch.streams.values())


def test_alternate_shuffle_changes_the_order(tmp_path):
    config = _multi_resolution_config(tmp_path, n_times=8, batch_size=1)
    data = get_gridded_data(
        config, _requirements({name: NAMES for name in SHAPES}), train=True
    )
    data.set_epoch(0)
    before = np.concatenate([_start_times(b, "era5_1deg") for b in data.loader])
    data.alternate_shuffle()
    after = np.concatenate([_start_times(b, "era5_1deg") for b in data.loader])
    assert not np.array_equal(before, after)


def test_dataset_info_is_keyed_by_domain(tmp_path):
    config = _multi_resolution_config(tmp_path)
    data = get_gridded_data(
        config, _requirements({name: NAMES for name in SHAPES}), train=True
    )
    assert sorted(data.dataset_info) == ["atmos_1deg", "atmos_2deg", "atmos_4deg"]
    for name, shape in SHAPES.items():
        domain = name.replace("era5_", "atmos_")
        assert data.dataset_info[domain].img_shape == shape


def test_streams_sharing_a_domain_give_one_dataset_info(tmp_path):
    config = TranslateDataLoaderConfig(
        streams=[
            StreamConfig(
                name="era5",
                domain="atmos",
                dataset=_write_stream(tmp_path / "e", (4, 8), 6),
            ),
            StreamConfig(
                name="ifs",
                domain="atmos",
                dataset=_write_stream(tmp_path / "i", (4, 8), 6),
            ),
        ],
        batch_size=2,
    )
    data = get_gridded_data(
        config, _requirements({"era5": NAMES}, {"ifs": NAMES}), train=True
    )
    assert list(data.dataset_info) == ["atmos"]
    assert data.stream_domains == {"era5": "atmos", "ifs": "atmos"}


def test_streams_sharing_a_domain_must_be_compatible(tmp_path):
    config = TranslateDataLoaderConfig(
        streams=[
            StreamConfig(
                name="era5",
                domain="atmos",
                dataset=_write_stream(tmp_path / "e", (4, 8), 6),
            ),
            StreamConfig(
                name="ifs",
                domain="atmos",
                dataset=_write_stream(tmp_path / "i", (8, 16), 6),
            ),
        ],
        batch_size=2,
    )
    with pytest.raises(ValueError, match="both serve domain 'atmos'"):
        get_gridded_data(config, _requirements({"era5": NAMES}, {"ifs": NAMES}), True)


def test_stream_no_objective_consumes_raises(tmp_path):
    config = _multi_resolution_config(tmp_path)
    with pytest.raises(ValueError, match="no objective consumes them"):
        get_gridded_data(
            config, _requirements({"era5_1deg": NAMES, "era5_2deg": NAMES}), True
        )


def test_objective_referencing_an_unconfigured_stream_raises(tmp_path):
    config = _multi_resolution_config(tmp_path)
    with pytest.raises(ValueError, match="not configured"):
        get_gridded_data(
            config,
            _requirements({**{name: NAMES for name in SHAPES}, "era5_8deg": NAMES}),
            True,
        )


def test_stream_name_defaults_to_domain():
    stream = StreamConfig(domain="atmos_1deg", dataset=XarrayDataConfig(data_path="/x"))
    assert stream.stream_name == "atmos_1deg"
    named = StreamConfig(
        domain="atmos_1deg", dataset=XarrayDataConfig(data_path="/x"), name="era5"
    )
    assert named.stream_name == "era5"


def test_duplicate_stream_names_are_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        TranslateDataLoaderConfig(
            streams=[
                StreamConfig(domain="atmos", dataset=XarrayDataConfig(data_path="/a")),
                StreamConfig(
                    domain="other",
                    dataset=XarrayDataConfig(data_path="/b"),
                    name="atmos",
                ),
            ],
            batch_size=2,
        )


def test_no_streams_is_rejected():
    with pytest.raises(ValueError, match="At least one data stream"):
        TranslateDataLoaderConfig(streams=[], batch_size=2)

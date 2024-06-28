import dataclasses
from typing import Literal, Optional, Sequence, Union

import xarray as xr

from fme.core.distributed import Distributed


@dataclasses.dataclass
class Slice:
    """
    Configuration of a python `slice` built-in.

    Required because `slice` cannot be initialized directly by dacite.

    Attributes:
        start: Start index of the slice.
        stop: Stop index of the slice.
        step: Step of the slice.
    """

    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop, self.step)


@dataclasses.dataclass
class TimeSlice:
    """
    Configuration of a slice of times. Step is an integer-valued index step.

    Note: start_time and stop_time may be provided as partial time strings and the
        stop_time will be included in the slice. See more details in `Xarray docs`_.

    Attributes:
        start_time: Start time of the slice.
        stop_time: Stop time of the slice.
        step: Step of the slice.

    .. _Xarray docs:
       https://docs.xarray.dev/en/latest/user-guide/weather-climate.html#non-standard-calendars-and-dates-outside-the-nanosecond-precision-range  # noqa
    """

    start_time: Optional[str] = None
    stop_time: Optional[str] = None
    step: Optional[int] = None

    def slice(self, times: xr.CFTimeIndex) -> slice:
        return times.slice_indexer(self.start_time, self.stop_time, self.step)


@dataclasses.dataclass
class XarrayDataConfig:
    """
    Attributes:
        data_path: Path to the data.
        file_pattern: Glob pattern to match files in the data_path.
        n_repeats: Number of times to repeat the dataset (in time). It is up
            to the user to ensure that the input dataset to repeat results in
            data that is reasonably continuous across repetitions.
        engine: Backend for xarray.open_dataset. Currently supported options
            are "netcdf4" (the default) and "h5netcdf". Only valid when using
            XarrayDataset.
        subset: Slice defining a subset of the XarrayDataset to load. This can
            either be a `Slice` of integer indices or a `TimeSlice` of timestamps.
        infer_timestep: Whether to infer the timestep from the provided data.
            This should be set to True (the default) for ACE training. It may
            be useful to toggle this to False for applications like downscaling,
            which do not depend on the timestep of the data and therefore lack
            the additional requirement that the data be ordered and evenly
            spaced in time. It must be set to True if n_repeats > 1 in order
            to be able to infer the full time coordinate.
    """

    data_path: str
    file_pattern: str = "*.nc"
    n_repeats: int = 1
    engine: Optional[Literal["netcdf4", "h5netcdf", "zarr"]] = None
    subset: Union[Slice, TimeSlice] = dataclasses.field(default_factory=Slice)
    infer_timestep: bool = True

    def __post_init__(self):
        if self.n_repeats > 1 and not self.infer_timestep:
            raise ValueError(
                "infer_timestep must be True if n_repeats is greater than 1"
            )


@dataclasses.dataclass
class DataLoaderConfig:
    """
    Attributes:
        dataset: A sequence of configurations each defining a dataset
            to be loaded. This sequence of datasets will be concatenated.
        batch_size: Number of samples per batch.
        num_data_workers: Number of parallel workers to use for data loading.
        prefetch_factor: how many batches a single data worker will attempt to
            hold in host memory at a given time.
        strict_ensemble: Whether to enforce that the ensemble members have the same
            dimensions and coordinates.
    """

    dataset: Sequence[XarrayDataConfig]
    batch_size: int
    num_data_workers: int
    prefetch_factor: Optional[int] = None
    strict_ensemble: bool = True

    def __post_init__(self):
        dist = Distributed.get_instance()
        if self.batch_size % dist.world_size != 0:
            raise ValueError(
                "batch_size must be divisible by the number of parallel "
                f"workers, got {self.batch_size} and {dist.world_size}"
            )

import dataclasses
from typing import Literal, Optional, Union

import xarray as xr

from fme.core.distributed import Distributed


@dataclasses.dataclass
class Slice:
    """
    Configuration of a python `slice` built-in.

    Required because `slice` cannot be initialized directly by dacite.
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
        n_repeats: Number of times to repeat the dataset (in time).
        engine: Backend for xarray.open_dataset. Currently supported options
            are "netcdf4" (the default) and "h5netcdf". Only valid when using
            XarrayDataset.
    """

    data_path: str
    file_pattern: str = "*.nc"
    n_repeats: int = 1
    engine: Optional[Literal["netcdf4", "h5netcdf"]] = None


@dataclasses.dataclass
class DataLoaderConfig:
    """
    Attributes:
        dataset: Parameters to define the dataset.
        batch_size: Number of samples per batch.
        num_data_workers: Number of parallel workers to use for data loading.
        data_type: Type of data to load.
        subset: Slice defining a subset of the XarrayDataset to load. For
            data_type="ensemble_xarray" case this will be applied to each ensemble
            member before concatenation.
    """

    dataset: XarrayDataConfig
    batch_size: int
    num_data_workers: int
    data_type: Literal["xarray", "ensemble_xarray"]
    subset: Union[Slice, TimeSlice] = dataclasses.field(default_factory=Slice)
    n_samples: Optional[int] = None
    strict_ensemble: bool = True

    def __post_init__(self):
        if self.n_samples is not None:
            raise ValueError(
                "n_samples is not a valid parameter for DataLoaderParams, "
                "use subset.stop instead"
            )
        dist = Distributed.get_instance()
        if self.batch_size % dist.world_size != 0:
            raise ValueError(
                "batch_size must be divisible by the number of parallel "
                f"workers, got {self.batch_size} and {dist.world_size}"
            )

        if self.dataset.n_repeats != 1 and self.data_type == "ensemble_xarray":
            raise ValueError("n_repeats must be 1 when using ensemble_xarray")

import dataclasses
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd
import torch
import xarray as xr

from fme.core.typing_ import Slice, TensorDict


@dataclasses.dataclass
class TimeSlice:
    """
    Configuration of a slice of times. Step is an integer-valued index step.

    Note: start_time and stop_time may be provided as partial time strings and the
        stop_time will be included in the slice. See more details in `Xarray docs`_.

    Parameters:
        start_time: Start time of the slice.
        stop_time: Stop time of the slice.
        step: Step of the slice.

    .. _Xarray docs:
       https://docs.xarray.dev/en/latest/user-guide/weather-climate.html#non-standard-calendars-and-dates-outside-the-nanosecond-precision-range
    """  # noqa: E501

    start_time: str | None = None
    stop_time: str | None = None
    step: int | None = None

    def slice(self, time: xr.CFTimeIndex) -> slice:
        return time.slice_indexer(self.start_time, self.stop_time, self.step)


def _convert_interval_to_int(
    interval: pd.Timedelta,
    timestep: timedelta,
):
    """Convert interval to integer number of timesteps."""
    if interval % timestep != timedelta(0):
        raise ValueError(
            f"Requested interval length {interval} is not a "
            f"multiple of the timestep {timestep}."
        )

    return interval // timestep


@dataclasses.dataclass
class RepeatedInterval:
    """
    Configuration for a repeated interval within a block.  This configuration
    is used to generate a boolean mask for a dataset that will return values
    within the interval and repeat that throughout the dataset.

    Parameters:
        interval_length: Length of the interval to return values from
        start: Start position of the interval within the repeat block.
        block_length: Total length of the block to be repeated over the length of
            the dataset, including the interval length.

    Note:
        The interval_length, start, and block_length can be provided as either
        all integers or all strings representing timedeltas of the block.
        If provided as strings, the timestep must be provided when calling
        `get_boolean_mask`.

    Examples:
        To return values from the first 3 items of every 6 items, use:

        >>> fme.ace.RepeatedInterval(interval_length=3, block_length=6, start=0)  # doctest: +IGNORE_OUTPUT

        To return a days worth of values starting after 2 days from every 7-day
        block, use:

        >>> fme.ace.RepeatedInterval(interval_length="1d", block_length="7d", start="2d")  # doctest: +IGNORE_OUTPUT
    """  # noqa: E501

    interval_length: int | str
    start: int | str
    block_length: int | str

    def __post_init__(self):
        types = {type(self.interval_length), type(self.block_length), type(self.start)}
        if len(types) > 1:
            raise ValueError(
                "All attributes of RepeatedInterval must be of the "
                "same type (either all int or all str)."
            )

        self._is_time_delta_str = isinstance(self.interval_length, str)

        if self._is_time_delta_str:
            self.interval_length = pd.Timedelta(self.interval_length)
            self.block_length = pd.Timedelta(self.block_length)
            self.start = pd.Timedelta(self.start)

    def get_boolean_mask(
        self, length: int, timestep: timedelta | None = None
    ) -> np.ndarray:
        """
        Return a boolean mask for the repeated interval.

        Args:
            length: Length of the dataset.
            timestep: Timestep of the dataset.
        """
        if self._is_time_delta_str:
            if timestep is None:
                raise ValueError(
                    "Timestep must be provided when using time deltas "
                    "for RepeatedInterval."
                )

            interval_length = _convert_interval_to_int(self.interval_length, timestep)
            block_length = _convert_interval_to_int(self.block_length, timestep)
            start = _convert_interval_to_int(self.start, timestep)
        else:
            interval_length = self.interval_length
            block_length = self.block_length
            start = self.start

        if start + interval_length > block_length:
            raise ValueError(
                "The interval (with start point) must fit within the repeat block."
            )

        block = np.zeros(block_length, dtype=bool)
        block[start : start + interval_length] = True
        num_blocks = length // block_length + 1
        mask = np.tile(block, num_blocks)[:length]
        return mask


@dataclasses.dataclass
class OverwriteConfig:
    """Configuration to overwrite field values in XarrayDataset.

    Parameters:
        constant: Fill field with constant value.
        multiply_scalar: Multiply field by scalar value.
    """

    constant: Mapping[str, float] = dataclasses.field(default_factory=dict)
    multiply_scalar: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        key_overlap = set(self.constant.keys()) & set(self.multiply_scalar.keys())
        if key_overlap:
            raise ValueError(
                "OverwriteConfig cannot have the same variable in both constant "
                f"and multiply_scalar: {key_overlap}"
            )

    def apply(self, tensors: TensorDict) -> TensorDict:
        for var, fill_value in self.constant.items():
            data = tensors[var]
            tensors[var] = torch.ones_like(data) * torch.tensor(
                fill_value, dtype=data.dtype, device=data.device
            )
        for var, multiplier in self.multiply_scalar.items():
            data = tensors[var]
            tensors[var] = data * torch.tensor(
                multiplier, dtype=data.dtype, device=data.device
            )
        return tensors

    @property
    def variables(self):
        return set(self.constant.keys()) | set(self.multiply_scalar.keys())


@dataclasses.dataclass
class FillNaNsConfig:
    """
    Configuration to fill NaNs with a constant value or others.

    Parameters:
        method: Type of fill operation. Currently only 'constant' is supported.
        value: Value to fill NaNs with.
    """

    method: Literal["constant"] = "constant"
    value: float = 0.0


@dataclasses.dataclass
class XarrayDataConfig:
    """
    Parameters:
        data_path: Path to the data.
        file_pattern: Glob pattern to match files in the data_path.
        n_repeats: Number of times to repeat the dataset (in time). It is up
            to the user to ensure that the input dataset to repeat results in
            data that is reasonably continuous across repetitions.
        engine: Backend used in xarray.open_dataset call.
        spatial_dimensions: Specifies the spatial dimensions for the grid, default
            is lat/lon. If 'latlon', it is assumed that the last two dimensions are
            latitude and longitude, respectively. If 'healpix', it is assumed that the
            last three dimensions are face, height, and width, respectively.
        subset: Slice defining a subset of the XarrayDataset to load. This can
            either be a `Slice` of integer indices or a `TimeSlice` of timestamps.
            This feature is applied directly to the dataset samples. For example,
            if the file(s) have the time coordinate (t0, t1, t2, t3) and
            requirements.n_timesteps=2, then subset=Slice(stop=2) will
            provide two samples: (t0, t1), (t1, t2).
        infer_timestep: Whether to infer the timestep from the provided data.
            This should be set to True (the default) for ACE training. It may
            be useful to toggle this to False for applications like downscaling,
            which do not depend on the timestep of the data and therefore lack
            the additional requirement that the data be ordered and evenly
            spaced in time. It must be set to True if n_repeats > 1 in order
            to be able to infer the full time coordinate.
        dtype: Data type to cast the data to. If None, no casting is done. It is
            required that 'torch.{dtype}' is a valid dtype.
        overwrite: Optional OverwriteConfig to overwrite loaded field values.
        fill_nans: Optional FillNaNsConfig to fill NaNs with a constant value.
        isel: Optional xarray isel arguments to be passed to the dataset. Will
            raise ValueError if time is included here, since the subset argument
            is used specifically for selecting times. Horizontal dimensions are
            also not currently supported.

    Examples:
        If data is stored in a directory with multiple netCDF files which can be
        concatenated along the time dimension, use:

        >>> fme.ace.XarrayDataConfig(data_path="/some/directory", file_pattern="*.nc") # doctest: +IGNORE_OUTPUT

        If data is stored in a single zarr store at ``/some/directory/dataset.zarr``,
        use:

        >>> fme.ace.XarrayDataConfig(
        ...     data_path="/some/directory",
        ...     file_pattern="dataset.zarr",
        ...     engine="zarr"
        ... ) # doctest: +IGNORE_OUTPUT
    """  # noqa: E501

    data_path: str
    file_pattern: str = "*.nc"
    n_repeats: int = 1
    engine: Literal["netcdf4", "h5netcdf", "zarr"] = "netcdf4"
    spatial_dimensions: Literal["healpix", "latlon"] = "latlon"
    subset: Slice | TimeSlice | RepeatedInterval = dataclasses.field(
        default_factory=Slice
    )
    infer_timestep: bool = True
    dtype: str | None = "float32"
    overwrite: OverwriteConfig = dataclasses.field(default_factory=OverwriteConfig)
    fill_nans: FillNaNsConfig | None = None
    isel: Mapping[str, Slice | int] = dataclasses.field(default_factory=dict)

    def _default_file_pattern_check(self):
        if self.engine == "zarr" and self.file_pattern == "*.nc":
            raise ValueError(
                "The file pattern is set to the default NetCDF file pattern *.nc "
                "but the engine is specified as 'zarr'. Please set "
                "`XarrayDataConfig.file_pattern` to match the zarr filename."
            )

    @property
    def torch_dtype(self) -> torch.dtype | None:
        if self.dtype is None:
            return None
        else:
            try:
                torch_dtype = getattr(torch, self.dtype)
            except AttributeError:
                raise ValueError(f"Invalid dtype '{self.dtype}'")
            if not isinstance(torch_dtype, torch.dtype):
                raise ValueError(f"Invalid dtype '{self.dtype}'")
        return torch_dtype

    def __post_init__(self):
        if self.n_repeats > 1 and not self.infer_timestep:
            raise ValueError(
                "infer_timestep must be True if n_repeats is greater than 1"
            )
        if self.spatial_dimensions not in ["latlon", "healpix"]:
            raise ValueError(
                f"unexpected spatial_dimensions {self.spatial_dimensions},"
                " should be one of 'latlon' or 'healpix'"
            )
        self.torch_dtype  # check it can be retrieved
        self._default_file_pattern_check()
        self.zarr_engine_used = False
        if self.engine == "zarr":
            self.zarr_engine_used = True


@dataclasses.dataclass
class ConcatDatasetConfig:
    """
    Configuration for concatenating multiple datasets.
    Parameters:
        concat: List of XarrayDataConfig objects to concatenate.
        strict: Whether to enforce that the datasets to be concatenated
            have the same dimensions and coordinates.
    """

    concat: Sequence[XarrayDataConfig]
    strict: bool = True

    def __post_init__(self):
        self.zarr_engine_used = any(ds.engine == "zarr" for ds in self.concat)


@dataclasses.dataclass
class MergeDatasetConfig:
    """
    Configuration for merging multiple datasets.
    Parameters:
        merge: List of ConcatDatasetConfig or XarrayDataConfig to merge.
    """

    merge: Sequence[ConcatDatasetConfig | XarrayDataConfig]

    def __post_init__(self):
        self.zarr_engine_used = False
        for ds in self.merge:
            if isinstance(ds, ConcatDatasetConfig):
                if ds.zarr_engine_used:
                    self.zarr_engine_used = ds.zarr_engine_used
                    break
            elif isinstance(ds, XarrayDataConfig):
                if ds.engine == "zarr":
                    self.zarr_engine_used = True
                    break


@dataclasses.dataclass
class MergeNoConcatDatasetConfig:
    """
    Configuration for merging multiple datasets. No concatenation is allowed.
    Parameters:
        merge: List of XarrayDataConfig to merge.
    """

    merge: Sequence[XarrayDataConfig]

    def __post_init__(self):
        self.zarr_engine_used = False
        for ds in self.merge:
            if ds.engine == "zarr":
                self.zarr_engine_used = True
                break

import dataclasses
import datetime
import logging
import os
from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypeGuard, Union

import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.time import TimeSlice
from fme.core.typing_ import Slice

from .dataset_metadata import DatasetMetadata
from .monthly import MonthlyDataWriter
from .raw import NetCDFWriterConfig, RawDataWriter
from .time_coarsen import (
    MonthlyCoarsenConfig,
    PairedTimeCoarsen,
    TimeCoarsen,
    TimeCoarsenConfig,
)
from .utils import DIM_INFO_HEALPIX, DIM_INFO_LATLON
from .zarr import (
    SeparateICZarrWriterAdapter,
    ZarrWriterAdapter,
    ZarrWriterConfig,
    ensure_numpy_coords,
)

logger = logging.getLogger(__name__)

LAT_NAME = DIM_INFO_LATLON[0].name
LON_NAME = DIM_INFO_LATLON[1].name

DatetimeDataArray: TypeAlias = xr.DataArray


def _is_datetime_dataarray(data: xr.DataArray) -> TypeGuard[DatetimeDataArray]:
    """
    Check if the DataArray is a datetime data array with a 'dt' accessor.
    """
    return isinstance(data, xr.DataArray) and hasattr(data, "dt")


def _month_string_to_int(month_str: str) -> int:
    try:
        month = datetime.datetime.strptime(month_str, "%B")
        return month.month
    except ValueError:
        pass

    try:
        month = datetime.datetime.strptime(month_str, "%b")
        return month.month
    except ValueError:
        pass

    raise ValueError(
        f"Invalid month string: {month_str}. Use full month name "
        "(e.g., 'January') or abbreviated name (e.g., 'Jan')."
    )


def _get_time_mask(time: DatetimeDataArray, selections: Sequence[str]) -> xr.DataArray:
    """
    Build a boolean mask for the given time array based on the specified
    month or season selections.
    """
    if not _is_datetime_dataarray(time):
        raise ValueError("Input does not contain datetime data with 'dt' accessor.")

    seasons = ("DJF", "MAM", "JJA", "SON")
    mask = None
    for selection in selections:
        if selection in seasons:
            current_mask = time.dt.season == selection
        else:
            month_number = _month_string_to_int(selection)
            current_mask = time.dt.month == month_number

        if mask is None:
            mask = current_mask
        else:
            mask |= current_mask

    if mask is None:
        raise ValueError("Cannot build a mask for empty month selection.")

    return mask


@dataclasses.dataclass
class MonthSelector:
    """
    Specifies a selection of months for filtering data. Months can be specified
    using full names, three-letter abbreviated names, or season names (e.g., "DJF").

    Example:
        ```
        selector = MonthSelector(months=["January", "Feb", "MAM"])
        selected_data = selector.select(data)
        ```
    """

    months: list[str]

    def select(self, data: xr.Dataset) -> xr.Dataset:
        """
        Select data for the specified months or seasons.
        """
        if not self.months:
            return data

        time_mask = _get_time_mask(data.time, self.months)
        return data.isel(time=time_mask)


def _select_time(
    data: xr.Dataset,
    time_selection: TimeSlice | MonthSelector | Slice | None,
    start_timestep: int = 0,
    sample_dim: str = "sample",
    time_dim: str = "time",
) -> xr.Dataset:
    """
    Filter the dataset based on the time selection.
    """
    if time_selection is None:
        return data

    if time_dim not in data.coords:
        raise ValueError(
            f"Dataset must contain a '{time_dim}' coordinate for time selection."
        )

    if sample_dim not in data.dims:
        raise ValueError(
            f"Dataset must contain a '{sample_dim}' dimension for time selection."
        )

    def _time_subselector(sample_ds: xr.Dataset) -> xr.Dataset:
        if isinstance(time_selection, TimeSlice):
            ds_subselected = sample_ds.sel(**{time_dim: time_selection.as_raw_slice()})
        elif isinstance(time_selection, MonthSelector):
            ds_subselected = time_selection.select(sample_ds)
        elif isinstance(time_selection, Slice):
            sl = Slice.shift_left(time_selection, start_timestep)
            ds_subselected = sample_ds.isel(**{time_dim: sl.slice})
        else:
            raise ValueError(f"Unsupported time selection type: {type(time_selection)}")
        return ds_subselected

    data_arrays, time_arrays = [], []
    for i_sample in range(len(data[sample_dim])):
        sample_ds = data.isel({sample_dim: i_sample})
        sample_ds_subselected = _time_subselector(
            sample_ds.assign_coords({time_dim: sample_ds[time_dim]})
        )
        data_arrays.append(
            sample_ds_subselected.drop_vars(time_dim).expand_dims({sample_dim: 1})
        )
        time_arrays.append(
            sample_ds_subselected[time_dim]
            .drop_vars(time_dim)
            .expand_dims({sample_dim: 1})
        )
    combined_subsampled_data = xr.concat(data_arrays, dim=sample_dim)
    combined_time = xr.concat(time_arrays, dim=sample_dim)
    combined_data = combined_subsampled_data.assign({time_dim: combined_time})
    return combined_data


@dataclasses.dataclass
class FileWriterConfig:
    """
    Configuration for writing output data.

    Parameters:
        label: A label used for the filename output for this output dataset.
        names: The names of the variables to save. If not specified, all available
            variables will be saved.
        lat_extent: The latitude extent of the region as (min_lat, max_lat). If not set,
            all latitudes are included.
        lon_extent: The longitude extent of the region as (min_lon, max_lon). If not
            set, all longitudes are included.
        time_selection: Optional time selection criteria. Can be an Slice,
            MonthSelector, or TimeSlice. If None, all times are selected. A Slice
            can select an index range of steps in an inference, the MonthSelector can be
            used to target specific seasons or months for outputs, and a TimeSlice
            allows for datetime range selection.
        save_reference: Whether to save the reference/target data alongside predictions.
            If true, "_target" will be appended to the label for the target data, and
            "_predictions" will be appended to the label for the predictions data.
            Ignored if building a single writer via the `build` method.
        time_coarsen: Configuration for time averaging of outputs.
        format: Configuration for the output format (i.e. netCDF or zarr).
        separate_ensemble_members: Option to write ensemble members to separate files.
            In this case, time is a datetime coordinate. Only supported when using zarr
            format. Filenames will have the suffix `_ic{member_index}` appended before
            the file extension.

    """

    label: str
    names: list[str] | None = None
    lat_extent: Sequence[float] | None = None
    lon_extent: Sequence[float] | None = None
    time_selection: Slice | MonthSelector | TimeSlice | None = None
    save_reference: bool = True
    time_coarsen: TimeCoarsenConfig | MonthlyCoarsenConfig | None = None
    format: NetCDFWriterConfig | ZarrWriterConfig = dataclasses.field(
        default_factory=NetCDFWriterConfig
    )
    separate_ensemble_members: bool = False

    def __post_init__(self):
        if self.lat_extent:
            if len(self.lat_extent) != 2:
                raise ValueError("lat_extent must be a tuple of (min_lat, max_lat)")
            self.lat_slice = slice(*self.lat_extent)
        else:
            self.lat_slice = slice(None)

        if self.lon_extent:
            if len(self.lon_extent) != 2:
                raise ValueError("lon_extent must be a tuple of (min_lon, max_lon)")
            self.lon_slice = slice(*self.lon_extent)
        else:
            self.lon_slice = slice(None)

        if self.time_selection is not None:
            if self.time_coarsen is not None:
                logging.warning(
                    "Time coarsening is enabled. "
                    "Time subselection is applied *after* time coarsening."
                )

        if isinstance(self.format, ZarrWriterConfig):
            if self.time_selection is not None:
                raise NotImplementedError(
                    "Time selection is not currently supported when writing to zarr."
                )
            if isinstance(self.time_coarsen, MonthlyCoarsenConfig):
                raise NotImplementedError(
                    "Monthly coarsening is not currently supported for the zarr format."
                )

        if isinstance(self.time_coarsen, MonthlyCoarsenConfig):
            if self.time_selection is not None:
                raise NotImplementedError(
                    "Time selection is not currently supported when using monthly "
                    "coarsening."
                )

    @property
    def filenames(self) -> list[str]:
        base_filenames = [
            self.label,
            f"{self.label}_target",
            f"{self.label}_predictions",
        ]
        return [
            ".".join([base_filename, self.format.suffix])
            for base_filename in base_filenames
        ]

    def build_paired(
        self,
        experiment_dir: str,
        n_initial_conditions: int,
        n_timesteps: int,
        timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
        prediction_suffix: str = "predictions",
        reference_suffix: str = "target",
    ) -> Union["PairedFileWriter", PairedTimeCoarsen]:
        if self.save_reference:
            reference_label = f"{self.label}_{reference_suffix}"
            prediction_label = f"{self.label}_{prediction_suffix}"
            reference_writer = dataclasses.replace(self, label=reference_label).build(
                experiment_dir=experiment_dir,
                n_initial_conditions=n_initial_conditions,
                n_timesteps=n_timesteps,
                timestep=timestep,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=dataset_metadata,
            )
        else:
            prediction_label = self.label
            reference_writer = None
        prediction_writer = dataclasses.replace(self, label=prediction_label).build(
            experiment_dir=experiment_dir,
            n_initial_conditions=n_initial_conditions,
            n_timesteps=n_timesteps,
            timestep=timestep,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
        )
        paired_writer = PairedFileWriter(prediction_writer, reference_writer)
        # Time coarsening is built around writer in the single build method
        return paired_writer

    def build(
        self,
        experiment_dir: str,
        n_initial_conditions: int,
        n_timesteps: int,
        timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ) -> Union["FileWriter", TimeCoarsen]:
        """
        Build a FileWriter object for saving data within the specified region.

        Args:
            experiment_dir: The directory where experiment outputs are saved.
            n_initial_conditions: The number of initial conditions or ensemble members.
            n_timesteps: Total number of inference forward steps.
            timestep: The time delta between each timestep.
            variable_metadata: Metadata for each variable.
            coords: Coordinate arrays for the dataset. These should be the coordinates
                of the entire global domain, not the subset region coordinates.
            dataset_metadata: Metadata for the entire dataset.
        """
        if "face" in coords:
            spatial_dims = DIM_INFO_HEALPIX
        else:
            spatial_dims = DIM_INFO_LATLON

        if (self.lat_extent and LAT_NAME not in coords) or (
            self.lon_extent and LON_NAME not in coords
        ):
            raise ValueError(
                "Coordinates must include 'lat' and 'lon' if using lat/lon extents. "
                f"Got {list(coords.keys())}."
            )

        if isinstance(self.time_selection, TimeSlice) and n_initial_conditions > 1:
            raise NotImplementedError(
                "TimeSlice selection is not currently supported for multiple "
                "initial conditions."
            )
        elif (
            isinstance(self.time_selection, MonthSelector) and n_initial_conditions > 1
        ):
            raise NotImplementedError(
                "MonthSelector selection is not currently supported for multiple "
                "initial conditions."
            )

        subset_coords = xr.Dataset(coords)
        if self.lat_extent or self.lon_extent:
            subset_coords = subset_coords.sel(
                {LAT_NAME: self.lat_slice, LON_NAME: self.lon_slice}
            )
        subselect_coords_ = {str(k): v for k, v in subset_coords.coords.items()}

        raw_writer: (
            RawDataWriter
            | ZarrWriterAdapter
            | SeparateICZarrWriterAdapter
            | MonthlyDataWriter
        )
        if isinstance(self.format, ZarrWriterConfig):
            if isinstance(self.time_coarsen, TimeCoarsenConfig):
                n_timesteps_write = n_timesteps // self.time_coarsen.coarsen_factor
            else:
                n_timesteps_write = n_timesteps

            zarr_writer_cls: type[SeparateICZarrWriterAdapter | ZarrWriterAdapter]

            if self.separate_ensemble_members:
                dims = ("time", *(d.name for d in spatial_dims))
                zarr_writer_cls = SeparateICZarrWriterAdapter
            else:
                dims = ("sample", "time", *(d.name for d in spatial_dims))
                zarr_writer_cls = ZarrWriterAdapter
            raw_writer = zarr_writer_cls(
                path=os.path.join(experiment_dir, f"{self.label}.zarr"),
                dims=dims,
                data_coords=ensure_numpy_coords(subselect_coords_),
                n_timesteps=n_timesteps_write,
                n_initial_conditions=n_initial_conditions,
                data_vars=self.names,
                variable_metadata=variable_metadata,
                dataset_metadata=dataset_metadata,
                chunks=self.format.chunks,
                overwrite_check=self.format.overwrite_check,
            )
        else:
            if self.separate_ensemble_members:
                raise NotImplementedError(
                    "Writing separate ensemble members is not currently supported for "
                    "netcdf output."
                )
            if isinstance(self.time_coarsen, MonthlyCoarsenConfig):
                raw_writer = MonthlyDataWriter(
                    path=experiment_dir,
                    label=self.label,
                    n_samples=n_initial_conditions,
                    save_names=self.names,
                    variable_metadata=variable_metadata,
                    coords=subselect_coords_,
                    dataset_metadata=dataset_metadata,
                )
            else:
                raw_writer = RawDataWriter(
                    path=experiment_dir,
                    label=self.label,
                    n_initial_conditions=n_initial_conditions,
                    save_names=self.names,
                    variable_metadata=variable_metadata,
                    coords=subselect_coords_,
                    dataset_metadata=dataset_metadata,
                )
        writer = FileWriter(self, raw_writer, full_coords=coords)
        if isinstance(self.time_coarsen, TimeCoarsenConfig):
            return self.time_coarsen.build(writer)
        else:
            return writer


class FileWriter:
    """
    A data writer for saving outputs from ACE inference.
    """

    def __init__(
        self,
        config: FileWriterConfig,
        writer: RawDataWriter
        | MonthlyDataWriter
        | ZarrWriterAdapter
        | SeparateICZarrWriterAdapter,
        full_coords: Mapping[str, np.ndarray],
    ):
        self.config = config
        self.writer = writer
        self.full_coords = full_coords
        self._no_write_count = 0
        self._n_timesteps_seen = 0
        if "face" in full_coords:
            self._spatial_dims = DIM_INFO_HEALPIX
        else:
            self._spatial_dims = DIM_INFO_LATLON

    def _subselect_data(
        self,
        data: dict[str, torch.Tensor],
        batch_time: xr.DataArray,
        start_timestep: int = 0,
        sample_dim: str = "sample",
        time_dim: str = "time",
    ) -> tuple[dict[str, torch.Tensor], xr.DataArray]:
        use_names = self.config.names or data.keys()
        data_xr = xr.Dataset(
            {
                k: xr.DataArray(
                    v.cpu().numpy(),
                    dims=[
                        sample_dim,
                        time_dim,
                        *[d.name for d in self._spatial_dims],
                    ],
                )
                for k, v in data.items()
                if k in use_names
            },
            coords={time_dim: batch_time, **self.full_coords},
        )

        if self.config.lat_extent or self.config.lon_extent:
            # TODO: should eventually support selection straddling dateline
            data_xr = data_xr.sel(
                {
                    self._spatial_dims[0].name: self.config.lat_slice,
                    self._spatial_dims[1].name: self.config.lon_slice,
                }
            )

        data_xr = _select_time(
            data_xr,
            self.config.time_selection,
            start_timestep=start_timestep,
            sample_dim=sample_dim,
            time_dim=time_dim,
        )
        subselected_data = {
            str(k): torch.from_numpy(v.values)
            for k, v in data_xr.items()
            if v.sizes["time"] > 0
        }
        return subselected_data, data_xr["time"]

    def append_batch(
        self,
        data: dict[str, torch.Tensor],
        batch_time: xr.DataArray,
    ):
        """
        Filter region and times and append a batch of data to the writer.
        """
        start_timestep = self._n_timesteps_seen
        subselected_data, subselected_time = self._subselect_data(
            data, batch_time, start_timestep=start_timestep
        )
        self._n_timesteps_seen += batch_time.sizes.get("time", 0)

        # Warn on empty batch, but it might be expected in some cases
        # so ignore after 10 warnings
        if not subselected_data:
            self._no_write_count += 1
            if self._no_write_count < 10:
                logging.warning(
                    f"No data to write for region {self.config.label} at "
                    f"timestep {start_timestep}."
                )
            elif self._no_write_count == 10:
                logging.warning("Further warnings about empty data will be suppressed.")
            return
        self.writer.append_batch(
            data=subselected_data,
            batch_time=subselected_time,
        )

    def flush(self):
        """
        Flush the writer to ensure all data is written.
        """
        self.writer.flush()

    def finalize(self):
        self.writer.finalize()


class PairedFileWriter:
    def __init__(
        self,
        prediction_writer: FileWriter | TimeCoarsen,
        reference_writer: FileWriter | TimeCoarsen | None,
    ):
        self.prediction_writer = prediction_writer
        self.reference_writer = reference_writer

    def append_batch(
        self,
        target: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
        batch_time: xr.DataArray,
    ):
        self.prediction_writer.append_batch(
            data=prediction,
            batch_time=batch_time,
        )
        if self.reference_writer:
            self.reference_writer.append_batch(
                data=target,
                batch_time=batch_time,
            )

    def flush(self):
        self.prediction_writer.flush()
        if self.reference_writer:
            self.reference_writer.flush()

    def finalize(self):
        self.prediction_writer.finalize()
        if self.reference_writer:
            self.reference_writer.finalize()

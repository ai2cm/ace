import dataclasses
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TypeAlias, TypeGuard

import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.time import TimeSlice
from fme.core.typing_ import Slice

from .dataset_metadata import DatasetMetadata
from .raw import RawDataWriter

logger = logging.getLogger(__name__)


DatetimeDataArray: TypeAlias = xr.DataArray


def _is_datetime_dataarray(data: xr.DataArray) -> TypeGuard[DatetimeDataArray]:
    """
    Check if the DataArray is a datetime data array with a 'dt' accessor.
    """
    return isinstance(data, xr.DataArray) and hasattr(data, "dt")


def _month_string_to_int(month_str: str) -> int:
    try:
        month = datetime.strptime(month_str, "%B")
        return month.month
    except ValueError:
        pass

    try:
        month = datetime.strptime(month_str, "%b")
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
) -> xr.Dataset:
    """
    Filter the dataset based on the time selection.
    """
    if time_selection is None:
        return data

    if "time" not in data.coords:
        raise ValueError("Dataset must contain a 'time' coordinate for time selection.")

    if isinstance(time_selection, TimeSlice):
        return data.sel(time=time_selection.as_raw_slice())

    if isinstance(time_selection, MonthSelector):
        return time_selection.select(data)

    if isinstance(time_selection, Slice):
        sl = Slice.shift_left(time_selection, start_timestep)
        return data.isel(time=sl.slice)

    raise ValueError(f"Unsupported time selection type: {type(time_selection)}")


@dataclasses.dataclass
class SubselectWriterConfig:
    """
    Configuration for a subselection of data to write to file.

    Parameters:
        label: A label used for the filename output for this region.
        names: The names of the variables to save. If not speciefied, all variables
            in the dataset will be saved.
        lat_extent: The latitude extent of the region as (min_lat, max_lat).
        lon_extent: The longitude extent of the region as (min_lon, max_lon).
        time_selection: Optional time selection criteria. Can be an Slice,
            MonthSelector, or TimeSlice. If None, all times are selected. A Slice
            can select an index range of steps in an inference, the MonthSelector can be
            used to target specific seasons or months for outputs, and a TimeSlice
            allows for datetime range selection.
        latitude_name: The name of the latitude coordinate in the dataset.
        longitude_name: The name of the longitude coordinate in the dataset.
    """

    label: str
    names: list[str] | None = None
    lat_extent: Sequence[float] | None = None
    lon_extent: Sequence[float] | None = None
    time_selection: Slice | MonthSelector | TimeSlice | None = None
    latitude_name: str = "latitude"
    longitude_name: str = "longitude"
    save_reference: bool = False

    def __post_init__(self):
        if not self.lon_extent and not self.lat_extent and not self.time_selection:
            raise ValueError("No subselection details specified in the SubselectWriter")

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

    def build_paired(
        self,
        experiment_dir: str,
        n_initial_conditions: int,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
        prediction_suffix: str = "prediction",
        reference_suffix: str = "reference",
    ) -> "PairedSubselectWriter":
        prediction_writer = dataclasses.replace(
            self, label=f"{self.label}_{prediction_suffix}"
        ).build(
            experiment_dir=experiment_dir,
            n_initial_conditions=n_initial_conditions,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
        )
        if self.save_reference:
            reference_writer = dataclasses.replace(
                self, label=f"{self.label}_{reference_suffix}"
            ).build(
                experiment_dir=experiment_dir,
                n_initial_conditions=n_initial_conditions,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=dataset_metadata,
            )
        else:
            reference_writer = None
        return PairedSubselectWriter(prediction_writer, reference_writer)

    def build(
        self,
        experiment_dir: str,
        n_initial_conditions: int,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ) -> "SubselectWriter":
        """
        Build a SubselectWriter object for saving data within the specified region.

        Args:
            experiment_dir: The directory where experiment outputs are saved.
            n_initial_conditions: The number of initial conditions or ensemble members.
            variable_metadata: Metadata for each variable.
            coords: Coordinate arrays for the dataset. These should be the coordinates
                of the entire global domain, not the subset region coordinates.
            dataset_metadata: Metadata for the entire dataset.
        """
        if "face" in coords:
            raise NotImplementedError(
                "SubselectWriter does not yet support writing HEALPix coordinates."
            )

        if (self.lat_extent and self.latitude_name not in coords) or (
            self.lon_extent and self.longitude_name not in coords
        ):
            raise ValueError(
                f"Coordinates must include {self.latitude_name} and "
                f"{self.longitude_name}."
            )

        subset_coords = xr.Dataset(coords)
        subset_coords = subset_coords.sel(
            {
                self.latitude_name: self.lat_slice,
                self.longitude_name: self.lon_slice,
            }
        )
        subselect_coords_ = {str(k): v for k, v in subset_coords.coords.items()}

        raw_writer = RawDataWriter(
            path=experiment_dir,
            label=f"{self.label}.nc",
            n_initial_conditions=n_initial_conditions,
            save_names=self.names,
            variable_metadata=variable_metadata,
            coords=subselect_coords_,
            dataset_metadata=dataset_metadata,
        )
        return SubselectWriter(self, raw_writer, full_coords=coords)


class SubselectWriter:
    """
    A data writer for outputting subselected ACE data for downscaling.
    """

    def __init__(
        self,
        config: SubselectWriterConfig,
        writer: RawDataWriter,
        full_coords: Mapping[str, np.ndarray],
    ):
        self.config = config
        self.writer = writer
        self.full_coords = full_coords
        self._no_write_count = 0

    def _subselect_data(
        self,
        data: dict[str, torch.Tensor],
        batch_time: xr.DataArray,
        start_timestep: int = 0,
    ) -> dict[str, torch.Tensor]:
        use_names = self.config.names or data.keys()
        data_xr = xr.Dataset(
            {
                k: xr.DataArray(
                    v.cpu().numpy(),
                    dims=[
                        "batch",
                        "time",
                        self.config.latitude_name,
                        self.config.longitude_name,
                    ],
                )
                for k, v in data.items()
                if k in use_names
            },
            coords={"time": batch_time, **self.full_coords},
        )

        # TODO: should eventually support selection straddling dateline
        data_xr = data_xr.sel(
            {
                self.config.latitude_name: self.config.lat_slice,
                self.config.longitude_name: self.config.lon_slice,
            }
        )

        data_xr = _select_time(
            data_xr, self.config.time_selection, start_timestep=start_timestep
        )
        return {
            str(k): torch.from_numpy(v.values)
            for k, v in data_xr.items()
            if v.sizes["time"] > 0
        }

    def append_batch(
        self,
        data: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
    ):
        """
        Filter region and times and append a batch of data to the writer.
        """
        subselected = self._subselect_data(data, batch_time)

        # Warn on empty batch, but it might be expected in some cases
        # so ignore after 10 warnings
        if not subselected:
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
            data=subselected,
            start_timestep=start_timestep,
            batch_time=batch_time,
        )

    def flush(self):
        """
        Flush the writer to ensure all data is written.
        """
        self.writer.flush()


class PairedSubselectWriter:
    def __init__(
        self,
        prediction_writer: SubselectWriter,
        reference_writer: SubselectWriter | None,
    ):
        self.prediction_writer = prediction_writer
        self.reference_writer = reference_writer

    def append_batch(
        self,
        target: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
    ):
        self.prediction_writer.append_batch(
            data=prediction,
            start_timestep=start_timestep,
            batch_time=batch_time,
        )
        if self.reference_writer:
            self.reference_writer.append_batch(
                data=target,
                start_timestep=start_timestep,
                batch_time=batch_time,
            )

    def flush(self):
        self.prediction_writer.flush()
        if self.reference_writer:
            self.reference_writer.flush()

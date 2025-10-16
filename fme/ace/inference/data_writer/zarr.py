import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import cftime
import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.utils import encode_timestep
from fme.core.dataset.xarray import _get_timestep
from fme.core.writer import DATETIME_ENCODING_UNITS, ZarrWriter

from .dataset_metadata import DatasetMetadata

LEAD_TIME_UNITS = "microseconds"


def _variable_metadata_to_dict(
    variable_metadata: Mapping[str, VariableMetadata] | None,
) -> dict[str, dict[str, str]] | None:
    if variable_metadata is None:
        return None
    return {
        var: {"units": metadata.units, "long_name": metadata.long_name}
        for var, metadata in variable_metadata.items()
    }


def _get_ace_time_coords(batch_time: xr.DataArray, n_timesteps: int):
    dt_timedelta = _get_timestep(batch_time.isel(sample=0).values)
    dt_microseconds = encode_timestep(dt_timedelta)
    init_times_datetime = batch_time.isel(time=0).values
    init_times_numeric = cftime.date2num(
        init_times_datetime,
        units=DATETIME_ENCODING_UNITS,
        calendar=batch_time.dt.calendar,
    )
    lead_times_microseconds = dt_microseconds * np.arange(n_timesteps)
    lead_times_coord = xr.DataArray(
        lead_times_microseconds,
        dims=["time"],
        attrs={"units": "microseconds", "dtype": "timedelta64[us]"},
    )
    init_times_coord = xr.DataArray(
        init_times_numeric,
        dims=["sample"],
        attrs={"units": DATETIME_ENCODING_UNITS, "calendar": batch_time.dt.calendar},
    )
    valid_times_coord = init_times_coord + lead_times_coord
    valid_times_coord.attrs = {
        "units": DATETIME_ENCODING_UNITS,
        "calendar": batch_time.dt.calendar,
    }
    return lead_times_coord, init_times_coord, valid_times_coord


@dataclass
class ZarrWriterConfig:
    name: Literal["zarr"] = "zarr"  # defined for yaml+dacite ease of use
    chunks: dict[str, int] | None = field(
        default_factory=lambda: {"time": 1, "sample": 1}
    )
    overwrite_check: bool = False


class ZarrWriterAdapter:
    _SAMPLE_DIM = 0
    _TIME_DIM = 1

    def __init__(
        self,
        path: str,
        dims: tuple,
        data_coords: dict[str, xr.DataArray],
        n_timesteps: int,
        n_initial_conditions: int,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        dataset_metadata: DatasetMetadata | None = None,
        data_vars: list[str] | None = None,
        chunks: dict[str, int] | None = None,
        overwrite_check: bool = False,
    ):
        self.path = path
        self.dims = dims

        self.n_timesteps = n_timesteps
        self.n_initial_conditions = n_initial_conditions
        self.variable_metadata = _variable_metadata_to_dict(variable_metadata)

        dataset_metadata = copy.copy(dataset_metadata)

        self.dataset_metadata = (
            dataset_metadata.as_flat_str_dict() if dataset_metadata else {}
        )
        dataset_title = os.path.basename(path).removesuffix(".zarr").replace("_", " ")
        self.dataset_metadata["title"] = f"ACE {dataset_title} data"
        self.data_vars = data_vars

        self._set_chunks(chunks)

        self.overwrite_check = overwrite_check
        # writer is initialized when first batch is seen
        self._writer: ZarrWriter | None = None

        # spatial coords are passed at init, time coords are read from first batch
        self._nondim_coords: dict[str, xr.DataArray] = {}
        for vertical_nondim_coord in ["ak", "bk"]:
            if vertical_nondim_coord in data_coords:
                self._nondim_coords[vertical_nondim_coord] = xr.DataArray(
                    data_coords.pop(vertical_nondim_coord).values,
                    dims=("z_interface",),
                )
        self._horizontal_coords = data_coords

    def _set_chunks(self, chunks):
        # Time and sample chunks required to be 1
        _chunks = chunks or {}
        if _chunks.get("time", 1) != 1 or _chunks.get("sample", 1) != 1:
            raise ValueError("Chunks for 'time' and 'sample' dimensions must be 1.")
        self.chunks = {**_chunks, "time": 1, "sample": 1}

    @property
    def writer(self) -> ZarrWriter:
        if self._writer is None:
            raise RuntimeError("ZarrWriter is not initialized yet.")
        return self._writer

    def _initialize_writer(self, batch_time: xr.DataArray):
        # batch.time is dataarray with dims (sample, time) w/o coords
        lead_times_coord, init_times_coord, valid_times_coord = _get_ace_time_coords(
            batch_time, self.n_timesteps
        )
        self._nondim_coords.update(
            {
                "init_time": init_times_coord,
                "valid_time": valid_times_coord,
            }
        )

        self._dim_coords = {
            **self._horizontal_coords,
            "time": lead_times_coord,
            "sample": np.arange(self.n_initial_conditions),
        }
        self._writer = ZarrWriter(
            path=self.path,
            dims=self.dims,
            coords=self._dim_coords,
            data_vars=self.data_vars,
            chunks=self.chunks,
            shards={"time": batch_time.sizes["time"]},
            array_attributes=self.variable_metadata,
            group_attributes=self.dataset_metadata,
            time_units=LEAD_TIME_UNITS,
            time_calendar=None,
            nondim_coords=self._nondim_coords,
            mode="w",  # ACE data writers are expected to overwrite existing data
            overwrite_check=self.overwrite_check,
        )

    def _to_ndarray_mapping(
        self, data: dict[str, torch.Tensor]
    ) -> dict[str, np.ndarray]:
        vars = self.data_vars or list(data.keys())
        return {k: v.cpu().numpy() for k, v in data.items() if k in vars}

    def append_batch(
        self,
        data: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
    ) -> None:
        """
        data: Dict mapping variable name to tensor to
        start_timestep: Timestep (lead time dim) at which to start writing.
        batch_time: Time coordinate for each sample in the batch.
        """
        # Zarr store initialization needs the full time coordinate information,
        # which is not available until the first batch is seen.
        if self._writer is None:
            self._initialize_writer(batch_time)
        self.writer.record_batch(
            data=self._to_ndarray_mapping(data),
            position_slices={
                "time": slice(start_timestep, start_timestep + batch_time.sizes["time"])
            },
        )

    def flush(self):
        pass

    def finalize(self):
        pass


class SeparateICZarrWriterAdapter:
    """
    For simplicity, create a ZarrWriter for each IC sample.
    """

    def __init__(
        self,
        path: str,
        dims: tuple,
        data_coords: dict[str, np.ndarray],
        n_timesteps: int,
        n_initial_conditions: int,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        dataset_metadata: DatasetMetadata | None = None,
        data_vars: list[str] | None = None,
        chunks: dict[str, int] | None = None,
        overwrite_check: bool = False,
    ):
        self.path = path
        self.dims = dims
        # spatial coords are passed at init, time coords are read from first batch
        self.coords = data_coords
        self.n_timesteps = n_timesteps
        self.n_initial_conditions = n_initial_conditions
        self.data_vars = data_vars
        self.chunks = chunks
        self.overwrite_check = overwrite_check
        self._writers: list[ZarrWriter] | None = None

        self.variable_metadata = _variable_metadata_to_dict(variable_metadata)

        dataset_metadata = copy.copy(dataset_metadata)

        self.dataset_metadata = (
            dataset_metadata.as_flat_str_dict() if dataset_metadata else {}
        )
        dataset_title = os.path.basename(path).removesuffix(".zarr").replace("_", " ")
        self.dataset_metadata["title"] = f"ACE {dataset_title} data"

        # spatial coords are passed at init, time coords are read from first batch
        self._nondim_coords: dict[str, xr.DataArray] = {}
        for vertical_nondim_coord in ["ak", "bk"]:
            if vertical_nondim_coord in data_coords:
                self._nondim_coords[vertical_nondim_coord] = xr.DataArray(
                    data_coords.pop(vertical_nondim_coord).values,
                    dims=("z_interface",),
                )
        self._horizontal_coords = data_coords

    @property
    def writers(self) -> list[ZarrWriter]:
        if self._writers is None:
            raise RuntimeError("ZarrWriters are not initialized yet.")
        return self._writers

    def _initialize_writers(self, first_batch_time: xr.DataArray):
        self._writers = []
        for s in range(self.n_initial_conditions):
            _coords = copy.copy(self.coords)
            start_time = first_batch_time.isel(sample=s).values[0]
            timestep = first_batch_time.isel(sample=s).values[1] - start_time
            _coords["time"] = np.array(
                [start_time + i * timestep for i in range(self.n_timesteps)]
            )
            self._writers.append(
                ZarrWriter(
                    path=self.path.replace(".zarr", f"_ic{s:04d}.zarr"),
                    dims=self.dims,
                    coords=_coords,
                    data_vars=self.data_vars,
                    chunks=self.chunks,
                    shards={"time": first_batch_time.sizes["time"]},
                    array_attributes=self.variable_metadata,
                    group_attributes=self.dataset_metadata,
                    nondim_coords=self._nondim_coords,
                    mode="w",
                    overwrite_check=self.overwrite_check,
                )
            )

    def append_batch(
        self,
        data: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
    ) -> None:
        # Zarr store initialization needs the full time coordinate information,
        # which is not available until the first batch is seen.
        if self._writers is None:
            self._initialize_writers(batch_time)
        vars = self.data_vars or list(data.keys())
        for s in range(self.n_initial_conditions):
            position_slice = {
                "time": slice(start_timestep, start_timestep + batch_time.sizes["time"])
            }
            self.writers[s].record_batch(
                data={k: v.cpu().numpy()[s] for k, v in data.items() if k in vars},
                position_slices=position_slice,
            )

    def flush(self):
        pass

    def finalize(self):
        pass

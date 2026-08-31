import copy
import datetime
import os
from collections.abc import Iterable, Mapping, Sequence

import cftime
import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
import zarr

from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.monthly_aggregation import (
    COUNTS,
    ENSEMBLE_DIM,
    INIT_TIME,
    LEAD_TIME_DIM,
    LEAD_TIME_UNITS,
    TIME_UNITS,
    VALID_TIME,
    MonthIndexer,
    add_data,
    get_valid_times,
)
from fme.ace.inference.data_writer.raw import infer_calendar
from fme.ace.inference.data_writer.utils import (
    DIM_INFO_HEALPIX,
    DIM_INFO_LATLON,
    get_all_names,
)
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.writer import DATETIME_ENCODING_UNITS

FLOAT_DTYPE = "f4"


class MonthlyZarrWriter:
    """
    Write monthly mean data and sample counts to a zarr store.

    The zarr counterpart of ``MonthlyDataWriter``: the aggregation is the same,
    but the lead-time axis is pre-allocated from the run's length rather than
    grown as data arrives, since zarr has no unlimited dimension. Each
    ``append_batch`` reads back the months the batch touches, folds the new
    data into their stored means using the stored counts, and writes them back.

    Unlike ``MonthlyDataWriter``, the store may live on any fsspec-compatible
    filesystem, not only a local one.
    """

    def __init__(
        self,
        path: str,
        initial_condition_times: npt.NDArray[cftime.datetime],
        n_timesteps: int,
        timestep: datetime.timedelta,
        save_names: Sequence[str] | None,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
        chunks: Mapping[str, int] | None = None,
    ):
        """
        Args:
            path: Path of the zarr store, on any fsspec-compatible filesystem.
            initial_condition_times: 1D array of initial condition times
                (start time for each inference run).
            n_timesteps: Total number of inference forward steps, used to size
                the store's lead-time axis.
            timestep: The time delta between each timestep.
            save_names: Names of variables to save. If None, all predicted
                variables will be saved.
            variable_metadata: Metadata for each variable to be written.
            coords: Coordinate data to be written to the store.
            dataset_metadata: Metadata for the dataset.
            chunks: Optional mapping of dimension name to chunk size. Omitted
                dimensions are chunked whole, except the sample and lead-time
                dimensions, which are always chunked as 1.
        """
        self.path = path
        self.coords = coords
        self.variable_metadata = variable_metadata
        self._save_names = save_names
        self._calendar = infer_calendar(initial_condition_times)
        self._initial_condition_times = initial_condition_times
        self._month_indexer = MonthIndexer(len(initial_condition_times))
        self._final_times = [
            time + n_timesteps * timestep for time in initial_condition_times
        ]
        self._chunks = _validate_chunks(chunks)

        label = os.path.basename(path).removesuffix(".zarr")
        dataset_metadata = copy.copy(dataset_metadata)
        dataset_metadata.title = f"ACE {label.replace('_', ' ')} data file"
        self._dataset_metadata = dataset_metadata.as_flat_str_dict()

        # the store's spatial dimensions and month origin are only known once
        # the first batch is seen
        self._n_months: int | None = None
        self._root: zarr.Group | None = None

    @property
    def _store(self) -> zarr.Group:
        if self._root is None:
            raise RuntimeError("Zarr store is not initialized yet.")
        return self._root

    def _get_variable_names_to_save(
        self, *data_varnames: Iterable[str]
    ) -> Iterable[str]:
        return get_all_names(*data_varnames, allowlist=self._save_names)

    def _chunks_for(self, dims: Sequence[str], sizes: Sequence[int]) -> tuple[int, ...]:
        return tuple(
            self._chunks.get(dim, size) for dim, size in zip(dims, sizes, strict=True)
        )

    def _initialize_store(self, data: Mapping[str, torch.Tensor], n_months: int):
        """Create the store, with the lead-time axis pre-allocated."""
        dim_info = DIM_INFO_HEALPIX if "face" in self.coords else DIM_INFO_LATLON
        example = next(iter(data.values()))
        n_samples = example.shape[0]
        root = zarr.open_group(self.path, mode="w")
        root.update_attributes(self._dataset_metadata)

        lead_time = root.create_array(
            name=LEAD_TIME_DIM,
            shape=(n_months,),
            dtype="int64",
            dimension_names=[LEAD_TIME_DIM],
        )
        lead_time.attrs["units"] = LEAD_TIME_UNITS
        lead_time[:] = np.arange(n_months)

        init_time = root.create_array(
            name=INIT_TIME,
            shape=(n_samples,),
            dtype="int64",
            dimension_names=[ENSEMBLE_DIM],
        )
        init_time.attrs["units"] = DATETIME_ENCODING_UNITS
        init_time.attrs["calendar"] = self._calendar
        init_time[:] = cftime.date2num(
            self._initial_condition_times,
            units=DATETIME_ENCODING_UNITS,
            calendar=self._calendar,
        )

        valid_time = root.create_array(
            name=VALID_TIME,
            shape=(n_samples, n_months),
            dtype="int64",
            dimension_names=[ENSEMBLE_DIM, LEAD_TIME_DIM],
        )
        valid_time.attrs["units"] = TIME_UNITS
        valid_time.attrs["calendar"] = self._calendar
        valid_time[:] = get_valid_times(
            init_years=self._month_indexer.init_years,
            init_months=self._month_indexer.init_months,
            n_months=n_months,
            calendar=self._calendar,
        )

        root.create_array(
            name=COUNTS,
            shape=(n_samples, n_months),
            dtype="int64",
            fill_value=0,
            dimension_names=[ENSEMBLE_DIM, LEAD_TIME_DIM],
        )

        spatial_names = []
        spatial_sizes = []
        for dim in dim_info:
            dim_size = example.shape[dim.index]
            spatial_names.append(dim.name)
            spatial_sizes.append(dim_size)
            if dim.name in self.coords:
                coord = root.create_array(
                    name=dim.name,
                    shape=(dim_size,),
                    dtype=FLOAT_DTYPE,
                    dimension_names=[dim.name],
                )
                coord[:] = np.asarray(self.coords[dim.name])

        dims = (ENSEMBLE_DIM, LEAD_TIME_DIM, *spatial_names)
        sizes = (n_samples, n_months, *spatial_sizes)
        for name in self._get_variable_names_to_save(data.keys()):
            variable = root.create_array(
                name=name,
                shape=sizes,
                chunks=self._chunks_for(dims, sizes),
                dtype=FLOAT_DTYPE,
                # means start at zero rather than NaN, since add_data folds new
                # data into the value already stored; months the run never
                # reaches are identified by a zero count
                fill_value=0.0,
                dimension_names=dims,
            )
            attrs: dict[str, str] = {}
            if name in self.variable_metadata:
                attrs.update(self.variable_metadata[name].as_attrs())
            attrs["coordinates"] = " ".join([INIT_TIME, VALID_TIME, COUNTS])
            variable.attrs.update(attrs)
        zarr.consolidate_metadata(root.store)
        self._root = zarr.open_group(self.path, mode="r+")

    def append_batch(
        self,
        data: dict[str, torch.Tensor],
        batch_time: xr.DataArray,
    ):
        """
        Fold a batch of data into the stored monthly means.

        Args:
            data: Values to store.
            batch_time: Time coordinate for each sample in the batch.
        """
        n_samples_data = list(data.values())[0].shape[0]
        n_samples_time = batch_time.sizes[ENSEMBLE_DIM]
        if n_samples_data != n_samples_time:
            raise ValueError(
                f"Batch size mismatch, data has {n_samples_data} samples "
                f"and batch_time has {n_samples_time} samples."
            )
        n_times_data = list(data.values())[0].shape[1]
        n_times_time = batch_time.sizes[LEAD_TIME_DIM]
        if n_times_data != n_times_time:
            raise ValueError(
                f"Batch time dimension mismatch, data has {n_times_data} times "
                f"and batch_time has {n_times_time} times."
            )

        months = self._month_indexer.month_indices(batch_time)
        if self._n_months is None:
            self._n_months = self._month_indexer.n_months_through(self._final_times)
        n_months = self._n_months
        if np.min(months) < 0 or np.max(months) >= n_months:
            raise ValueError(
                f"Batch times span month indices {np.min(months)} to "
                f"{np.max(months)}, outside the pre-allocated range of "
                f"{n_months} months implied by the run's initial condition times, "
                "timestep and number of timesteps."
            )
        if self._root is None:
            self._initialize_store(data, n_months)
        month_min = int(np.min(months))
        month_slice = slice(month_min, int(np.max(months)) + 1)

        counts = self._store[COUNTS]
        start_counts = counts[:, month_slice]
        for variable_name in self._get_variable_names_to_save(data.keys()):
            variable = self._store[variable_name]
            month_data = variable[:, month_slice]
            add_data(
                target=month_data,
                target_start_counts=start_counts,
                source=data[variable_name].detach().cpu().numpy(),
                months_elapsed=months - month_min,
            )
            variable[:, month_slice] = month_data
        # counts must be added after data, as we use the base counts when
        # updating means
        counts[:, month_slice] = start_counts + np.stack(
            [
                np.bincount(
                    months[i_sample] - month_min, minlength=start_counts.shape[1]
                )
                for i_sample in range(n_samples_data)
            ]
        )

    def flush(self):
        """No-op: each append_batch writes through to the store."""

    def finalize(self):
        """No-op: each append_batch writes through to the store."""


def _validate_chunks(chunks: Mapping[str, int] | None) -> dict[str, int]:
    """Chunk sizes for a monthly store, requiring sample and lead time of 1."""
    _chunks = dict(chunks or {})
    if _chunks.get(LEAD_TIME_DIM, 1) != 1 or _chunks.get(ENSEMBLE_DIM, 1) != 1:
        raise ValueError(
            f"Chunks for '{LEAD_TIME_DIM}' and '{ENSEMBLE_DIM}' dimensions must be 1."
        )
    return {**_chunks, LEAD_TIME_DIM: 1, ENSEMBLE_DIM: 1}

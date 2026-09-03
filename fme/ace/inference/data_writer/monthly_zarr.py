import copy
import datetime
import os
from collections.abc import Iterable, Mapping, Sequence

import cftime
import numpy as np
import numpy.typing as npt
import torch
import xarray as xr

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
from fme.core.writer import DATETIME_ENCODING_UNITS, ZarrWriter

FLOAT_DTYPE = "f4"


class MonthlyZarrWriter:
    """
    Write monthly mean data and sample counts to a zarr store.

    The zarr counterpart of ``MonthlyDataWriter``. The lead-time axis is
    pre-allocated from the run's length, since zarr has no unlimited dimension,
    and each ``append_batch`` reads back the months the batch touches, folds
    the new data into their stored means using the stored counts, and writes
    them back.

    The store may live on any fsspec-compatible filesystem, unlike the netCDF
    writer's.
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
            coords: Coordinate data to be written to the store. Must include a
                coordinate for each spatial dimension, as for the raw zarr
                writer, since the store is sized before any data arrives.
            dataset_metadata: Metadata for the dataset.
            chunks: Optional mapping of dimension name to chunk size. Omitted
                dimensions are chunked whole, except the sample and lead-time
                dimensions, which are always chunked as 1. The store is
                unsharded. Both are deliberate: each ``append_batch`` reads back
                and rewrites exactly the (sample, month) cells the batch
                touches, so a chunk spanning more than one month or sample would
                mean read-modify-write amplification on every batch, and
                sharding would place unrelated months in one storage object. At
                monthly resolution the resulting chunk count is small.
        """
        self._save_names = save_names
        calendar = infer_calendar(initial_condition_times)
        n_samples = len(initial_condition_times)
        # The inference loop drops the initial condition step, so the run's
        # output times are IC + timestep through IC + n_timesteps * timestep.
        # Both ends of the month axis therefore follow from the constructor
        # args, and n_months with them.
        self._month_indexer = MonthIndexer(
            first_output_times=[time + timestep for time in initial_condition_times]
        )
        self._n_months = self._month_indexer.n_months_through(
            [time + n_timesteps * timestep for time in initial_condition_times]
        )

        dim_info = DIM_INFO_HEALPIX if "face" in coords else DIM_INFO_LATLON
        spatial_names = [dim.name for dim in dim_info]

        label = os.path.basename(path).removesuffix(".zarr")
        dataset_metadata = copy.copy(dataset_metadata)
        dataset_metadata.title = f"ACE {label.replace('_', ' ')} data file"

        self._writer = ZarrWriter(
            path=path,
            dims=(ENSEMBLE_DIM, LEAD_TIME_DIM, *spatial_names),
            coords={
                ENSEMBLE_DIM: np.arange(n_samples),
                LEAD_TIME_DIM: np.arange(self._n_months),
                **{name: np.asarray(coords[name]) for name in spatial_names},
            },
            chunks=_validate_chunks(chunks),
            array_attributes={
                name: metadata.as_attrs()
                for name, metadata in variable_metadata.items()
            },
            group_attributes=dataset_metadata.as_flat_str_dict(),
            mode="w",  # ACE data writers are expected to overwrite existing data
            # each batch rewrites the months it touches, by design
            overwrite_check=False,
            # the lead-time axis is a month count, not a date
            time_units=LEAD_TIME_UNITS,
            time_calendar=None,
            nondim_coords=_nondim_coords(
                initial_condition_times=initial_condition_times,
                month_indexer=self._month_indexer,
                n_months=self._n_months,
                n_samples=n_samples,
                calendar=calendar,
            ),
        )
        # With save_names=None the variables to create aren't known until the
        # first batch, so the store is initialized then.
        self._store_initialized = False

    def _get_variable_names_to_save(self, *data_varnames: Iterable[str]) -> list[str]:
        return list(get_all_names(*data_varnames, allowlist=self._save_names))

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
        if np.min(months) < 0 or np.max(months) >= self._n_months:
            raise ValueError(
                f"Batch times span month indices {np.min(months)} to "
                f"{np.max(months)}, outside the pre-allocated range of "
                f"{self._n_months} months implied by the run's initial condition "
                "times, timestep and number of timesteps."
            )

        names = self._get_variable_names_to_save(data.keys())
        if not self._store_initialized:
            self._writer.initialize_store(data_dtype=FLOAT_DTYPE, data_vars=names)
            self._store_initialized = True

        month_min = int(np.min(months))
        position_slices = {LEAD_TIME_DIM: slice(month_min, int(np.max(months)) + 1)}

        # counts is stored as a non-dimension coordinate, so that readers attach
        # it to the data, but it is read back and updated like the data itself.
        stored = self._writer.read_batch([*names, COUNTS], position_slices)
        start_counts = stored[COUNTS]
        for name in names:
            add_data(
                target=stored[name],
                target_start_counts=start_counts,
                source=data[name].detach().cpu().numpy(),
                months_elapsed=months - month_min,
            )
        # counts must be updated after the data, as the base counts are what
        # the mean update above folds into
        stored[COUNTS] = start_counts + np.stack(
            [
                np.bincount(
                    months[i_sample] - month_min, minlength=start_counts.shape[1]
                )
                for i_sample in range(n_samples_data)
            ]
        )
        self._writer.record_batch(data=stored, position_slices=position_slices)

    def flush(self):
        """No-op: each append_batch writes through to the store."""

    def finalize(self):
        """No-op: each append_batch writes through to the store."""


def _nondim_coords(
    initial_condition_times: npt.NDArray[cftime.datetime],
    month_indexer: MonthIndexer,
    n_months: int,
    n_samples: int,
    calendar: str,
) -> dict[str, xr.DataArray]:
    """Init time, valid time and counts, in the order readers should see them."""
    init_time = xr.DataArray(
        cftime.date2num(
            initial_condition_times,
            units=DATETIME_ENCODING_UNITS,
            calendar=calendar,
        ).astype(np.int64),
        dims=(ENSEMBLE_DIM,),
        attrs={"units": DATETIME_ENCODING_UNITS, "calendar": calendar},
    )
    valid_time = xr.DataArray(
        get_valid_times(
            init_years=month_indexer.init_years,
            init_months=month_indexer.init_months,
            n_months=n_months,
            calendar=calendar,
        ),
        dims=(ENSEMBLE_DIM, LEAD_TIME_DIM),
        attrs={"units": TIME_UNITS, "calendar": calendar},
    )
    # counts starts at zero, and so do the means, since add_data folds into the
    # stored value; a month the run never reaches keeps a mean of 0 and a count
    # of 0, so consumers must use counts to tell a real zero from an unwritten
    # month
    counts = xr.DataArray(
        np.zeros((n_samples, n_months), dtype=np.int64),
        dims=(ENSEMBLE_DIM, LEAD_TIME_DIM),
    )
    return {INIT_TIME: init_time, VALID_TIME: valid_time, COUNTS: counts}


def _validate_chunks(chunks: Mapping[str, int] | None) -> dict[str, int]:
    """Chunk sizes for a monthly store, requiring sample and lead time of 1."""
    _chunks = dict(chunks or {})
    if _chunks.get(LEAD_TIME_DIM, 1) != 1 or _chunks.get(ENSEMBLE_DIM, 1) != 1:
        raise ValueError(
            f"Chunks for '{LEAD_TIME_DIM}' and '{ENSEMBLE_DIM}' dimensions must be 1."
        )
    return {**_chunks, LEAD_TIME_DIM: 1, ENSEMBLE_DIM: 1}

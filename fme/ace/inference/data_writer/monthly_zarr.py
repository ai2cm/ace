import copy
import datetime
import logging
from collections.abc import Iterable, Mapping, Sequence

import cftime
import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
import zarr

from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.monthly import (
    COUNTS,
    INIT_TIME,
    LEAD_TIME_DIM,
    LEAD_TIME_UNITS,
    TIME_UNITS,
    VALID_TIME,
    add_data,
    get_days_since_reference,
)
from fme.ace.inference.data_writer.raw import infer_calendar
from fme.ace.inference.data_writer.utils import (
    DIM_INFO_HEALPIX,
    DIM_INFO_LATLON,
    get_all_names,
)
from fme.core.cloud import (
    exists,
    open_dataset_via_inter_filesystem_copy,
    to_netcdf_via_inter_filesystem_copy,
)
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.writer import DATETIME_ENCODING_UNITS

logger = logging.getLogger(__name__)

ENSEMBLE_DIM = "sample"
INIT_YEARS_ATTR = "fme_init_years"
INIT_MONTHS_ATTR = "fme_init_months"
MONTHLY_SNAPSHOT_DIR = "monthly_snapshots"


def _n_months_in_run(
    init_years: np.ndarray,
    init_months: np.ndarray,
    initial_condition_times: npt.NDArray[cftime.datetime],
    timestep: datetime.timedelta,
    n_timesteps: int,
) -> int:
    """Number of calendar months touched by the run, across all samples."""
    n_months = 0
    for i, init_time in enumerate(initial_condition_times):
        last_time = init_time + n_timesteps * timestep
        months = 12 * (last_time.year - init_years[i]) + (
            last_time.month - 1 - init_months[i]
        )
        n_months = max(n_months, int(months) + 1)
    return n_months


class MonthlyZarrWriter:
    """
    Write monthly mean data and sample counts to a zarr store.

    Unlike ``MonthlyDataWriter``, the month axis is pre-allocated from the
    whole run's length, so the store can be shared across the segments of a
    segmented run: each ``append_batch`` reads the affected months back, folds
    the new data into the stored means using the stored counts, and writes them
    back. Because that accumulation is not idempotent, ``finalize`` writes a
    netCDF snapshot of the store to ``snapshot_path``; a writer constructed
    with ``restore_path`` (the previous segment's snapshot) starts from that
    state, so re-running a partially completed segment reproduces the same
    result as running it once.
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
        snapshot_path: str | None = None,
        restore_path: str | None = None,
    ):
        """
        Args:
            path: Path of the zarr store, on any fsspec-compatible filesystem.
            initial_condition_times: 1D array of initial condition times. When
                ``restore_path`` is not given these must be the run's initial
                condition times, from which the store's month axis is computed.
            n_timesteps: Number of timesteps in the whole run (all segments).
            timestep: The time delta between each timestep.
            save_names: Names of variables to save, or None to save all.
            variable_metadata: Metadata for each variable.
            coords: Coordinate data for the spatial dimensions.
            dataset_metadata: Metadata for the dataset.
            snapshot_path: If given, ``finalize`` writes a netCDF snapshot of
                the store here.
            restore_path: If given, the store is rebuilt from this snapshot
                (written by the previous segment) instead of starting empty.
        """
        self.path = path
        self._save_names = save_names
        self.variable_metadata = variable_metadata
        self.coords = coords
        self._snapshot_path = snapshot_path
        self._label = path.rsplit("/", 1)[-1].removesuffix(".zarr")

        dataset_metadata = copy.copy(dataset_metadata)
        dataset_metadata.title = f"ACE {self._label.replace('_', ' ')} data file"
        self._dataset_metadata = dataset_metadata.as_flat_str_dict()

        self._calendar = infer_calendar(initial_condition_times)
        n_initial_conditions = len(initial_condition_times)
        self._initial_condition_times = initial_condition_times

        if restore_path is not None:
            if not exists(restore_path):
                raise RuntimeError(
                    f"Cannot restore monthly writer state from {restore_path}: "
                    "the snapshot does not exist. It is written by the previous "
                    "segment's finalize."
                )
            snapshot = open_dataset_via_inter_filesystem_copy(
                restore_path, decode_times=False, decode_timedelta=False
            )
            # the month origin of the whole run rides the snapshot; the initial
            # condition times of a resumed segment sit at its restart time and
            # must not redefine it
            # netCDF attributes collapse single-element lists to scalars
            self._init_years = np.atleast_1d(
                np.asarray(snapshot.attrs[INIT_YEARS_ATTR], dtype=int)
            )
            self._init_months = np.atleast_1d(
                np.asarray(snapshot.attrs[INIT_MONTHS_ATTR], dtype=int)
            )
            self._n_months = int(snapshot.sizes[LEAD_TIME_DIM])
            # write the normalized origin attributes so the restored store's
            # attributes match those of a store that was never snapshotted
            snapshot.attrs[INIT_YEARS_ATTR] = self._init_years.tolist()
            snapshot.attrs[INIT_MONTHS_ATTR] = self._init_months.tolist()
            snapshot.to_zarr(self.path, mode="w")
            self._store_initialized = True
        else:
            # months are indexed relative to each sample's first forward
            # (post-initial-condition) time
            first_forward_times = initial_condition_times + timestep
            self._init_years = np.array(
                [t.year for t in first_forward_times], dtype=int
            )
            self._init_months = np.array(
                [t.month - 1 for t in first_forward_times], dtype=int
            )
            self._n_months = _n_months_in_run(
                self._init_years,
                self._init_months,
                initial_condition_times,
                timestep,
                n_timesteps,
            )
            # the store's spatial dims are only known once data is seen
            self._store_initialized = False

        if n_initial_conditions != len(self._init_years):
            raise ValueError(
                f"Got {n_initial_conditions} initial conditions but the "
                f"restored snapshot has {len(self._init_years)} samples."
            )

    def _initialize_store(self, data: dict[str, torch.Tensor]):
        """Create the empty store, with the month axis pre-allocated."""
        dim_info = DIM_INFO_HEALPIX if "face" in self.coords else DIM_INFO_LATLON
        example = next(iter(data.values()))
        n_samples = example.shape[0]
        root = zarr.open_group(self.path, mode="w")
        root.update_attributes(
            {
                **self._dataset_metadata,
                INIT_YEARS_ATTR: self._init_years.tolist(),
                INIT_MONTHS_ATTR: self._init_months.tolist(),
            }
        )

        lead_time = root.create_array(
            name=LEAD_TIME_DIM,
            shape=(self._n_months,),
            dtype="int64",
            dimension_names=[LEAD_TIME_DIM],
        )
        lead_time.attrs["units"] = LEAD_TIME_UNITS
        lead_time[:] = np.arange(self._n_months)

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
            shape=(n_samples, self._n_months),
            dtype="int64",
            dimension_names=[ENSEMBLE_DIM, LEAD_TIME_DIM],
        )
        valid_time.attrs["units"] = TIME_UNITS
        valid_time.attrs["calendar"] = self._calendar
        reference_date = cftime.datetime(1970, 1, 1, calendar=self._calendar)
        days_since_reference = get_days_since_reference(
            years=self._init_years,
            months=self._init_months,
            n_months=self._n_months,
            reference_date=reference_date,
            calendar=self._calendar,
        )
        # use the 15th of each month, which is 14 days into the month
        valid_time[:] = days_since_reference + 14

        counts = root.create_array(
            name=COUNTS,
            shape=(n_samples, self._n_months),
            dtype="int64",
            dimension_names=[ENSEMBLE_DIM, LEAD_TIME_DIM],
        )
        counts[:] = 0

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
                    dtype="f4",
                    dimension_names=[dim.name],
                )
                coord[:] = np.asarray(self.coords[dim.name])

        dims = (ENSEMBLE_DIM, LEAD_TIME_DIM, *spatial_names)
        for name in self._get_variable_names_to_save(data.keys()):
            var = root.create_array(
                name=name,
                shape=(n_samples, self._n_months, *spatial_sizes),
                chunks=(1, 1, *spatial_sizes),
                dtype="f4",
                dimension_names=dims,
            )
            attrs: dict[str, str] = {}
            if name in self.variable_metadata:
                attrs.update(self.variable_metadata[name].as_attrs())
            attrs["coordinates"] = " ".join([INIT_TIME, VALID_TIME, COUNTS])
            var.attrs.update(attrs)
            var[:] = 0.0
        zarr.consolidate_metadata(root.store)
        self._store_initialized = True

    def _get_variable_names_to_save(
        self, *data_varnames: Iterable[str]
    ) -> Iterable[str]:
        return get_all_names(*data_varnames, allowlist=self._save_names)

    def _get_month_indices(self, batch_time: xr.DataArray) -> np.ndarray:
        years = batch_time.dt.year.values
        # datetime months are 1-indexed, we want 0-indexed
        months = batch_time.dt.month.values - 1
        return 12 * (years - self._init_years[:, None]) + (
            months - self._init_months[:, None]
        )

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
        n_samples_time = batch_time.sizes["sample"]
        if n_samples_data != n_samples_time:
            raise ValueError(
                f"Batch size mismatch, data has {n_samples_data} samples "
                f"and batch_time has {n_samples_time} samples."
            )
        n_times_data = list(data.values())[0].shape[1]
        n_times_time = batch_time.sizes["time"]
        if n_times_data != n_times_time:
            raise ValueError(
                f"Batch time dimension mismatch, data has {n_times_data} times "
                f"and batch_time has {n_times_time} times."
            )

        if not self._store_initialized:
            self._initialize_store(data)

        months = self._get_month_indices(batch_time)
        if np.min(months) < 0 or np.max(months) >= self._n_months:
            raise ValueError(
                f"Batch times span month indices {np.min(months)} to "
                f"{np.max(months)}, outside the store's pre-allocated range of "
                f"{self._n_months} months."
            )
        month_min = int(np.min(months))
        month_slice = slice(month_min, int(np.max(months)) + 1)

        root = zarr.open_group(self.path, mode="r+")
        count_data = root[COUNTS][:, month_slice]
        for variable_name in self._get_variable_names_to_save(data.keys()):
            array = data[variable_name].detach().cpu().numpy()
            month_data = root[variable_name][:, month_slice]
            add_data(
                target=month_data,
                target_start_counts=count_data,
                source=array,
                months_elapsed=months - month_min,
            )
            root[variable_name][:, month_slice] = month_data
        # counts must be added after data, as we use the base counts when
        # updating means
        counts = root[COUNTS]
        for i_sample in range(n_samples_data):
            counts[i_sample] = counts[i_sample] + np.bincount(
                months[i_sample], minlength=self._n_months
            )

    def flush(self):
        pass

    def finalize(self):
        """Write a snapshot of the store, if configured to do so.

        In a segmented run this runs before the segment's restart files are
        written, so a segment with restart files always has a snapshot.
        """
        if self._snapshot_path is None:
            return
        if not self._store_initialized:
            logger.warning(
                f"No data was written to {self.path}; skipping its snapshot."
            )
            return
        snapshot = xr.open_zarr(
            self.path, decode_times=False, decode_timedelta=False
        ).load()
        snapshot.attrs[INIT_YEARS_ATTR] = self._init_years.tolist()
        snapshot.attrs[INIT_MONTHS_ATTR] = self._init_months.tolist()
        to_netcdf_via_inter_filesystem_copy(snapshot, self._snapshot_path)

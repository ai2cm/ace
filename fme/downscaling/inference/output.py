from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from fme.core.dataset.merged import MergeNoConcatDatasetConfig
from fme.core.dataset.time import RepeatedInterval, TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.distributed import Distributed
from fme.core.typing_ import Slice
from fme.core.writer import ZarrWriter

from ..data import (
    ClosedInterval,
    DataLoaderConfig,
    LatLonCoordinates,
    enforce_lat_bounds,
)
from ..data.config import XarrayEnsembleDataConfig
from ..predictors import PatchPredictionConfig
from ..requirements import DataRequirements
from .constants import DIMS
from .work_items import SliceItemDataset, SliceWorkItemGriddedData, get_work_items
from .zarr_utils import determine_zarr_chunks


def _identity_collate(batch):
    """
    Collate function that returns the single batch item.

    Used with batch_size=1 to extract the single item from the batch list.
    Must be a module-level function (not lambda) to be picklable for multiprocessing.
    """
    return batch[0]


class DownscalingOutput:
    """
    Container for a single downscaling output.

    Encapsulates all data and metadata needed to generate downscaled outputs
    for a specific region, time range, and ensemble configuration.

    Parameters:
        name: Identifier for this target (used as output filenames).
        save_vars: List of variable names to save to zarr.
        n_ens: Total number of ensemble members to generate.
        max_samples_per_gpu: Max number of time and/or ensemble samples per GPU batch.
            The breakdown of time vs ensemble per batch is determined automatically.
        data: GriddedData containing the input coarse data and loader.
        patch: Configuration for patching large domains.
        chunks: Zarr chunk sizes for each dimension.
        shards: Zarr shard sizes for each dimension.
        dims: Dimension names including ensemble (time, ensemble, lat, lon).
        overwrite: If True, overwrite an existing zarr store at this output's
            path instead of failing. Useful when a prior attempt was killed
            (preemption, OOM, etc.) partway through and left a partial store
            behind and a from-scratch restart is acceptable. Mutually
            exclusive with resume.
        resume: If True, open an existing zarr store at this output's path
            (creating it if it doesn't exist yet) and skip regenerating any
            work item whose target slice is already fully written -- see
            ZarrWriter.is_slice_written. Requires save_vars to be set
            explicitly (skip-detection needs a concrete variable list to
            check; with save_vars=None nothing will ever be skipped, which
            is safe but makes resume a no-op). Mutually exclusive with
            overwrite.
    """

    def __init__(
        self,
        name: str,
        save_vars: list[str] | None,
        n_ens: int,
        max_samples_per_gpu: int,
        data: SliceWorkItemGriddedData,
        patch: PatchPredictionConfig,
        chunks: dict[str, int],
        shards: dict[str, int],
        dims: tuple[str, ...] = DIMS,
        overwrite: bool = False,
        resume: bool = False,
    ) -> None:
        self.name = name
        self.save_vars = save_vars
        self.n_ens = n_ens
        self.max_samples_per_gpu = max_samples_per_gpu
        self.data = data
        self.patch = patch
        self.chunks = chunks
        self.shards = shards
        self.dims = dims
        self.overwrite = overwrite
        self.resume = resume

    def get_writer(
        self,
        latlon_coords: LatLonCoordinates,
        output_dir: str,
    ) -> ZarrWriter:
        """
        Create a ZarrWriter for this target.

        Args:
            latlon_coords: High-resolution spatial coordinates for outputs
            output_dir: Directory to store output zarr file
        """
        ensemble = list(range(self.n_ens))
        coords = dict(
            zip(
                self.dims,
                [
                    self.data.all_times.to_numpy(),
                    np.array(ensemble),
                    latlon_coords.lat.cpu().numpy(),
                    latlon_coords.lon.cpu().numpy(),
                ],
            )
        )
        dims = tuple(coords.keys())

        mode: Literal["r+", "a", "w", "w-"]
        if self.resume:
            mode = "a"
        elif self.overwrite:
            mode = "w"
        else:
            mode = "w-"

        return ZarrWriter(
            path=f"{output_dir}/{self.name}.zarr",
            dims=dims,
            coords=coords,
            data_vars=self.save_vars,
            chunks=self.chunks,
            shards=self.shards,
            mode=mode,
            # Resume's own is_slice_written check is the safety net during a
            # resumed run (it only lets a genuinely-incomplete slice reach
            # record_batch at all) -- ZarrWriter's own overwrite_check would
            # otherwise block legitimately re-writing a boundary work item
            # that has some but not all of its variables already written
            # from the prior (killed) attempt.
            overwrite_check=not self.resume,
        )


@dataclass
class DownscalingOutputConfig(ABC):
    """
    Base class for configuring downscaling output generation targets.

    Output targets define what data to generate, where to generate it, and how
    to save it.

    Parameters:
        name: Unique identifier for this target (used in output filename)
        n_ens: Number of ensemble members to generate when downscaling
        save_vars: List of variable names to save to zarr output.  If None,
            all variables from the model output will be saved.
        zarr_chunks: Optional chunk sizes for zarr dimensions. If None, automatically
            calculated to target lat/lon shape <=10MB per chunk. Ensemble and time
            dimensions chunks are length 1.
        zarr_shards: Optional shard sizes for zarr dimensions. If None, defaults to
            maximum output size for a single unit of downscaling work.  This ensures
            that parallel generation tasks write to separate shards.
        max_samples_per_gpu: Number of time and/or ensemble samples to include in a
            single GPU generation. Controls memory usage and time to generate.
        overwrite: If True, overwrite an existing zarr store at this output's
            path instead of failing with a "store already exists" error.
            False (default) matches prior behavior. Redoes all work from
            scratch -- use this for a from-scratch restart. Mutually
            exclusive with resume.
        resume: If True, skip regenerating work items whose target slice is
            already fully written in an existing store at this output's
            path, instead of always generating fresh (and instead of
            failing if the store already exists). False (default) matches
            prior behavior. Requires save_vars to be set explicitly (see
            DownscalingOutput's docstring). Mutually exclusive with
            overwrite.
    """

    name: str
    n_ens: int
    save_vars: list[str] | None = None
    zarr_chunks: dict[str, int] | None = None
    zarr_shards: dict[str, int] | None = None
    overwrite: bool = False
    resume: bool = False
    max_samples_per_gpu: int = 4

    def __post_init__(self):
        if self.overwrite and self.resume:
            raise ValueError(
                "overwrite and resume are mutually exclusive: overwrite "
                "always redoes all work from scratch, resume skips work "
                "already done."
            )

    @abstractmethod
    def build(
        self,
        loader_config: DataLoaderConfig,
        requirements: DataRequirements,
        patch: PatchPredictionConfig,
        fine_shape: tuple[int, int],
    ) -> DownscalingOutput:
        """
        Build an OutputTarget from this configuration.

        Args:
            loader_config: Base data loader configuration to modify
            requirements: Model's data requirements (variable names, etc.)
            patch: Default patch prediction configuration
            fine_shape: Fine shape of the output used as metadata
                for the shape of the output to insert into the dataset
        """
        pass

    @staticmethod
    def _single_xarray_config(
        coarse: Sequence[
            XarrayDataConfig | XarrayEnsembleDataConfig | MergeNoConcatDatasetConfig
        ],
    ) -> list[XarrayDataConfig]:
        """
        Ensures that the data configuration is a single xarray config.
        Necessary because we will be using the top-level DataLoaderConfig
        to build the data, and we'll be replacing time and spatial extents.
        """
        # TODO: Consider only supporting a single xarray config
        #       for this run type since we use Zarr not netCDF.  Just more
        #       complexity to enforce all possible rather than just supporting
        #       a single config.
        if len(coarse) != 1:
            raise NotImplementedError(
                "Only a single XarrayDataConfig is supported in OutputTargetConfig "
                " coarse specification."
            )

        data_config = coarse[0]
        if not isinstance(data_config, XarrayDataConfig):
            raise NotImplementedError(
                "Only XarrayDataConfig objects are supported in OutputTargetConfig "
                " coarse specification."
            )

        return [data_config]

    def _replace_loader_config(
        self,
        time,
        coarse,
        lat_extent,
        lon_extent,
        loader_config: DataLoaderConfig,
    ) -> DataLoaderConfig:
        new_coarse = [replace(coarse[0], subset=time)]

        # TODO: log the replacements for debugging
        new_loader_config = replace(
            loader_config,
            coarse=new_coarse,
            lat_extent=lat_extent,
            lon_extent=lon_extent,
        )
        return new_loader_config

    def _build_gridded_data(
        self,
        loader_config: DataLoaderConfig,
        requirements: DataRequirements,
        dist: Distributed | None = None,
        fine_shape: tuple[int, int] | None = None,
    ) -> SliceWorkItemGriddedData:
        xr_dataset, properties = loader_config.get_xarray_dataset(
            names=requirements.coarse_names, n_timesteps=1
        )
        coords = properties.horizontal_coordinates
        if not isinstance(coords, LatLonCoordinates):
            raise ValueError(
                "Downscaling data loader only supports datasets with latlon coords."
            )
        dataset = loader_config.build_batchitem_dataset(xr_dataset, properties)

        work_items = get_work_items(
            n_times=len(dataset),
            n_ens=self.n_ens,
            max_samples_per_gpu=self.max_samples_per_gpu,
        )

        slice_dataset = SliceItemDataset(
            slice_items=work_items,
            dataset=dataset,
            spatial_shape=fine_shape,
        )

        # each SliceItemDataset work item loads its own full batch, so batch_size=1
        dist = dist or Distributed.get_instance()
        loader = DataLoader(
            slice_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=loader_config.num_data_workers,
            collate_fn=_identity_collate,
            drop_last=False,
            multiprocessing_context=loader_config.mp_context,
            persistent_workers=True if loader_config.num_data_workers > 0 else False,
            sampler=(
                DistributedSampler(slice_dataset, shuffle=False)
                if dist.is_distributed()
                else None
            ),
        )

        return SliceWorkItemGriddedData(
            loader,
            variable_metadata=dataset.variable_metadata,
            all_times=xr_dataset.sample_start_times,
            dtype=slice_dataset.dtype,
            max_output_shape=slice_dataset.max_output_shape,
        )

    def _build(
        self,
        time: TimeSlice | RepeatedInterval | Slice,
        lat_extent: ClosedInterval,
        lon_extent: ClosedInterval,
        loader_config: DataLoaderConfig,
        requirements: DataRequirements,
        patch: PatchPredictionConfig,
        coarse: list[XarrayDataConfig],
        fine_shape: tuple[int, int] | None = None,
    ) -> DownscalingOutput:
        updated_loader_config = self._replace_loader_config(
            time,
            coarse,
            lat_extent,
            lon_extent,
            loader_config,
        )

        gridded_data = self._build_gridded_data(
            updated_loader_config,
            requirements,
            fine_shape=fine_shape,
        )

        if self.zarr_chunks is None:
            # Get element size from dtype by creating a dummy tensor
            element_size = torch.tensor([], dtype=gridded_data.dtype).element_size()
            chunks = determine_zarr_chunks(
                dims=DIMS,
                data_shape=gridded_data.max_output_shape,
                bytes_per_element=element_size,
            )
        else:
            chunks = self.zarr_chunks

        if self.zarr_shards is None:
            shards = dict(zip(DIMS, gridded_data.max_output_shape))
        else:
            shards = self.zarr_shards

        return DownscalingOutput(
            name=self.name,
            save_vars=self.save_vars,
            n_ens=self.n_ens,
            max_samples_per_gpu=self.max_samples_per_gpu,
            data=gridded_data,
            patch=patch,
            chunks=chunks,
            shards=shards,
            dims=DIMS,
            overwrite=self.overwrite,
            resume=self.resume,
        )


@dataclass
class EventConfig(DownscalingOutputConfig):
    """
    Configuration for generating a single time snapshot over a spatial region.

    Useful for capturing specific events like hurricane landfall, extreme weather
    events, or any single-timestep high-resolution snapshot of a region.

    If n_ens > max_samples_per_gpu, this event can be run in a distributed manner
    where each GPU generates a subset of the ensemble members for the event.

    Parameters:
        name: Unique identifier for this target (used in output filename)
        n_ens: Number of ensemble members to generate when downscaling
        save_vars: List of variable names to save to zarr output.  If None,
            all variables from the model output will be saved.
        zarr_chunks: Optional chunk sizes for zarr dimensions. If None, automatically
            calculated to target lat/lon shape <=10MB per chunk. Ensemble and time
            dimensions chunks are length 1.
        zarr_shards: Optional shard sizes for zarr dimensions. If None, defaults to
            maximum output size for a single unit of downscaling work.  This ensures
            that parallel generation tasks write to separate shards.
        max_samples_per_gpu: Number of time and/or ensemble samples to include in a
            single GPU generation. Controls memory usage and time to generate.
        event_time: Timestamp or integer index of the event. If string, must match
            time_format. Required field.
        time_format: strptime format for parsing event_time string.
            Default: "%Y-%m-%dT%H:%M:%S" (ISO 8601)
        lat_extent: Latitude bounds in degrees limited to [-88, 88].
        Defaults to (-66, 70) which covers continental land masses aside
            from Antarctica.
        lon_extent: Longitude bounds in degrees [-180, 360]. Default: full extent
            of the underlying data.
    """

    # event_time required, but must specify as optional kwarg to allow subclassing
    event_time: str | int = ""
    time_format: str = "%Y-%m-%dT%H:%M:%S"
    lat_extent: ClosedInterval = field(
        default_factory=lambda: ClosedInterval(-66.0, 70)
    )
    lon_extent: ClosedInterval = field(
        default_factory=lambda: ClosedInterval(float("-inf"), float("inf"))
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.event_time:
            raise ValueError("event_time must be specified for EventConfig.")
        enforce_lat_bounds(self.lat_extent)

    def build(
        self,
        loader_config: DataLoaderConfig,
        requirements: DataRequirements,
        patch: PatchPredictionConfig,
        fine_shape: tuple[int, int] | None = None,
    ) -> DownscalingOutput:
        # Convert single time to TimeSlice
        time: Slice | TimeSlice
        if isinstance(self.event_time, str):
            stop_time = (
                datetime.strptime(self.event_time, self.time_format)
                + timedelta(hours=3)  # half timestep to not include next time
            ).strftime(self.time_format)
            time = TimeSlice(self.event_time, stop_time)
        else:
            time = Slice(self.event_time, self.event_time + 1)

        coarse = self._single_xarray_config(loader_config.coarse)

        return self._build(
            time=time,
            lat_extent=self.lat_extent,
            lon_extent=self.lon_extent,
            loader_config=loader_config,
            requirements=requirements,
            patch=patch,
            coarse=coarse,
            fine_shape=fine_shape,
        )


@dataclass
class TimeRangeConfig(DownscalingOutputConfig):
    """
    Configuration for generating a time segment over a spatial region.

    This is the most common and flexible configuration, suitable for generating
    downscaled data over regions like CONUS, continental areas, or custom domains
    over extended time periods.

    Parameters:
        name: Unique identifier for this target (used in output filename)
        n_ens: Number of ensemble members to generate when downscaling
        save_vars: List of variable names to save to zarr output.  If None,
            all variables from the model output will be saved.
        zarr_chunks: Optional chunk sizes for zarr dimensions. If None, automatically
            calculated to target lat/lon shape <=10MB per chunk. Ensemble and time
            dimensions chunks are length 1.
        zarr_shards: Optional shard sizes for zarr dimensions. If None, defaults to
            maximum output size for a single unit of downscaling work.  This ensures
            that parallel generation tasks write to separate shards.
        max_samples_per_gpu: Number of time and/or ensemble samples to include in a
            single GPU generation. Controls memory usage and time to generate.

        time_range: Time selection specification. Can be:

            - TimeSlice: Start/stop timestamps (e.g.,
              TimeSlice(start_time="2021-01-01", stop_time="2021-12-31"))
            - Slice: Integer indices (e.g., Slice(0, 365))
            - RepeatedInterval: Repeating time pattern
        lat_extent: Latitude bounds in degrees limited to [-88, 88].
            Defaults to (-66, 70) which covers continental land masses aside
            from Antarctica.
        lon_extent: Longitude bounds in degrees [-180, 360]. Default: full extent
            of the underlying data.
    """

    time_range: TimeSlice | RepeatedInterval | Slice = field(
        default_factory=lambda: Slice(-1, 1)
    )
    lat_extent: ClosedInterval = field(
        default_factory=lambda: ClosedInterval(-66.0, 70.0)
    )
    lon_extent: ClosedInterval = field(
        default_factory=lambda: ClosedInterval(float("-inf"), float("inf"))
    )

    def __post_init__(self):
        super().__post_init__()
        if self.time_range == Slice(-1, 1):
            raise ValueError("time_range must be specified for RegionConfig.")
        enforce_lat_bounds(self.lat_extent)

    def build(
        self,
        loader_config: DataLoaderConfig,
        requirements: DataRequirements,
        patch: PatchPredictionConfig,
        fine_shape: tuple[int, int] | None = None,
    ) -> DownscalingOutput:
        coarse = self._single_xarray_config(loader_config.coarse)
        return self._build(
            time=self.time_range,
            lat_extent=self.lat_extent,
            lon_extent=self.lon_extent,
            loader_config=loader_config,
            requirements=requirements,
            patch=patch,
            coarse=coarse,
            fine_shape=fine_shape,
        )

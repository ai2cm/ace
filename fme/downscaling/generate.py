"""
Generate and save downscaled data using a trained FME model.

This module provides a flexible API for generating high-resolution downscaled
outputs from trained diffusion models. It supports:

Usage:
    python -m fme.downscaling.generate config.yaml

    # Multi-GPU
    torchrun --nproc_per_node=8 -m fme.downscaling.generate config.yaml

Output Structure:
    Each target generates a zarr file at: {output_dir}/{target_name}.zarr

    Zarr dimensions: (time, ensemble, latitude, longitude)

Example:
        /results/downscaling_run/
        ├── hurricane_landfall.zarr/
        │   ├── PRATEsfc/
        │   └── WIND10m/
        ├── miami_timeseries.zarr/
        └── conus_summer_2021.zarr/
"""

import argparse
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from itertools import product

import dacite
import numpy as np
import torch
import xarray as xr
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from fme.core import logging_utils
from fme.core.cli import prepare_directory
from fme.core.dataset.time import RepeatedInterval, TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.logging_utils import LoggingConfig
from fme.core.typing_ import Slice
from fme.core.writer import ZarrWriter

from .data import (
    BatchData,
    ClosedInterval,
    DataLoaderConfig,
    LatLonCoordinates,
    Topography,
)
from .data.config import BatchItemDatasetAdapter, XarrayEnsembleDataConfig
from .models import CheckpointModelConfig, DiffusionModel
from .predictors import (
    CascadePredictor,
    CascadePredictorConfig,
    PatchPredictionConfig,
    PatchPredictor,
)
from .requirements import DataRequirements
from .train import count_parameters

TIME_NAME = "time"
ENSEMBLE_NAME = "ensemble"
LAT_NAME = "latitude"
LON_NAME = "longitude"
DIMS = (TIME_NAME, ENSEMBLE_NAME, LAT_NAME, LON_NAME)


@dataclass
class _SliceWorkItem:
    """
    Work item specification for generation: which time and ensemble slices to process.

    This is an immutable specification of work to be done. To attach batch data,
    use the `with_batch()` class method to create a `_LoadedWorkItem`.
    """

    time_slice: slice  # times to grab from the dataset
    ens_slice: slice  # ensemble members to generate
    is_padding: bool = False  # For even GPU distribution

    def __post_init__(self):
        self.n_ens = self.ens_slice.stop - self.ens_slice.start

    @property
    def time_indices(self) -> list[int]:
        """Get list of time indices to load from the dataset."""
        sl_ = self.time_slice
        return list(range(sl_.start, sl_.stop))

    @property
    def insert_slices(self) -> dict[str, slice]:
        """Get zarr position slices for writing output."""
        return {
            TIME_NAME: self.time_slice,
            ENSEMBLE_NAME: self.ens_slice,
        }

    @classmethod
    def with_batch(
        cls, work_item: "_SliceWorkItem", batch: BatchData
    ) -> "_LoadedWorkItem":
        """
        Create a LoadedWorkItem with batch data attached.
        """
        return _LoadedWorkItem(
            time_slice=work_item.time_slice,
            ens_slice=work_item.ens_slice,
            is_padding=work_item.is_padding,
            batch=batch,
        )


@dataclass
class _LoadedWorkItem(_SliceWorkItem):
    """
    Work item with batch data attached, ready for generation.

    Created via _SliceWorkItem.with_batch() after loading data from the dataset.
    """

    batch: BatchData | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.batch is None:
            raise ValueError(
                "LoadedWorkItem must be created with batch data via with_batch()"
            )


class _SliceItemDataset:
    """
    Dataset that loads a batch of data with metadata about time
    and ensemble slices for generation and saving out to Zarr.
    """

    def __init__(
        self,
        slice_items: list[_SliceWorkItem],
        dataset: BatchItemDatasetAdapter,
        topography: Topography,
    ) -> None:
        self.slice_items = slice_items
        self.dataset = dataset
        self.topography = topography
        self._dtype = None

    def __len__(self) -> int:
        return len(self.slice_items)

    def __getitem__(self, idx: int) -> tuple[_LoadedWorkItem, Topography]:
        work_spec = self.slice_items[idx]
        data_items = [self.dataset[i] for i in work_spec.time_indices]
        batch = BatchData.from_sequence(data_items)
        loaded_item = _SliceWorkItem.with_batch(work_spec, batch)
        return loaded_item, self.topography

    @property
    def max_output_shape(self):
        first_item = self.slice_items[0]
        n_times = first_item.time_slice.stop - first_item.time_slice.start
        n_ensembles = first_item.ens_slice.stop - first_item.ens_slice.start
        spatial = self.topography.data.shape
        return (n_times, n_ensembles, *spatial)

    @property
    def dtype(self) -> torch.dtype:
        if self._dtype is not None:
            return self._dtype

        sample_item = self.dataset[0]
        sample_tensor = next(iter(sample_item.data.values()))
        self._dtype = sample_tensor.dtype
        return self._dtype


class OutputTarget:
    """
    Container for a single output generation target.

    Encapsulates all data and metadata needed to generate downscaled outputs
    for a specific region, time range, and ensemble configuration.

    Attributes:
        name: Identifier for this target (used as output filenames)
        save_vars: List of variable names to save to zarr
        all_times: DataArray of the full time coordinate of the target output
        n_ens: Total number of ensemble members to generate
        n_items_per_gpu: Number of time×ensemble items per GPU batch (memory control)
        data: GriddedData containing the input coarse data and loader
        patch: Configuration for patching large domains
        chunks: Zarr chunk sizes for each dimension
        shards: Zarr shard sizes (optional, for grouping chunks)
        dims: Dimension names including ensemble (time, ensemble, lat, lon)
    """

    def __init__(
        self,
        name: str,
        save_vars: list[str],
        all_times: xr.DataArray,
        n_ens: int,
        n_items_per_gpu: int,
        data: DataLoader,
        patch: PatchPredictionConfig,
        chunks: dict[str, int],
        shards: dict[str, int] | None,
        dims: tuple[str, ...] = DIMS,
    ) -> None:
        self.name = name
        self.save_vars = save_vars
        self.all_times = all_times
        self.n_ens = n_ens
        self.n_items_per_gpu = n_items_per_gpu
        self.data = data
        self.patch = patch
        self.chunks = chunks
        self.shards = shards
        self.dims = dims

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
                    self.all_times,
                    np.array(ensemble),
                    latlon_coords.lat.numpy(),
                    latlon_coords.lon.numpy(),
                ],
            )
        )
        dims = tuple(coords.keys())

        return ZarrWriter(
            path=f"{output_dir}/{self.name}.zarr",
            dims=dims,
            coords=coords,
            data_vars=self.save_vars,
            chunks=self.chunks,
            shards=self.shards,
        )


def _generate_slices(total: int, step: int) -> list[slice]:
    """Generate list of slices to cover a total range using a fixed step size."""
    slices = []
    start = 0
    while start < total:
        end = min(start + step, total)
        slices.append(slice(start, end))
        start = end
    return slices


def _get_work_items(
    n_times: int, n_ens: int, n_items_per_gpu: int, dist: Distributed | None = None
) -> list[_SliceWorkItem]:
    """
    Create work items for generation based on time and ensemble slices.

    Args:
        n_times: Number of time steps in the data
        n_ens: Total number of ensemble members to generate
        n_items_per_gpu: Number of time×ensemble items per GPU batch
        dist: Distributed instance for inferring padding work items (optional)
    """
    # 6 ens, 4 times, 4 items per gpu
    # item 0: (time=[0], ens=[0, 1, 2, 3])
    # item 1: (time=[0], ens=[4, 5])
    # item 2:
    work_items: list[_SliceWorkItem] = []
    n_ens_per_slice = min(n_ens, n_items_per_gpu)
    n_time_per_slice = max(1, n_items_per_gpu // n_ens_per_slice)

    ens_slices = _generate_slices(n_ens, n_ens_per_slice)
    time_slices = _generate_slices(n_times, n_time_per_slice)

    work_items = [
        _SliceWorkItem(time_sl, ens_sl)
        for (time_sl, ens_sl) in product(time_slices, ens_slices)
    ]

    # Pad work items to evenly distribute across GPUs
    dist = dist or Distributed.get_instance()
    if dist.is_distributed():
        remainder = len(work_items) % dist.world_size
        if remainder != 0:
            n_padding = dist.world_size - remainder
            # repeat last item as padding
            padding_item = _SliceWorkItem(
                time_slice=work_items[-1].time_slice,
                ens_slice=work_items[-1].ens_slice,
                is_padding=True,
            )
            work_items.extend([padding_item] * n_padding)

    return work_items


def _total_size_mb(shape: tuple[int, ...], bytes_per_element: int) -> int:
    """Calculate total size in MB for a given shape."""
    total_elements = np.prod(shape)
    return total_elements * bytes_per_element // (1024 * 1024)


class NotReducibleError(Exception):
    """Raised when shape cannot be reduced below target size."""

    pass


def _recursive_chunksize_search(
    shape: tuple[int, ...], bytes_per_element: int, reduce_dim: int, target_mb: int = 20
) -> tuple[int, ...]:
    """
    Recursively find optimal chunk shape by halving dimensions.

    Strategy:
    - Reduces dimensions in order: time, ensemble, lat/lon (alternating)
    - Each dimension is repeatedly halved until moving to the next
    - Spatial dimensions (lat/lon) alternate for balanced subdivisions
    - Raises error if target size cannot be achieved

    Args:
        shape: Current shape to evaluate
        bytes_per_element: Size of data type in bytes
        reduce_dim: Index of dimension to try reducing (0-3)
        target_mb: Target size in MB (default: 20)

    Returns:
        Optimized chunk shape meeting target size

    Raises:
        NotReducibleError: If all dimensions exhausted but still over target
    """
    if _total_size_mb(shape, bytes_per_element) <= target_mb:
        return shape
    elif bytes_per_element / 1024**2 > target_mb:
        raise NotReducibleError(
            f"Element size {bytes_per_element} bytes exceeds target chunk size "
            f"{target_mb}MB."
        )

    # Try to halve the current dimension
    reduce_dim_size = shape[reduce_dim] // 2

    if reduce_dim_size < 1:
        # Current dimension can't be reduced further, move to next
        reduce_dim += 1
        return _recursive_chunksize_search(
            shape, bytes_per_element, reduce_dim, target_mb
        )

    # Successfully halved the dimension, update shape
    new_shape = list(shape)
    new_shape[reduce_dim] = reduce_dim_size

    # Determine next dimension to try
    next_reduce_dim = reduce_dim
    if reduce_dim == 2:
        # Alternate between lat and lon for balanced spatial chunks
        next_reduce_dim = 3
    elif reduce_dim == 3:
        next_reduce_dim = 2
    # For dimensions 0 (time) and 1 (ensemble), keep reducing the same dimension

    return _recursive_chunksize_search(
        tuple(new_shape), bytes_per_element, next_reduce_dim, target_mb
    )


def _determine_zarr_chunks(
    dims: list[str], data_shape: tuple[int, ...], bytes_per_element: int
) -> dict[str, int]:
    """
    Auto-generate zarr chunk sizes for the output data.

    Automatically determines chunk sizes targeting <=20MB per chunk by
    recursively halving dimensions until the target size is reached.

    Args:
        dims: Dimension names (time, ensemble, latitude, longitude)
        data_shape: Shape tuple matching dims
        bytes_per_element: Size of data type (e.g., 4 for float32)
    """
    if len(data_shape) != 4:
        raise ValueError(
            "Data shape must be of length 4 (time, ensemble, latitude, longitude)."
        )

    chunk_shape = _recursive_chunksize_search(
        data_shape, bytes_per_element, reduce_dim=0, target_mb=20
    )
    return dict(zip(dims, chunk_shape))


@dataclass
class OutputTargetConfig(ABC):
    """
    Base class for configuring downscaling output generation targets.

    Output targets define what data to generate, where to generate it, and how
    to save it.

    Attributes:
        name: Unique identifier for this target (used in output filename)
        n_ens: Number of ensemble members to generate when downscaling
        save_vars: List of variable names to save to zarr output
        zarr_chunks: Optional chunk sizes for zarr dimensions. If None, automatically
            calculated to target <=20MB per chunk.
        zarr_shards: Optional shard sizes for grouping chunks. Recommended
            when using many chunks with remote stores.
        coarse: Optional override for coarse input data source. If None, uses
            the data config from GenerationConfig.
        patch: Optional override for patch prediction config. If None, uses
            the default from GenerationConfig.
        n_item_per_gpu: Number of time x ensemble items to generate per GPU batch.
            Controls memory usage and time to generate.
    """

    name: str
    n_ens: int
    save_vars: list[str]
    zarr_chunks: dict[str, int] | None = None
    zarr_shards: dict[str, int] | None = None
    coarse: list[XarrayDataConfig] | None = None
    patch: PatchPredictionConfig | None = None
    n_item_per_gpu: int = 4

    @abstractmethod
    def build(
        self,
        loader_config: DataLoaderConfig,
        requirements: DataRequirements,
        patch: PatchPredictionConfig,
    ) -> OutputTarget:
        """
        Build an OutputTarget from this configuration.

        Args:
            loader_config: Base data loader configuration to modify
            requirements: Model's data requirements (variable names, etc.)
            patch: Default patch prediction configuration
        """
        pass

    @staticmethod
    def _single_xarray_config(
        coarse: list[XarrayDataConfig]
        | Sequence[XarrayDataConfig | XarrayEnsembleDataConfig],
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

    def _build(
        self,
        time: TimeSlice | RepeatedInterval | Slice,
        lat_extent: ClosedInterval,
        lon_extent: ClosedInterval,
        loader_config: DataLoaderConfig,
        requirements: DataRequirements,
        patch: PatchPredictionConfig,
        coarse: list[XarrayDataConfig],
        dist: Distributed | None = None,
    ) -> OutputTarget:
        updated_loader_config = self._replace_loader_config(
            time,
            coarse,
            lat_extent,
            lon_extent,
            loader_config,
        )
        data = updated_loader_config.build(requirements=requirements)
        xr_dataset, properties = updated_loader_config.get_xarray_dataset(
            names=requirements.coarse_names, n_timesteps=1
        )
        coords = properties.horizontal_coordinates
        if not isinstance(coords, LatLonCoordinates):
            raise ValueError(
                "Downscaling data loader only supports datasets with latlon coords."
            )
        dataset = updated_loader_config.build_batchitem_dataset(xr_dataset, properties)
        topography = updated_loader_config.build_topography(
            coords,
            requires_topography=requirements.use_fine_topography,
        )
        if topography is None:
            raise ValueError("Topography is required for downscaling generation.")

        work_items = _get_work_items(
            n_times=len(dataset),
            n_ens=self.n_ens,
            n_items_per_gpu=self.n_item_per_gpu,
        )

        slice_dataset = _SliceItemDataset(
            slice_items=work_items,
            dataset=dataset,
            topography=topography,
        )

        # each item loads its own batch, so batch_size=1
        dist = dist or Distributed.get_instance()
        loader = DataLoader(
            slice_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=loader_config.num_data_workers,
            collate_fn=lambda x: x,  # type: ignore
            drop_last=False,
            multiprocessing_context=loader_config.mp_context,
            persistent_workers=True if loader_config.num_data_workers > 0 else False,
            sampler=(
                DistributedSampler(slice_dataset, shuffle=False)
                if dist.is_distributed()
                else None
            ),
        )

        if self.zarr_chunks is None:
            chunks = _determine_zarr_chunks(
                dims=data.dims,
                data_shape=slice_dataset.max_output_shape,
                bytes_per_element=slice_dataset.dtype.element_size(),
            )
        else:
            chunks = self.zarr_chunks

        return OutputTarget(
            name=self.name,
            save_vars=self.save_vars,
            all_times=xr_dataset.all_times,
            n_ens=self.n_ens,
            n_items_per_gpu=self.n_item_per_gpu,
            data=loader,
            patch=patch,
            chunks=chunks,
            shards=self.zarr_shards,
        )


@dataclass
class EventConfig(OutputTargetConfig):
    """
    Configuration for generating a single time snapshot over a spatial region.

    Useful for capturing specific events like hurricane landfall, extreme weather
    events, or any single-timestep high-resolution snapshot of a region.

    In distributed mode, the data is replicated across GPUs since there's only
    one time step to process. All ensemble generation still happens in parallel.

    Attributes:
        event_time: Timestamp or integer index of the event. If string, must match
            time_format. Required field.
        time_format: strptime format for parsing event_time string.
            Default: "%Y-%m-%dT%H:%M:%S" (ISO 8601)
        lat_extent: Latitude bounds in degrees [-90, 90]. Default: full globe
        lon_extent: Longitude bounds in degrees [0, 360]. Default: full globe

    Example:
        >>> config = EventConfig(
        ...     name="hurricane_landfall",
        ...     event_time="2020-08-27T12:00:00",
        ...     lat_extent=ClosedInterval(25.0, 35.0),
        ...     lon_extent=ClosedInterval(270.0, 280.0),
        ...     n_ens=32,
        ...     save_vars=["PRATEsfc", "WIND10m"]
        ... )
    """

    # event_time required, but must specify default to allow subclassing
    event_time: str | int = ""
    time_format: str = "%Y-%m-%dT%H:%M:%S"
    lat_extent: ClosedInterval = field(
        default_factory=lambda: ClosedInterval(-90.0, 90.0)
    )
    lon_extent: ClosedInterval = field(
        default_factory=lambda: ClosedInterval(float("-inf"), float("inf"))
    )

    def __post_init__(self):
        if not self.event_time:
            raise ValueError("event_time must be specified for EventConfig.")

    def build(
        self,
        loader_config: DataLoaderConfig,
        requirements: DataRequirements,
        patch: PatchPredictionConfig,
    ) -> OutputTarget:
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

        coarse = self._single_xarray_config(self.coarse or loader_config.coarse)

        return self._build(
            time=time,
            lat_extent=self.lat_extent,
            lon_extent=self.lon_extent,
            loader_config=loader_config,
            requirements=requirements,
            patch=self.patch or patch,
            coarse=coarse,
        )


@dataclass
class TimeseriesConfig(OutputTargetConfig):
    """
    Configuration for generating a time series at a single spatial point.

    Useful for generating high-resolution time series at specific locations of
    interest (cities, weather stations, etc.). The point is expanded to a small
    region based on latlon_tolerance to capture the nearest grid cell(s).

    Attributes:
        time_range: Time selection specification. Can be:
            - TimeSlice: Start/stop timestamps (e.g.,
              TimeSlice("2021-01-01", "2021-02-01"))
            - Slice: Integer indices (e.g., Slice(0, 100))
            - RepeatedInterval: Repeating time pattern
            Required field.
        lat_point: Latitude of point in degrees [-90, 90]. Required field.
        lon_point: Longitude of point in degrees [0, 360]. Required field.
        latlon_tolerance: Distance in degrees to expand around point to capture
            grid cell. Default: 0.25 degrees (~27km at equator)

    Example:
        >>> config = TimeseriesConfig(
        ...     name="miami_timeseries",
        ...     time_range=TimeSlice("2021-07-01T00:00:00", "2021-08-01T00:00:00"),
        ...     lat_point=25.77,
        ...     lon_point=279.93,  # -80.07 in 0-360 convention
        ...     n_ens=16,
        ...     save_vars=["TMP2m", "PRATEsfc"]
        ... )
    """

    # time_range, lat_point, and lon_point are required, but must specify default
    # to allow subclassing
    time_range: TimeSlice | RepeatedInterval | Slice = field(
        default_factory=lambda: Slice(-1, 1)
    )
    lat_point: float = -1000.0
    lon_point: float = -1000.0
    latlon_tolerance: float = 0.25

    def __post_init__(self):
        if self.lat_point == -1000.0:
            raise ValueError("lat_point must be specified for TimeseriesConfig.")
        if self.lon_point == -1000.0:
            raise ValueError("lon_point must be specified for TimeseriesConfig.")
        if self.time_range == Slice(-1, 1):
            raise ValueError("time_range must be specified for TimeseriesConfig.")

        raise NotImplementedError(
            "TimeseriesConfig is not yet implemented because it requires more "
            "complex handling of how to make sure we still have the necessary coarse "
            "input patch (e.g., 16x16).  Just use a RegionConfig for now."
        )

    def build(
        self,
        loader_config: DataLoaderConfig,
        requirements: DataRequirements,
        patch: PatchPredictionConfig,
    ) -> OutputTarget:
        lat_extent = ClosedInterval(
            self.lat_point - self.latlon_tolerance,
            self.lat_point + self.latlon_tolerance,
        )
        lon_extent = ClosedInterval(
            self.lon_point - self.latlon_tolerance,
            self.lon_point + self.latlon_tolerance,
        )
        coarse = self._single_xarray_config(self.coarse or loader_config.coarse)

        return self._build(
            time=self.time_range,
            lat_extent=lat_extent,
            lon_extent=lon_extent,
            loader_config=loader_config,
            requirements=requirements,
            patch=self.patch or patch,
            coarse=coarse,
        )


@dataclass
class RegionConfig(OutputTargetConfig):
    """
    Configuration for generating a time series over a spatial region.

    This is the most common and flexible configuration, suitable for generating
    downscaled data over regions like CONUS, continental areas, or custom domains
    over extended time periods.

    Attributes:
        time_range: Time selection specification. Can be:
            - TimeSlice: Start/stop timestamps (e.g.,
              TimeSlice("2021-01-01", "2021-12-31"))
            - Slice: Integer indices (e.g., Slice(0, 365))
            - RepeatedInterval: Repeating time pattern
            Required field.
        lat_extent: Latitude bounds in degrees [-90, 90]. Default: full globe
        lon_extent: Longitude bounds in degrees [0, 360]. Default: full globe
    """

    time_range: TimeSlice | RepeatedInterval | Slice = field(
        default_factory=lambda: Slice(-1, 1)
    )
    lat_extent: ClosedInterval = field(
        default_factory=lambda: ClosedInterval(-90.0, 90.0)
    )
    lon_extent: ClosedInterval = field(
        default_factory=lambda: ClosedInterval(float("-inf"), float("inf"))
    )

    def __post_init__(self):
        if self.time_range == Slice(-1, 1):
            raise ValueError("time_range must be specified for RegionConfig.")

    def build(
        self,
        loader_config: DataLoaderConfig,
        requirements: DataRequirements,
        patch: PatchPredictionConfig,
    ) -> OutputTarget:
        coarse = self._single_xarray_config(self.coarse or loader_config.coarse)
        return self._build(
            time=self.time_range,
            lat_extent=self.lat_extent,
            lon_extent=self.lon_extent,
            loader_config=loader_config,
            requirements=requirements,
            patch=self.patch or patch,
            coarse=coarse,
        )


class Downscaler:
    """
    Orchestrates downscaling generation across multiple output targets.

    Each target can have different spatial extents, time ranges, and ensemble sizes.
    Generation is performed sequentially across targets.
    """

    def __init__(
        self,
        model: DiffusionModel | CascadePredictor,
        output_targets: list[OutputTarget],
        output_dir: str = ".",
    ):
        self.model = model
        self.output_targets = output_targets
        self.output_dir = output_dir

    def run_all(self):
        """Run generation for all output targets."""
        logging.info(f"Starting generation for {len(self.output_targets)} target(s)")

        for target in self.output_targets:
            # Clear GPU cache before each target
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.run_target_generation(target=target)

        logging.info("All targets completed successfully")

    def _get_generation_model(
        self,
        topography: Topography,
        target: OutputTarget,
    ) -> DiffusionModel | PatchPredictor | CascadePredictor:
        """
        Set up the model, wrapping with PatchPredictor if needed.  While models are
        probably capable of generating any domain size, we haven't tested for domains
        smaller than the model patch size, so we raise an error in that case, and prompt
        the user to use patching for larger domains because that provides better
        generations.
        """
        model_patch_shape = self.model.coarse_shape
        actual_shape = tuple(topography.data.shape)

        if model_patch_shape == actual_shape:
            # short circuit, no patching necessary
            return self.model
        elif any(
            expected > actual
            for expected, actual in zip(model_patch_shape, actual_shape)
        ):
            # we don't support generating regions smaller than the model patch size
            raise ValueError(
                f"Model coarse shape {model_patch_shape} is larger than "
                f"actual topography shape {actual_shape} for target {target.name}."
            )
        elif target.patch.needs_patch_predictor:
            # Use a patch predictor
            logging.info(f"Using PatchPredictor for target: {target.name}")
            return PatchPredictor(
                model=self.model,
                coarse_horizontal_overlap=target.patch.coarse_horizontal_overlap,
            )
        else:
            # User should enable patching
            raise ValueError(
                f"Model coarse shape {model_patch_shape} does not match "
                f"actual input shape {actual_shape} for target {target.name}, "
                "and patch prediction is not configured. Generation for larger domains "
                "requires patch prediction."
            )

    def run_target_generation(self, target: OutputTarget):
        """Execute the generation loop for this target."""
        logging.info(f"Generating downscaled outputs for target: {target.name}")

        # initialize writer and model in loop for coord info
        model = None
        writer = None
        total_batches = len(target.data)

        loaded_item: _LoadedWorkItem
        topography: Topography
        for i, (loaded_item, topography) in enumerate(target.data):
            if writer is None:
                writer = target.get_writer(
                    latlon_coords=topography.coords,
                    output_dir=self.output_dir,
                )
            if model is None:
                model = self._get_generation_model(topography=topography, target=target)

            logging.info(
                f"[{target.name}] Batch {i+1}/{total_batches}, "
                f"generating work slice {loaded_item.insert_slices} "
            )

            output = model.generate_on_batch_no_target(
                loaded_item.batch, topography=topography, n_samples=loaded_item.n_ens
            )
            if not loaded_item.is_padding:
                output_np = {k: output[k].cpu().numpy() for k in target.save_vars}
                writer.record_batch(
                    output_np, position_slices=loaded_item.insert_slices
                )

        logging.info(f"Completed generation for target: {target.name}")


@dataclass
class GenerationConfig:
    """
    Top-level configuration for downscaling generation.

    Defines the model, base data source, and one or more output targets to generate.
    Fine-resolution outputs are generated from coarse-resolution inputs without
    requiring fine-resolution target data (unlike training/evaluation).

    Each output target can specify different spatial regions, time ranges, ensemble
    sizes, and output variables. Targets are processed sequentially, with generation
    parallelized across GPUs using distributed data loading.

    Attributes:
        model: Model configuration (checkpoint or cascade predictor)
        data: Base data loader configuration. Individual targets can override
            specific aspects (time range, spatial extent) while inheriting the
            base configuration.
        output_dir: Directory for saving generated zarr files and logs
        output_targets: List of output specifications (EventConfig, TimeseriesConfig,
            or RegionConfig). Each target generates a separate zarr file.
        logging: Logging configuration (file, screen, wandb)
        patch: Default patch prediction configuration. Individual targets can override
            this if needed for their specific domain size.

    Example:
        >>> config = GenerationConfig(
        ...     model=CheckpointModelConfig(checkpoint_path="/path/to/model.ckpt"),
        ...     data=DataLoaderConfig(
        ...         coarse=[XarrayDataConfig(
        ...             data_path="/data/coarse",
        ...             file_pattern="coarse.zarr",
        ...             engine="zarr"
        ...         )],
        ...         batch_size=8,
        ...         num_data_workers=4,
        ...         strict_ensemble=True,
        ...     ),
        ...     output_dir="/results/generation",
        ...     output_targets=[
        ...         RegionConfig(
        ...             name="conus_summer",
        ...             time_range=TimeSlice("2021-06-01", "2021-09-01"),
        ...             lat_extent=ClosedInterval(22.0, 50.0),
        ...             lon_extent=ClosedInterval(227.0, 299.0),
        ...             n_ens=8,
        ...             save_vars=["PRATEsfc", "TMP2m"]
        ...         )
        ...     ],
        ...     logging=LoggingConfig(log_to_screen=True, log_to_wandb=False),
        ...     patch=PatchPredictionConfig(divide_generation=True)
        ... )
    """

    model: CheckpointModelConfig | CascadePredictorConfig
    data: DataLoaderConfig
    output_dir: str
    output_targets: list[OutputTargetConfig]
    logging: LoggingConfig
    patch: PatchPredictionConfig = field(default_factory=PatchPredictionConfig)

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.output_dir, log_filename)

    def configure_wandb(self, resumable: bool = False, **kwargs):
        config = to_flat_dict(asdict(self))
        env_vars = logging_utils.retrieve_env_vars()
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resumable=resumable, **kwargs
        )

    def build(self) -> Downscaler:
        targets = [
            target_cfg.build(
                loader_config=self.data,
                requirements=self.model.data_requirements,
                patch=self.patch,
            )
            for target_cfg in self.output_targets
        ]
        model = self.model.build()
        return Downscaler(model=model, output_targets=targets)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    generation_config: GenerationConfig = dacite.from_dict(
        data_class=GenerationConfig,
        data=config,
        config=dacite.Config(strict=True),
    )
    prepare_directory(generation_config.output_dir, config)

    generation_config.configure_logging(log_filename="out.log")
    logging_utils.log_versions()
    beaker_url = logging_utils.log_beaker_url()
    generation_config.configure_wandb(resumable=True, notes=beaker_url)

    logging.info("Starting downscaling generation...")
    downscaler = generation_config.build()
    logging.info(f"Number of parameters: {count_parameters(downscaler.model.modules)}")
    downscaler.run_all()


def parse_args():
    parser = argparse.ArgumentParser(description="Downscaling generation script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)

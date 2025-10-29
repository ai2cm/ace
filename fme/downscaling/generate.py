"""
Generate and save downscaled data using a trained FME model.

This module provides a flexible API for generating high-resolution downscaled
outputs from trained diffusion models. It supports:

Usage:
    python -m fme.downscaling.generate config.yaml

    # Multi-GPU
    torchrun --nproc_per_node=8 -m fme.downscaling.generate config.yaml

Example YAML Configuration:
    ```yaml
    # Model configuration
    model:
      checkpoint_path: /path/to/trained/model.ckpt
      fine_topography_path: /path/to/fine/topography.zarr  # Optional

    # Base data configuration (inherited by all targets)
    data:
      coarse:
        - data_path: /data/coarse
          file_pattern: coarse_*.zarr
          engine: zarr
      batch_size: 8
      num_data_workers: 4
      strict_ensemble: true

    # Output directory for all generated zarr files
    output_dir: /results/downscaling_run

    # Logging configuration
    logging:
      project: downscaling-generation
      log_to_screen: true
      log_to_file: true
      log_to_wandb: false

    # Default patch configuration for large domains
    patch:
      divide_generation: true
      coarse_horizontal_overlap: 1

    # Output targets - can mix different types
    output_targets:
      # 1. Event: Single time snapshot over a region
      - name: hurricane_landfall
        event_time: "2020-08-27T12:00:00"
        lat_extent:
          lower: 25.0
          upper: 35.0
        lon_extent:
          lower: 270.0
          upper: 280.0
        n_ens: 32
        save_vars:
          - PRATEsfc
          - WIND10m
        n_item_per_gpu: 4

      # 2. Time series: Point location over time
      - name: miami_timeseries
        time_range:
          start: "2021-07-01T00:00:00"
          stop: "2021-08-01T00:00:00"
        lat_point: 25.77
        lon_point: 279.93  # -80.07 in 0-360 convention
        n_ens: 16
        save_vars:
          - TMP2m
          - PRATEsfc

      # 3. Region: Full spatiotemporal generation
      - name: conus_summer_2021
        time_range:
          start: "2021-06-01T00:00:00"
          stop: "2021-09-01T00:00:00"
        lat_extent:
          lower: 22.0
          upper: 50.0
        lon_extent:
          lower: 227.0
          upper: 299.0
        n_ens: 8
        save_vars:
          - PRATEsfc
          - TMP2m
          - eastward_wind_at_ten_meters
        # Optional: specify zarr chunking
        zarr_chunks:
          time: 1
          ensemble: 1
          latitude: 448
          longitude: 1152
        zarr_shards:
          time: 8
          ensemble: 8
          latitude: 448
          longitude: 1152
        # Optional: override default patch config for this target
        patch:
          divide_generation: false
    ```

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

Memory Management:
    - n_item_per_gpu: Controls GPU memory by limiting time×ensemble items per batch
    - Decrease if encountering OOM errors
    - Typical values: 2-8 depending on domain size and GPU memory
"""

import argparse
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta

import dacite
import numpy as np
import torch
import xarray as xr
import yaml

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
    GriddedData,
    LatLonCoordinates,
)
from .data.config import XarrayEnsembleDataConfig
from .models import CheckpointModelConfig, DiffusionModel
from .predictors import (
    CascadePredictor,
    CascadePredictorConfig,
    PatchPredictionConfig,
    PatchPredictor,
)
from .requirements import DataRequirements
from .train import count_parameters


class OutputTarget:
    """
    Container for a single output generation target.

    Encapsulates all data and metadata needed to generate downscaled outputs
    for a specific region, time range, and ensemble configuration.

    Attributes:
        name: Identifier for this target (used as output filenames)
        save_vars: List of variable names to save to zarr
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
        n_ens: int,
        n_items_per_gpu: int,
        data: GriddedData,
        patch: PatchPredictionConfig,
        chunks: dict[str, int],
        shards: dict[str, int] | None,
    ) -> None:
        self.name = name
        self.save_vars = save_vars
        self.n_ens = n_ens
        self.n_items_per_gpu = n_items_per_gpu
        self.data = data
        self.patch = patch
        self.chunks = chunks
        self.shards = shards

        self.dims = list(data.dims)
        self.dims.insert(1, "ensemble")  # insert ensemble dim after time

    def get_writer(
        self, latlon_coords: LatLonCoordinates, output_dir: str
    ) -> ZarrWriter:
        """
        Create a ZarrWriter for this target.

        Args:
            latlon_coords: High-resolution spatial coordinates for outputs
            output_dir: Directory to store output zarr file

        Returns:
            Configured ZarrWriter for incremental writes
        """
        ensemble = list(range(self.n_ens))
        coords = dict(
            zip(
                self.dims,
                [
                    self.data.all_times.to_numpy(),
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

    def _get_time_slice(self, time: xr.DataArray) -> slice:
        """Convert time coordinate to zarr array index slice."""
        start_time = time[0].item()
        end_time = time[-1].item()
        if start_time == end_time:
            time_idx = self.data.all_times.get_loc(start_time)
            time_slice = slice(time_idx, time_idx + 1)
        else:
            time_idxs = self.data.all_times.get_indexer(
                {"time": [start_time, end_time]}
            )
            time_slice = slice(time_idxs[0], time_idxs[1] + 1)

        return time_slice

    def _get_ens_slices(self) -> list[slice]:
        """
        Divide ensemble members into GPU-friendly slices.

        Returns list of slices based on n_items_per_gpu to control memory usage.
        """
        ens_start = 0
        ens_slices = []
        while ens_start < self.n_ens:
            ens_end = min(ens_start + self.n_items_per_gpu, self.n_ens)
            ens_slices.append(slice(ens_start, ens_end))
            ens_start = ens_end

        return ens_slices

    def get_insert_slices(self, data: BatchData) -> list[dict[str, slice]]:
        """
        Calculate zarr insertion slices for a batch of data.

        For each time batch, generates multiple ensemble slices based on
        n_items_per_gpu to control memory usage during generation.

        Args:
            data: Batch containing time coordinates

        Returns:
            List of dimension-to-slice mappings for zarr writes
        """
        time_slice = self._get_time_slice(data.time)
        ens_slices = self._get_ens_slices()
        # Always storing the entire spatial region
        lat_slice = slice(None)
        lon_slice = slice(None)

        return [
            dict(zip(self.dims, (time_slice, ens_slice, lat_slice, lon_slice)))
            for ens_slice in ens_slices
        ]


@dataclass
class OutputTargetConfig(ABC):
    """
    Base class for configuring downscaling output generation targets.

    Output targets define what data to generate, where to generate it, and how
    to save it. Different subclasses support different spatiotemporal patterns:
    - EventConfig: Single time snapshot over a spatial region
    - TimeseriesConfig: Time series at a single point
    - RegionConfig: Time series over a spatial region

    Attributes:
        name: Unique identifier for this target (used in output filename)
        n_ens: Number of ensemble members to generate
        save_vars: List of variable names to save to zarr output
        zarr_chunks: Optional chunk sizes for zarr dimensions. If None, automatically
            calculated to target 2-20MB per chunk.
        zarr_shards: Optional shard sizes for grouping chunks. Recommended when
            total chunks exceed ~1000 for better I/O performance.
        coarse: Optional override for coarse input data source. If None, uses
            the data config from GenerationConfig.
        patch: Optional override for patch prediction config. If None, uses
            the default from GenerationConfig.
        n_item_per_gpu: Number of time×ensemble items to generate per GPU batch.
            Controls memory usage - decrease if encountering OOM errors.
            Default: 4
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

        Returns:
            Configured OutputTarget ready for generation
        """
        pass

    @staticmethod
    def _single_xarray_config(
        coarse: list[XarrayDataConfig]
        | Sequence[XarrayDataConfig | XarrayEnsembleDataConfig],
    ) -> list[XarrayDataConfig]:
        # TODO: We really only should be supporting a single xarray config
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
        dist: Distributed,
    ) -> DataLoaderConfig:
        # return a new DataloaderConfig with replaced values
        new_coarse = [replace(coarse[0], subset=time)]

        local_batch_size = dist.local_batch_size(loader_config.batch_size)
        n_ens = self.n_ens
        n_items_per_gpu = self.n_item_per_gpu

        n_ens_per_slice = min(n_ens, n_items_per_gpu)
        n_time_per_slice = max(1, n_items_per_gpu // n_ens_per_slice)
        local_batch_size = min(local_batch_size, n_time_per_slice)
        new_batch_size = dist.world_size * local_batch_size

        # TODO: log the replacements for debugging
        new_loader_config = replace(
            loader_config,
            coarse=new_coarse,
            lat_extent=lat_extent,
            lon_extent=lon_extent,
            batch_size=new_batch_size,
        )
        return new_loader_config

    def _get_chunks(
        self, dims: list[str], data_shape: tuple[int, ...], bytes_per_element: int
    ) -> dict[str, int]:
        """
        Calculate optimal zarr chunk sizes for the output data.

        Automatically determines chunk sizes targeting 2-20MB per chunk by
        recursively halving dimensions until the target size is reached.

        Strategy:
        - Reduces dimensions in order: time, ensemble, lon/lat (alternating)
        - Each dimension is repeatedly halved until moving to the next
        - Spatial dimensions (lat/lon) alternate for balanced subdivisions
        - Raises error if target size cannot be achieved

        Args:
            dims: Dimension names (time, ensemble, latitude, longitude)
            data_shape: Shape tuple matching dims
            bytes_per_element: Size of data type (e.g., 4 for float32)

        Returns:
            Dictionary mapping dimension names to chunk sizes

        Raises:
            ValueError: If data_shape is not 4D or cannot be reduced to target size
        """
        if len(data_shape) != 4:
            raise ValueError(
                "Data shape must be of length 4 (time, ensemble, latitude, longitude)."
            )

        def _total_size_mb(shape: tuple[int, ...], bytes_per_element: int) -> int:
            """Calculate total size in MB for a given shape."""
            total_elements = 1
            for dim_size in shape:
                total_elements *= dim_size
            return total_elements * bytes_per_element // (1024 * 1024)

        class NotReducibleError(Exception):
            """Raised when shape cannot be reduced below target size."""

            pass

        def _recursive_chunksize_search(
            shape: tuple[int, ...], bytes_per_element: int, reduce_dim: int
        ) -> tuple[int, ...]:
            """
            Recursively find optimal chunk shape by halving dimensions.

            Args:
                shape: Current shape to evaluate
                bytes_per_element: Size of data type in bytes
                reduce_dim: Index of dimension to try reducing (0-3)

            Returns:
                Optimized chunk shape

            Raises:
                NotReducibleError: If all dimensions exhausted but still over target
            """
            if _total_size_mb(shape, bytes_per_element) <= 20:
                return shape

            reduce_dim_size = shape[reduce_dim] // 2
            if reduce_dim_size < 1:
                # Current dimension can't be reduced further, move to next
                reduce_dim += 1
                if reduce_dim >= len(shape):
                    raise NotReducibleError()
            elif reduce_dim == 2:
                # Switch between lat and lon reductions for balanced spatial chunks
                reduce_dim = 3
            elif reduce_dim == 3:
                reduce_dim = 2

            new_shape = list(shape)
            new_shape[reduce_dim] = reduce_dim_size

            return _recursive_chunksize_search(
                tuple(new_shape), bytes_per_element, reduce_dim
            )

        chunk_shape = _recursive_chunksize_search(
            data_shape, bytes_per_element, reduce_dim=0
        )
        return dict(zip(dims, chunk_shape))

    def _build(
        self,
        time: TimeSlice | RepeatedInterval | Slice,
        lat_extent: ClosedInterval,
        lon_extent: ClosedInterval,
        loader_config: DataLoaderConfig,
        requirements: DataRequirements,
        patch: PatchPredictionConfig,
        coarse: list[XarrayDataConfig],
    ) -> OutputTarget:
        updated_loader_config = self._replace_loader_config(
            time,
            coarse,
            lat_extent,
            lon_extent,
            loader_config,
            Distributed.get_instance(),
        )
        data = updated_loader_config.build(requirements=requirements)

        sample_batch: BatchData = next(iter(data.loader))
        sample_tensor: torch.Tensor = next(iter(sample_batch.data.values()))

        if self.zarr_chunks is None:
            chunks = self._get_chunks(
                dims=data.dims,
                data_shape=tuple(sample_tensor.shape),
                bytes_per_element=sample_tensor.element_size(),
            )
        else:
            chunks = self.zarr_chunks

        return OutputTarget(
            name=self.name,
            save_vars=self.save_vars,
            n_ens=self.n_ens,
            n_items_per_gpu=self.n_item_per_gpu,
            data=data,
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

    def _adjust_batchsize_if_distributed(
        self,
        coarse: list[XarrayDataConfig],
        loader_config: DataLoaderConfig,
        dist: Distributed,
    ) -> tuple[list[XarrayDataConfig], DataLoaderConfig]:
        # For single events there will not be enough samples to distribute.
        # Repeat so each GPU gets one sample.
        if dist.is_distributed():
            batch_size = dist.world_size
            coarse = [replace(coarse[0], n_repeats=batch_size)]
            loader_config = replace(
                loader_config,
                batch_size=batch_size,
            )
        return coarse, loader_config

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
                + timedelta(hours=12)
            ).strftime(self.time_format)
            time = TimeSlice(self.event_time, stop_time)
        else:
            time = Slice(self.event_time, self.event_time + 1)

        coarse = self._single_xarray_config(self.coarse or loader_config.coarse)
        coarse, loader_config = self._adjust_batchsize_if_distributed(
            coarse, loader_config, Distributed.get_instance()
        )

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
    time_range: TimeSlice | RepeatedInterval | Slice = Slice(-1, 1)
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

    time_range: TimeSlice | RepeatedInterval | Slice = Slice(-1, 1)
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


class ZarrGenerator:
    """
    Handles generation and writing of downscaled outputs for a single target.

    Responsible for:
    - Setting up the model (with PatchPredictor if needed)
    - Creating the ZarrWriter
    - Running the generation loop over time batches and ensemble slices
    - Writing outputs to zarr incrementally
    """

    def __init__(
        self,
        target: OutputTarget,
        model: DiffusionModel | CascadePredictor,
        output_dir: str,
    ):
        self.target = target
        self.base_model = model
        self.output_dir = output_dir
        self.dist = Distributed.get_instance()

    def _setup_model(self) -> DiffusionModel | PatchPredictor | CascadePredictor:
        """Set up the model, wrapping with PatchPredictor if needed."""
        if self.target.patch.needs_patch_predictor:
            logging.info(f"Using PatchPredictor for target: {self.target.name}")
            return PatchPredictor(
                model=self.base_model,
                coarse_horizontal_overlap=self.target.patch.coarse_horizontal_overlap,
            )
        return self.base_model

    def run(self):
        """Execute the generation loop for this target."""
        logging.info(f"Generating downscaled outputs for target: {self.target.name}")

        model = self._setup_model()
        writer = None
        total_batches = len(self.target.data.loader)

        for i, (data, topography) in enumerate(self.target.data.loader):
            # Initialize writer on first batch
            if writer is None:
                writer = self.target.get_writer(
                    latlon_coords=topography.coords,
                    output_dir=self.output_dir,
                )

            # Calculate insert slices for this batch's time
            insert_slices = self.target.get_insert_slices(data)

            # Generate each ensemble slice for this time batch
            for ens_idx, insert_slice in enumerate(insert_slices):
                n_ens_sl = insert_slice["ensemble"]
                n_ens = n_ens_sl.stop - n_ens_sl.start

                logging.info(
                    f"[{self.target.name}] Batch {i+1}/{total_batches}, "
                    f"Ensemble slice {ens_idx+1}/{len(insert_slices)} "
                    f"(generating {n_ens} members)"
                )

                output = model.generate_on_batch_no_target(
                    data, topography=topography, n_samples=n_ens
                )
                output_np = {k: output[k].cpu().numpy() for k in self.target.save_vars}
                writer.record_batch(output_np, position_slices=insert_slice)

        logging.info(f"Completed generation for target: {self.target.name}")


class Downscaler:
    """
    Orchestrates downscaling generation across multiple output targets.

    Each target can have different spatial extents, time ranges, and ensemble sizes.
    Generation is performed sequentially across targets, with GPU memory cleared
    between targets.
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

    def run(self):
        """Run generation for all output targets."""
        logging.info(f"Starting generation for {len(self.output_targets)} target(s)")

        for target in self.output_targets:
            # Clear GPU cache before each target
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Create generator for this target and run
            generator = ZarrGenerator(
                target=target,
                model=self.model,
                output_dir=self.output_dir,
            )
            generator.run()

        logging.info("All targets completed successfully")


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
    downscaler.run()


def parse_args():
    parser = argparse.ArgumentParser(description="Downscaling generation script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)

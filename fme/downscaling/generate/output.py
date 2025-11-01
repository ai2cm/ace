from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, DistributedSampler

from fme.core.dataset.time import RepeatedInterval, TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.distributed import Distributed
from fme.core.typing_ import Slice
from fme.core.writer import ZarrWriter

from ..data import ClosedInterval, DataLoaderConfig, LatLonCoordinates
from ..data.config import XarrayEnsembleDataConfig
from ..predictors import PatchPredictionConfig
from ..requirements import DataRequirements
from .constants import DIMS
from .work_items import SliceItemDataset, get_work_items
from .zarr_utils import determine_zarr_chunks


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

        work_items = get_work_items(
            n_times=len(dataset),
            n_ens=self.n_ens,
            n_items_per_gpu=self.n_item_per_gpu,
        )

        slice_dataset = SliceItemDataset(
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
            collate_fn=lambda x: x[0],  # type: ignore
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
            # Get element size from dtype by creating a dummy tensor
            element_size = torch.tensor([], dtype=slice_dataset.dtype).element_size()
            chunks = determine_zarr_chunks(
                dims=DIMS,
                data_shape=slice_dataset.max_output_shape,
                bytes_per_element=element_size,
            )
        else:
            chunks = self.zarr_chunks

        return OutputTarget(
            name=self.name,
            save_vars=self.save_vars,
            all_times=xr_dataset.sample_start_times,
            n_ens=self.n_ens,
            n_items_per_gpu=self.n_item_per_gpu,
            data=loader,
            patch=patch,
            chunks=chunks,
            shards=self.zarr_shards,
            dims=DIMS,
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

import dataclasses
from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.concat import XarrayConcat
from fme.core.dataset.dataset import DatasetABC
from fme.core.dataset.merged import MergeNoConcatDatasetConfig
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.device import using_gpu
from fme.core.distributed import Distributed
from fme.downscaling.data.datasets import (
    BatchData,
    BatchItemDatasetAdapter,
    ContiguousDistributedSampler,
    FineCoarsePairedDataset,
    GriddedData,
    HorizontalSubsetDataset,
    PairedBatchData,
    PairedGriddedData,
    PairedVideoBatchData,
    PairedVideoGriddedData,
    VideoBatchItemDatasetAdapter,
    VideoFineCoarsePairedDataset,
)
from fme.downscaling.data.utils import (
    ClosedInterval,
    adjust_fine_coord_range,
    find_roll_anchor,
    find_roll_anchor_from_interval,
    get_latlon_coords_from_properties,
    roll_lon_coords,
)
from fme.downscaling.requirements import DataRequirements


def _roll_lons_to_extent_convention(
    coarse_lon: torch.Tensor,
    fine_lon: torch.Tensor,
    lon_extent: ClosedInterval,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Roll coarse and fine lon coords into the convention of lon_extent so that
    adjust_fine_coord_range can align fine and coarse subselection for domains
    crossing the prime meridian (lon_start < 0 or lon_stop > 360). No-op for
    in-range extents.

    The fine roll is anchored one half-coarse-spacing before lon_start so the
    fine half-cells below the first coarse grid point remain accessible.
    """
    lon_start, _ = lon_extent.finite_values
    coarse_roll = find_roll_anchor_from_interval(coarse_lon, lon_extent)
    rolled_coarse_lon = roll_lon_coords(coarse_lon, coarse_roll, lon_start)

    if coarse_roll > 0 and len(rolled_coarse_lon) >= 2:
        coarse_spacing = float((rolled_coarse_lon[1] - rolled_coarse_lon[0]).item())
        fine_anchor = lon_start - coarse_spacing / 2
        fine_roll = find_roll_anchor(fine_lon, fine_anchor % 360.0)
        rolled_fine_lon = roll_lon_coords(fine_lon, fine_roll, fine_anchor)
    else:
        rolled_fine_lon = fine_lon

    return rolled_coarse_lon, rolled_fine_lon


def _build_aligned_subset_pair(
    dataset_fine: XarrayConcat,
    properties_fine: DatasetProperties,
    dataset_coarse: XarrayConcat,
    properties_coarse: DatasetProperties,
    lat_extent: ClosedInterval,
    lon_extent: ClosedInterval,
) -> tuple[HorizontalSubsetDataset, HorizontalSubsetDataset]:
    """Subset fine and coarse datasets so their selections align exactly.

    The coarse dataset is subset with the requested lat/lon extent. The fine
    extent is adjusted (adjust_fine_coord_range) so the fine subselection lines up
    with the coarse one; both grids are first rolled into the extent's longitude
    convention so this holds for prime-meridian-crossing domains as well (see
    _roll_lons_to_extent_convention).
    """
    coarse_coords = get_latlon_coords_from_properties(properties_coarse)
    fine_coords = get_latlon_coords_from_properties(properties_fine)

    rolled_coarse_lon, rolled_fine_lon = _roll_lons_to_extent_convention(
        coarse_lon=coarse_coords.lon,
        fine_lon=fine_coords.lon,
        lon_extent=lon_extent,
    )
    fine_lat_extent = adjust_fine_coord_range(
        lat_extent,
        full_coarse_coord=coarse_coords.lat,
        full_fine_coord=fine_coords.lat,
    )
    fine_lon_extent = adjust_fine_coord_range(
        lon_extent,
        full_coarse_coord=rolled_coarse_lon,
        full_fine_coord=rolled_fine_lon,
    )

    dataset_fine_subset = HorizontalSubsetDataset(
        dataset_fine,
        properties=properties_fine,
        lat_interval=fine_lat_extent,
        lon_interval=fine_lon_extent,
    )
    dataset_coarse_subset = HorizontalSubsetDataset(
        dataset_coarse,
        properties=properties_coarse,
        lat_interval=lat_extent,
        lon_interval=lon_extent,
    )
    return dataset_fine_subset, dataset_coarse_subset


def enforce_lat_bounds(lat: ClosedInterval):
    if lat.start < -88.0 or lat.stop > 88.0:
        raise ValueError(
            "Latitude bounds must be within +/-88 degrees, "
            f"got {lat.start} to {lat.stop}."
            "This is enforced because the 3 km X-SHiELD dataset "
            "does not have 32 fine grid midpoints between the last two "
            "coarse latitude midpoints of the 100 km dataset, which breaks "
            "the assumption used for subsetting fine grid latitudes."
        )


@dataclasses.dataclass
class XarrayEnsembleDataConfig:
    """
    Configuration for an ensemble dataset.
    This config's expand method returns a sequence of xarray datasets, each
    with the same data_config, where each individual dataset is an ensemble member
    selected from the ensemble dimension.

    Parameters:
        data_config: XarrayDataConfig for the dataset.
        ensemble_dim: Name of the ensemble dimension in the dataset.
        n_ensemble_members: Number of ensemble members to load. They will be taken
            in order from index 0 of the ensemble_dim.
    """

    data_config: XarrayDataConfig
    ensemble_dim: str
    n_ensemble_members: int

    def __post_init__(self):
        if self.n_ensemble_members <= 0:
            raise ValueError(
                f"n_ensemble_members must be > 0, got {self.n_ensemble_members}"
            )
        if self.ensemble_dim in self.data_config.isel:
            raise ValueError(
                f"Ensemble dimension {self.ensemble_dim} cannot be in the "
                "base data_config.isel"
            )

    def expand(self) -> list[XarrayDataConfig]:
        configs = []
        for i in range(self.n_ensemble_members):
            configs.append(
                dataclasses.replace(
                    self.data_config,
                    isel={self.ensemble_dim: i},
                )
            )
        return configs

    @property
    def zarr_engine_used(self) -> bool:
        return self.data_config.zarr_engine_used


def build_from_config_sequence(
    configs: Sequence[
        XarrayDataConfig | XarrayEnsembleDataConfig | MergeNoConcatDatasetConfig
    ],
    names: Sequence[str],
    n_timesteps: IntSchedule,
    strict_ensemble: bool,
) -> tuple[XarrayConcat, DatasetProperties]:
    """Build XarrayConcat and properties from a mix of xarray and merge configs."""
    expanded: list[XarrayDataConfig | MergeNoConcatDatasetConfig] = []
    for config in configs:
        if isinstance(config, XarrayEnsembleDataConfig):
            expanded.extend(config.expand())
        else:
            expanded.append(config)
    datasets: list[DatasetABC] = []
    properties: DatasetProperties | None = None
    for config in expanded:
        ds, prop = config.build(names, n_timesteps)
        datasets.append(ds)
        if properties is None:
            properties = prop
        else:
            properties.update(prop, strict=strict_ensemble)
    if properties is None:
        raise ValueError("At least one dataset must be provided.")
    return XarrayConcat(datasets, strict=strict_ensemble), properties


def _full_configs(
    configs: Sequence[
        XarrayDataConfig | MergeNoConcatDatasetConfig | XarrayEnsembleDataConfig
    ],
) -> list[XarrayDataConfig | MergeNoConcatDatasetConfig]:
    """Expands XarrayEnsembleDataConfig to multiple XarrayDataConfig;
    other configs are unchanged.
    """
    all_configs: list[XarrayDataConfig | MergeNoConcatDatasetConfig] = []
    for config in configs:
        if isinstance(config, XarrayEnsembleDataConfig):
            all_configs += config.expand()
        else:
            all_configs.append(config)
    return all_configs


@dataclasses.dataclass
class DataLoaderConfig:
    """
    Configuration for loading downscaling data for generation.
    Input coarse dataset will be processed into batches, usually with
    a horizontal extent to define a portion of the full domain for use in
    generation.
    If the model requires topography, the dataset to use should be specified
    in the `topography` field. Topography data may be at higher resolution than
    the data, e.g. when fine topography is loaded as an input.

    Args:
        coarse: The dataset configuration. May be a sequence of
            XarrayDataConfig, XarrayEnsembleDataConfig, or
            MergeNoConcatDatasetConfig.
        batch_size: The batch size to use for the dataloader.
        num_data_workers: The number of data workers to use for the dataloader.
            (For multi-GPU runtime, it's the number of workers per GPU.)
        strict_ensemble: Whether to enforce that the datasets to be concatened
            have the same dimensions and coordinates.
        topography: Deprecated field for specifying the topography dataset.
            StaticInput data are expected to be stored and serialized within a
            model through the Trainer build process.
        lat_extent: The latitude extent to use for the dataset specified in
            degrees, limited to (-88.0, 88.0). The extent is inclusive, so the start and
            stop values are included in the extent. Defaults to [-66, 70] which
            covers continental land masses aside from Antarctica.
        lon_extent: The longitude extent to use for the dataset specified in
            degrees (0, 360). The extent is inclusive, so the start and
            stop values are included in the extent.
        repeat: The number of times to repeat the underlying xarray dataset
            time dimension.  Useful to include longer sequences of small
            data for testing.
        drop_last: Use drop_last option in sampler. Defaults to False. If True,
            drop the last samples required to have even batch sizes across ranks.
            If false, pad with extra samples to make ranks have the same size batches.
    """

    coarse: Sequence[
        XarrayDataConfig | XarrayEnsembleDataConfig | MergeNoConcatDatasetConfig
    ]
    batch_size: int
    num_data_workers: int
    strict_ensemble: bool
    topography: str | None = None
    lat_extent: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(-66, 70)
    )
    lon_extent: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(float("-inf"), float("inf"))
    )
    repeat: int = 1
    drop_last: bool = False

    def __post_init__(self):
        enforce_lat_bounds(self.lat_extent)
        if self.topography is not None:
            raise ValueError(
                "The `topography` field on DataLoaderConfig is deprecated and will be "
                "removed in a future release. `StaticInputs` are now stored within "
                " the model when it is first built and trained."
            )

    @property
    def full_config(self) -> Sequence[XarrayDataConfig | MergeNoConcatDatasetConfig]:
        return _full_configs(self.coarse)

    @property
    def mp_context(self):
        context = None
        if self.num_data_workers == 0:
            return None
        for config in self.full_config:
            if config.zarr_engine_used is True:
                context = "forkserver"
                break
        return context

    def _repeat_if_requested(self, dataset: XarrayConcat) -> XarrayConcat:
        return XarrayConcat([dataset] * self.repeat)

    def get_xarray_dataset(
        self,
        names: list[str],
        n_timesteps: int,
    ) -> tuple[XarrayConcat, DatasetProperties]:
        return build_from_config_sequence(
            configs=self.coarse,
            names=names,
            n_timesteps=IntSchedule.from_constant(n_timesteps),
            strict_ensemble=self.strict_ensemble,
        )

    def build_batchitem_dataset(
        self,
        dataset: XarrayConcat,
        properties: DatasetProperties,
    ) -> BatchItemDatasetAdapter:
        # n_timesteps is hardcoded to 1 for downscaling, so the sample_start_times
        # are the full time range for the dataset
        if dataset.sample_n_times != 1:
            raise ValueError(
                "Downscaling data loading should always have n_timesteps=1 "
                "in model data requirements."
                f" Got {dataset.sample_n_times} instead."
            )
        dataset = self._repeat_if_requested(dataset)

        dataset_subset = HorizontalSubsetDataset(
            dataset,
            properties=properties,
            lat_interval=self.lat_extent,
            lon_interval=self.lon_extent,
        )
        return BatchItemDatasetAdapter(
            dataset_subset,
            dataset_subset.subset_latlon_coordinates,
            properties=properties,
        )

    def build(
        self,
        requirements: DataRequirements,
        dist: Distributed | None = None,
    ) -> GriddedData:
        xr_dataset, properties = self.get_xarray_dataset(
            names=requirements.coarse_names, n_timesteps=1
        )
        if not isinstance(properties.horizontal_coordinates, LatLonCoordinates):
            raise ValueError(
                "Downscaling data loader only supports datasets with latlon coords."
            )
        dataset = self.build_batchitem_dataset(
            dataset=xr_dataset,
            properties=properties,
        )
        all_times = xr_dataset.sample_start_times
        if dist is None:
            dist = Distributed.get_instance()
        # Shuffle is not used for generation, it is set to False.
        sampler = (
            ContiguousDistributedSampler(dataset, drop_last=self.drop_last)
            if dist.is_distributed()
            else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=dist.local_batch_size(int(self.batch_size)),
            num_workers=self.num_data_workers,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
            collate_fn=BatchData.from_sequence,
            pin_memory=using_gpu(),
            multiprocessing_context=self.mp_context,
            persistent_workers=True if self.num_data_workers > 0 else False,
        )
        example = dataset[0]
        return GriddedData(
            _loader=dataloader,
            shape=example.horizontal_shape,
            dims=example.latlon_coordinates.dims,
            variable_metadata=dataset.variable_metadata,
            all_times=all_times,
        )


@dataclasses.dataclass
class PairedDataLoaderConfig:
    """
    Configuration for loading downscaling datasets.  The input fine and
    coarse Xarray datasets will be processed into batches, usually with
    a horizontal extent to define a portion of the full domain for use in
    training or validation. Additionally, a user may specify to take
    random subsets of the initial domain by using the coarse random extent
    arguments.

    The build ensures the compatibility of the fine/coarse datasets by
    checking that the fine coordinates are evenly divisible by the coarse
    coordinates, and that the scale factors are equal.

    Args:
        fine: The fine dataset configuration. May be a sequence of
            XarrayDataConfig or MergeNoConcatDatasetConfig.
        coarse: The coarse dataset configuration. May be a sequence of
            XarrayDataConfig, XarrayEnsembleDataConfig, or
            MergeNoConcatDatasetConfig.
        batch_size: The batch size to use for the dataloader.
        num_data_workers: The number of data workers to use for the dataloader.
            (For multi-GPU runtime, it's the number of workers per GPU.)
        strict_ensemble: Whether to enforce that the datasets to be concatened
            have the same dimensions and coordinates.
        lat_extent: The latitude extent to use for the dataset specified in
            degrees [-88, 88].  The extent is inclusive, so the start and
            stop values are included in the extent.
            Defaults to [-66, 70] which covers continental land masses aside
            from Antarctica.
        lon_extent: The longitude extent to use for the dataset specified in
            degrees (0, 360). The extent is inclusive, so the start and
            stop values are included in the extent.
        repeat: The number of times to repeat the underlying xarray dataset
            time dimension.  Useful to include longer sequences of small
            data for testing.
        topography: Deprecated field for specifying the topography dataset.
        sample_with_replacement: If provided, the dataset will be
            sampled randomly with replacement to the given size each period,
            instead of retrieving each sample once (either shuffled or not).
        drop_last: Use drop_last option in sampler. Defaults to False. If True,
            drop the last samples required to have even batch sizes across ranks.
            If false, pad with extra samples to make ranks have the same size batches.
    """

    fine: Sequence[XarrayDataConfig | MergeNoConcatDatasetConfig]
    coarse: Sequence[
        XarrayDataConfig | XarrayEnsembleDataConfig | MergeNoConcatDatasetConfig
    ]
    batch_size: int
    num_data_workers: int
    strict_ensemble: bool
    lat_extent: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(-66.0, 70.0)
    )
    lon_extent: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(float("-inf"), float("inf"))
    )
    repeat: int = 1
    topography: str | None = None
    sample_with_replacement: int | None = None
    drop_last: bool = False
    n_timesteps: int = 1
    time_stride: int | None = None

    def __post_init__(self):
        enforce_lat_bounds(self.lat_extent)
        if self.topography is not None:
            raise ValueError(
                "The `topography` field on PairedDataLoaderConfig is deprecated and "
                "will be removed in a future release. `StaticInputs` are now stored "
                "within the model when it is first built and trained."
            )
        if self.n_timesteps < 1:
            raise ValueError(f"n_timesteps must be >= 1, got {self.n_timesteps}.")
        if self.time_stride is not None and self.time_stride < 1:
            raise ValueError(
                f"time_stride must be >= 1, got {self.time_stride}."
            )

    @property
    def clip_start_stride(self) -> int:
        """Frames between consecutive video-clip starts (see ``build_video``).

        Defaults to ``n_timesteps - 1`` -- per-day non-overlapping clips that
        share only their boundary frame (for the 24h/``n_timesteps=9`` setup this
        is exactly one calendar day). ``time_stride=1`` recovers a full
        stride-one sliding window; ``time_stride=n_timesteps`` gives fully
        disjoint clips with no shared frame.
        """
        if self.time_stride is not None:
            return self.time_stride
        return max(1, self.n_timesteps - 1)

    def _first_data_config(
        self,
        config: XarrayDataConfig | MergeNoConcatDatasetConfig,
    ) -> XarrayDataConfig:
        """Return the first XarrayDataConfig for data_path/file_pattern lookup."""
        if isinstance(config, XarrayDataConfig):
            return config
        return config.merge[0]

    def _repeat_if_requested(self, dataset: XarrayConcat) -> XarrayConcat:
        return XarrayConcat([dataset] * self.repeat)

    def _mp_context(self):
        mp_context = None
        if self.num_data_workers == 0:
            return None
        for fine_config in self.fine:
            if fine_config.zarr_engine_used is True:
                mp_context = "forkserver"
                break
        for coarse_config in self.coarse:
            if coarse_config.zarr_engine_used is True:
                mp_context = "forkserver"
                break
        return mp_context

    @property
    def coarse_full_config(
        self,
    ) -> Sequence[XarrayDataConfig | MergeNoConcatDatasetConfig]:
        return _full_configs(self.coarse)

    def build(
        self,
        train: bool,
        requirements: DataRequirements,
        dist: Distributed | None = None,
    ) -> PairedGriddedData:
        if dist is None:
            dist = Distributed.get_instance()

        # Load initial datasets
        n_timesteps = IntSchedule.from_constant(requirements.n_timesteps)
        dataset_fine, properties_fine = build_from_config_sequence(
            configs=self.fine,
            names=requirements.fine_names,
            n_timesteps=n_timesteps,
            strict_ensemble=self.strict_ensemble,
        )

        dataset_coarse, properties_coarse = build_from_config_sequence(
            configs=self.coarse,
            names=requirements.coarse_names,
            n_timesteps=n_timesteps,
            strict_ensemble=self.strict_ensemble,
        )

        # Ensure that bounds for subselecting on latlon grids return fine grid data
        # that aligns with the coarse grid.
        if not isinstance(
            properties_coarse.horizontal_coordinates, LatLonCoordinates
        ) or not isinstance(properties_fine.horizontal_coordinates, LatLonCoordinates):
            raise ValueError(
                "Downscaling data loader only supports datasets with latlon coords."
            )

        # Check that timestamps on datasets are aligned
        if not dataset_fine.sample_start_times.equals(
            dataset_coarse.sample_start_times
        ):
            raise ValueError(
                "Fine and coarse datasets must have the same sample start times."
            )

        # n_timesteps is hardcoded to 1 for downscaling, so the sample_start_times
        # are the full time range for the dataset
        if dataset_fine.sample_n_times != 1:
            raise ValueError(
                "Downscaling data loading should always have n_timesteps=1 "
                "in model data requirements."
                f" Got {dataset_fine.sample_n_times} instead."
            )
        all_times = dataset_fine.sample_start_times

        dataset_fine = self._repeat_if_requested(dataset_fine)
        dataset_coarse = self._repeat_if_requested(dataset_coarse)

        dataset_fine_subset, dataset_coarse_subset = _build_aligned_subset_pair(
            dataset_fine=dataset_fine,
            properties_fine=properties_fine,
            dataset_coarse=dataset_coarse,
            properties_coarse=properties_coarse,
            lat_extent=self.lat_extent,
            lon_extent=self.lon_extent,
        )

        # Convert datasets to produce BatchItems
        dataset_fine_subset = BatchItemDatasetAdapter(
            dataset_fine_subset,
            dataset_fine_subset.subset_latlon_coordinates,
            properties=properties_fine,
        )

        dataset_coarse_subset = BatchItemDatasetAdapter(
            dataset_coarse_subset,
            dataset_coarse_subset.subset_latlon_coordinates,
            properties=properties_coarse,
        )

        dataset = FineCoarsePairedDataset(
            dataset_fine_subset,
            dataset_coarse_subset,
        )
        sampler = self._get_sampler(
            dataset=dataset, dist=dist, train=train, drop_last=self.drop_last
        )
        dataloader = DataLoader(
            dataset,
            batch_size=dist.local_batch_size(int(self.batch_size)),
            num_workers=self.num_data_workers,
            shuffle=(sampler is None) and train,
            sampler=sampler,
            drop_last=True,
            pin_memory=using_gpu(),
            collate_fn=PairedBatchData.from_sequence,
            multiprocessing_context=self._mp_context(),
            persistent_workers=True if self.num_data_workers > 0 else False,
        )

        example = dataset[0]
        variable_metadata = {
            **dataset_fine_subset.variable_metadata,
            **dataset_coarse_subset.variable_metadata,
        }

        return PairedGriddedData(
            _loader=dataloader,
            coarse_shape=example.coarse.horizontal_shape,
            downscale_factor=example.downscale_factor,
            dims=example.fine.latlon_coordinates.dims,
            variable_metadata=variable_metadata,
            all_times=all_times,
            fine_coords=get_latlon_coords_from_properties(properties_fine),
        )

    def build_video(
        self,
        train: bool,
        requirements: DataRequirements,
        dist: Distributed | None = None,
    ) -> PairedVideoGriddedData:
        """Build a paired fine/coarse loader of video clips.

        Each sample is a clip of ``self.n_timesteps`` consecutive frames with an
        explicit leading time axis (``(T, lat, lon)`` per field), plus per-frame
        physical-time features for a cBottle-style ``CalendarEmbedding``. The
        clip length is taken from ``self.n_timesteps`` (not the model's
        ``requirements.n_timesteps``), so this path leaves the model untouched.

        Consecutive clips are spaced ``self.clip_start_stride`` frames apart.
        By default (``time_stride=None``) this is ``n_timesteps - 1``, i.e.
        per-day non-overlapping clips that share only their boundary frame; for
        a 24h clip at 3-hourly resolution use ``n_timesteps=9`` (endpoints at 0h
        and 24h plus 7 interior frames). Set ``time_stride=1`` for a full
        stride-one sliding window, or ``time_stride=n_timesteps`` for fully
        disjoint clips.
        """
        if dist is None:
            dist = Distributed.get_instance()

        n_timesteps = IntSchedule.from_constant(self.n_timesteps)
        dataset_fine, properties_fine = build_from_config_sequence(
            configs=self.fine,
            names=requirements.fine_names,
            n_timesteps=n_timesteps,
            strict_ensemble=self.strict_ensemble,
        )
        dataset_coarse, properties_coarse = build_from_config_sequence(
            configs=self.coarse,
            names=requirements.coarse_names,
            n_timesteps=n_timesteps,
            strict_ensemble=self.strict_ensemble,
        )

        if not isinstance(
            properties_coarse.horizontal_coordinates, LatLonCoordinates
        ) or not isinstance(properties_fine.horizontal_coordinates, LatLonCoordinates):
            raise ValueError(
                "Downscaling data loader only supports datasets with latlon coords."
            )
        if not dataset_fine.sample_start_times.equals(
            dataset_coarse.sample_start_times
        ):
            raise ValueError(
                "Fine and coarse datasets must have the same sample start times."
            )
        if dataset_fine.sample_n_times != self.n_timesteps:
            raise ValueError(
                f"Expected clips of {self.n_timesteps} timesteps, got "
                f"{dataset_fine.sample_n_times}."
            )
        all_times = dataset_fine.sample_start_times

        dataset_fine = self._repeat_if_requested(dataset_fine)
        dataset_coarse = self._repeat_if_requested(dataset_coarse)

        dataset_fine_subset, dataset_coarse_subset = _build_aligned_subset_pair(
            dataset_fine=dataset_fine,
            properties_fine=properties_fine,
            dataset_coarse=dataset_coarse,
            properties_coarse=properties_coarse,
            lat_extent=self.lat_extent,
            lon_extent=self.lon_extent,
        )

        fine_adapter = VideoBatchItemDatasetAdapter(
            dataset_fine_subset,
            dataset_fine_subset.subset_latlon_coordinates,
            properties=properties_fine,
        )
        coarse_adapter = VideoBatchItemDatasetAdapter(
            dataset_coarse_subset,
            dataset_coarse_subset.subset_latlon_coordinates,
            properties=properties_coarse,
        )

        paired_dataset = VideoFineCoarsePairedDataset(fine_adapter, coarse_adapter)

        # Subsample the (stride-one) clip starts to the requested clip spacing.
        stride = self.clip_start_stride
        dataset: Dataset
        if stride > 1:
            keep = list(range(0, len(paired_dataset), stride))
            dataset = Subset(paired_dataset, keep)
            all_times = all_times[::stride]
        else:
            dataset = paired_dataset

        sampler = self._get_sampler(
            dataset=dataset, dist=dist, train=train, drop_last=self.drop_last
        )
        dataloader = DataLoader(
            dataset,
            batch_size=dist.local_batch_size(int(self.batch_size)),
            num_workers=self.num_data_workers,
            shuffle=(sampler is None) and train,
            sampler=sampler,
            drop_last=True,
            pin_memory=using_gpu(),
            collate_fn=PairedVideoBatchData.from_sequence,
            multiprocessing_context=self._mp_context(),
            persistent_workers=True if self.num_data_workers > 0 else False,
        )

        example = dataset[0]
        variable_metadata = {
            **fine_adapter.variable_metadata,
            **coarse_adapter.variable_metadata,
        }
        return PairedVideoGriddedData(
            _loader=dataloader,
            coarse_shape=example.coarse.horizontal_shape,
            downscale_factor=example.downscale_factor,
            n_timesteps=self.n_timesteps,
            dims=example.fine.latlon_coordinates.dims,
            variable_metadata=variable_metadata,
            all_times=all_times,
            fine_coords=get_latlon_coords_from_properties(properties_fine),
        )

    def _get_sampler(
        self, dataset: Dataset, dist: Distributed, train: bool, drop_last: bool = False
    ) -> RandomSampler | DistributedSampler | None:
        # Use RandomSampler with replacement for both distributed and
        # non-distributed cases
        if self.sample_with_replacement is not None:
            local_sample_with_replacement_dataset_size = (
                self.sample_with_replacement // dist.world_size
            )
            return RandomSampler(
                dataset,
                num_samples=local_sample_with_replacement_dataset_size,
                replacement=True,
            )
        if dist.is_distributed():
            if train:
                sampler = DistributedSampler(
                    dataset, shuffle=train, drop_last=drop_last
                )
            else:
                sampler = ContiguousDistributedSampler(dataset, drop_last=drop_last)
        else:
            sampler = None

        return sampler

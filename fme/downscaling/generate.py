"""Generate and save downscaled data using a trained FME model."""

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

from .data import BatchData, ClosedInterval, DataLoaderConfig, GriddedData
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

    def get_writer(self, sample_data: BatchData, output_dir: str) -> ZarrWriter:
        latlon_coords = sample_data.latlon_coordinates[0]
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
        ens_start = 0
        ens_slices = []
        while ens_start < self.n_ens:
            ens_end = min(ens_start + self.n_items_per_gpu, self.n_ens)
            ens_slices.append(slice(ens_start, ens_end))
            ens_start = ens_end

        return ens_slices

    def get_insert_slices(self, data: BatchData) -> list[dict[str, slice]]:
        time_slice = self._get_time_slice(data.time)
        ens_slices = self._get_ens_slices()
        # Always storing the entire region
        lat_slice = slice(None)
        lon_slice = slice(None)

        return [
            dict(zip(self.dims, (time_slice, ens_slice, lat_slice, lon_slice)))
            for ens_slice in ens_slices
        ]


@dataclass
class OutputTargetConfig(ABC):
    """Base class for output target configurations."""

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
        """Build the output target from this configuration."""
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
        if len(data_shape) != 4:
            raise ValueError(
                "Data shape must be of length 4 (time, ensemble, latitude, longitude)."
            )

        def _total_size_mb(shape: tuple[int, ...], bytes_per_element: int) -> int:
            total_elements = 1
            for dim_size in shape:
                total_elements *= dim_size
            return total_elements * bytes_per_element // (1024 * 1024)

        class NotReducibleError(Exception):
            pass

        def _recursive_chunksize_search(
            shape: tuple[int, ...], bytes_per_element: int, reduce_dim: int
        ) -> tuple[int, ...]:
            if _total_size_mb(shape, bytes_per_element) <= 20:
                return shape

            reduce_dim_size = shape[reduce_dim] // 2
            if reduce_dim_size < 1:
                # don't adjust move reduce dim
                reduce_dim += 1
                if reduce_dim >= len(shape):
                    raise NotReducibleError()
            elif reduce_dim == 2:
                # switch between lat and lon reductions for final search:
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
    """Single time snapshot over a spatial region."""

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
    """Time range at a single spatial point."""

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
    """Time range over a spatial region."""

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


class Downscaler:
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
        for target in self.output_targets:
            logging.info(f"Generating downscaled outputs for target: {target.name}")

            _model: DiffusionModel | PatchPredictor | CascadePredictor = self.model
            if target.patch.needs_patch_predictor:
                _model = PatchPredictor(
                    model=self.model,
                    coarse_horizontal_overlap=target.patch.coarse_horizontal_overlap,
                )

            insert_slices, writer = None, None
            for i, (data, topography) in enumerate(target.data.loader):
                # TODO: could use some kind of checkpointing to allow restarts
                if i == 0:
                    insert_slices = target.get_insert_slices(data)
                    writer = target.get_writer(
                        sample_data=data, output_dir=self.output_dir
                    )
                else:
                    assert insert_slices is not None
                    assert writer is not None

                for insert_slice in insert_slices:
                    logging.info(
                        f"Generating batch {i}, ensemble slice {insert_slice} "
                        f"for target: {target.name}"
                    )
                    n_ens_sl = insert_slice["ensemble"]
                    n_ens = n_ens_sl.stop - n_ens_sl.start
                    output = _model.generate_on_batch_no_target(
                        data, topography=topography, n_samples=n_ens
                    )
                    output_numpy = {
                        k: output[k].cpu().numpy() for k in target.save_vars
                    }
                    writer.record_batch(output_numpy, position_slices=insert_slice)


@dataclass
class GenerationConfig:
    model: CheckpointModelConfig | CascadePredictorConfig
    data: DataLoaderConfig
    output_dir: str
    output_targets: list[OutputTargetConfig]
    logging: LoggingConfig
    patch: PatchPredictionConfig = field(default_factory=PatchPredictionConfig)
    """
    This class is used to configure the downscaling generation of target regions.
    Fine-resolution outputs are generated from coarse-resolution inputs.
    In contrast to the Evaluator, there is no fine-resolution target data
    to compare the generated outputs against.
    """

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

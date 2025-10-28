"""Generate and save downscaled data using a trained FME model."""

import argparse
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta

import dacite
import yaml

from fme.core import logging_utils
from fme.core.cli import prepare_directory
from fme.core.dataset.time import RepeatedInterval, TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.logging_utils import LoggingConfig
from fme.core.typing_ import Slice

from .data import ClosedInterval, DataLoaderConfig, GriddedData
from .data.config import XarrayEnsembleDataConfig
from .models import CheckpointModelConfig, DiffusionModel
from .predictors import CascadePredictor, CascadePredictorConfig, PatchPredictionConfig
from .requirements import DataRequirements
from .train import count_parameters


class OutputTarget:
    def __init__(
        self,
        name: str,
        save_vars: list[str],
        n_ens: int,
        data: GriddedData,
        patch: PatchPredictionConfig,
        chunks: dict[str, int],
        shards: dict[str, int] | None,
    ) -> None:
        self.name = name
        self.save_vars = save_vars
        self.n_ens = n_ens
        self.data = data
        self.patch = patch
        self.chunks = chunks
        self.shards = shards


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

    @staticmethod
    def _replace_loader_config(
        time, coarse, lat_extent, lon_extent, loader_config
    ) -> DataLoaderConfig:
        # return a new DataloaderConfig with replaced values
        new_coarse = [replace(coarse[0], subset=time)]

        new_loader_config = replace(
            loader_config,
            coarse=new_coarse,
            lat_extent=lat_extent,
            lon_extent=lon_extent,
        )
        return new_loader_config

    def _get_chunks(
        self, data_shape: tuple[int, ...], bytes_per_element: int
    ) -> dict[str, int]:
        # time, ensemble, latitude, longitude
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
        ) -> dict[str, int]:
            chunk = {
                "time": shape[0],
                "ensemble": shape[1],
                "latitude": shape[2],
                "longitude": shape[3],
            }
            if _total_size_mb(shape, bytes_per_element) <= 20:
                return chunk

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

        return _recursive_chunksize_search(data_shape, bytes_per_element, reduce_dim=0)

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
            time, coarse, lat_extent, lon_extent, loader_config
        )
        data = updated_loader_config.build(requirements=requirements)

        sample_dict = next(iter(data.loader))
        sample_tensor = next(iter(sample_dict.values()))
        if self.zarr_chunks is None:
            chunks = self._get_chunks(
                data_shape=sample_tensor.shape,
                bytes_per_element=sample_tensor.element_size(),
            )
        else:
            chunks = self.zarr_chunks

        return OutputTarget(
            name=self.name,
            save_vars=self.save_vars,
            n_ens=self.n_ens,
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
    ):
        self.model = model
        self.output_targets = output_targets

    def run(self):
        for target in self.output_targets:
            logging.info(f"Generating downscaled outputs for target: {target.name}")


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

"""Generate and save downscaled data using a trained FME model."""

import argparse
import logging
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta

import dacite
import yaml

from fme.core import logging_utils
from fme.core.cli import prepare_directory
from fme.core.dataset.time import RepeatedInterval, TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.dicts import to_flat_dict
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
        data: GriddedData,
        patch: PatchPredictionConfig,
        chunks: dict[str, int] | None = None,
        shards: dict[str, int] | None = None,
    ) -> None:
        self.data = data
        self.patch = patch
        self.chunks = chunks
        self.shards = shards


@dataclass
class OutputTargetConfig:
    name: str
    n_ens: int
    save_vars: list[str]
    time: str | int | RepeatedInterval | Slice | TimeSlice | None = None
    time_format: str = "%Y-%m-%dT%H:%M:%S"
    lat_extent: ClosedInterval = field(
        default_factory=lambda: ClosedInterval(-90.0, 90.0)
    )
    lon_extent: ClosedInterval = field(
        default_factory=lambda: ClosedInterval(float("-inf"), float("inf"))
    )
    zarr_chunks: dict[str, int] | None = None
    zarr_shards: dict[str, int] | None = None
    coarse: list[XarrayDataConfig] | None = None
    patch: PatchPredictionConfig | None = None

    def _time_selection_slice(self, time: str) -> TimeSlice:
        """Returns a TimeSlice containing only the first 6h time(s).
        Event evaluation only load the first snapshot.
        Filling the slice stop isn't necessary but guards against
        future code trying to iterate over the entire dataloader.
        """
        _stop = (
            datetime.strptime(time, self.time_format) + timedelta(hours=12)
        ).strftime(self.time_format)
        return TimeSlice(time, _stop)

    def build(
        self,
        loader_config: DataLoaderConfig,
        requirements: DataRequirements,
        patch: PatchPredictionConfig,
    ) -> OutputTarget:
        # override with parent configuration if not specified for target
        coarse = self.coarse or loader_config.coarse
        patch = self.patch or patch

        time: Slice | RepeatedInterval | TimeSlice
        if isinstance(self.time, str):
            time = self._time_selection_slice(self.time)
        elif isinstance(self.time, int):
            time = Slice(self.time, self.time + 1)
        elif self.time is None:
            time = Slice()
        else:
            time = self.time

        new_coarse = []
        for xarray_config in coarse:
            if isinstance(xarray_config, XarrayEnsembleDataConfig):
                raise NotImplementedError(
                    "XarrayEnsembleDataConfig is not supported in OutputTargetConfig."
                )
            new_coarse.append(
                replace(
                    xarray_config,
                    subset=time,
                )
            )

        new_loader_config = replace(
            loader_config,
            coarse=new_coarse,
            lat_extent=self.lat_extent,
            lon_extent=self.lon_extent,
        )

        data = new_loader_config.build(requirements=requirements)
        return OutputTarget(
            data=data,
            patch=patch,
            chunks=self.zarr_chunks,
            shards=self.zarr_shards,
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
        pass


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

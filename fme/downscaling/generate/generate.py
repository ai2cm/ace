import logging
from dataclasses import asdict, dataclass, field

import dacite
import numpy as np
import torch
import yaml

from fme.core import logging_utils
from fme.core.cli import prepare_directory
from fme.core.dicts import to_flat_dict
from fme.core.logging_utils import LoggingConfig

from ..data import DataLoaderConfig, Topography
from ..models import CheckpointModelConfig, DiffusionModel
from ..predictors import (
    CascadePredictor,
    CascadePredictorConfig,
    PatchPredictionConfig,
    PatchPredictor,
)
from ..train import count_parameters
from .output import EventConfig, OutputTarget, RegionConfig
from .work_items import LoadedWorkItem


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
        model_patch_shape = self.model.fine_shape
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

        loaded_item: LoadedWorkItem
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
            output_np = {k: output[k].cpu().numpy() for k in target.save_vars}
            insert_slices = loaded_item.insert_slices
            
            if loaded_item.is_padding:
                logging.info("Creating empty slices for padding work item.")
                output_np_empty = {
                    k: np.empty([0, 0] + list(output_np[k].shape[2:]), dtype=output_np[k].dtype) 
                    for k in output_np.keys()
                }
                output_np = output_np_empty
                insert_slices_empty = {k: slice(0, 0) for k in insert_slices}
                insert_slices = insert_slices_empty
            
            writer.record_batch(
                output_np, position_slices=insert_slices
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
    experiment_dir: str
    output_targets: list[EventConfig | RegionConfig]
    logging: LoggingConfig
    patch: PatchPredictionConfig = field(default_factory=PatchPredictionConfig)

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

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
        return Downscaler(
            model=model, output_targets=targets, output_dir=self.experiment_dir
        )


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    generation_config: GenerationConfig = dacite.from_dict(
        data_class=GenerationConfig,
        data=config,
        config=dacite.Config(strict=True),
    )
    prepare_directory(generation_config.experiment_dir, config)

    generation_config.configure_logging(log_filename="out.log")
    logging_utils.log_versions()
    beaker_url = logging_utils.log_beaker_url()
    generation_config.configure_wandb(resumable=True, notes=beaker_url)

    logging.info("Starting downscaling generation...")
    downscaler = generation_config.build()
    logging.info(f"Number of parameters: {count_parameters(downscaler.model.modules)}")
    downscaler.run_all()

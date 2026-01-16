import logging
from dataclasses import asdict, dataclass, field

import dacite
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
from .output import DownscalingOutput, EventConfig, TimeRangeConfig
from .work_items import LoadedSliceWorkItem


class Downscaler:
    """
    Orchestrates downscaling generation across multiple outputs.

    Each output can have different spatial extents, time ranges, and ensemble sizes.
    Generation is performed sequentially across outputs.
    """

    def __init__(
        self,
        model: DiffusionModel | CascadePredictor,
        outputs: list[DownscalingOutput],
        output_dir: str = ".",
    ):
        self.model = model
        self.outputs = outputs
        self.output_dir = output_dir

    def run_all(self):
        """Run generation for all outputs."""
        logging.info(f"Starting generation for {len(self.outputs)} output(s)")

        for output in self.outputs:
            # Clear GPU cache before each output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.run_output_generation(output=output)

        logging.info("All outputs completed successfully")

    def _get_generation_model(
        self,
        topography: Topography,
        output: DownscalingOutput,
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
                f"actual topography shape {actual_shape} for output {output.name}."
            )
        elif output.patch.needs_patch_predictor:
            # Use a patch predictor
            logging.info(f"Using PatchPredictor for output: {output.name}")
            return PatchPredictor(
                model=self.model,
                coarse_horizontal_overlap=output.patch.coarse_horizontal_overlap,
            )
        else:
            # User should enable patching
            raise ValueError(
                f"Model coarse shape {model_patch_shape} does not match "
                f"actual input shape {actual_shape} for output {output.name}, "
                "and patch prediction is not configured. Generation for larger domains "
                "requires patch prediction."
            )

    def _on_device_generator(self, loader):
        for loaded_item, topography in loader:
            yield loaded_item.to_device(), topography.to_device()

    def run_output_generation(self, output: DownscalingOutput):
        """Execute the generation loop for this output."""
        logging.info(f"Generating downscaled outputs for output: {output.name}")

        # initialize writer and model in loop for coord info
        model = None
        writer = None
        total_batches = len(output.data.loader)

        loaded_item: LoadedSliceWorkItem
        topography: Topography
        for i, (loaded_item, topography) in enumerate(output.data.get_generator()):
            if writer is None:
                writer = output.get_writer(
                    latlon_coords=topography.coords,
                    output_dir=self.output_dir,
                )
                writer.initialize_store(topography.data.cpu().numpy().dtype)
            if model is None:
                model = self._get_generation_model(topography=topography, output=output)

            logging.info(
                f"[{output.name}] Batch {i+1}/{total_batches}, "
                f"generating work slice {loaded_item.dim_insert_slices} "
            )

            output_data = model.generate_on_batch_no_target(
                loaded_item.batch, topography=topography, n_samples=loaded_item.n_ens
            )
            output_np = {key: value.cpu().numpy() for key, value in output_data.items()}
            insert_slices = loaded_item.dim_insert_slices

            if not loaded_item.is_padding:
                writer.record_batch(output_np, position_slices=insert_slices)
            else:
                logging.info("Skipping padding work item. No data will be written.")

        logging.info(f"Completed generation for output: {output.name}")


@dataclass
class InferenceConfig:
    """
    Top-level configuration for downscaling generation entrypoint.

    Defines the model, base data source, and one or more outputs to generate.
    Fine-resolution outputs are generated from coarse-resolution inputs without
    requiring fine-resolution target data (unlike training/evaluation).

    Each output can specify different spatial regions, time ranges, ensemble
    sizes, and output variables. Outputs are processed sequentially, with generation
    parallelized across GPUs using distributed data loading.

    Parameters:
        model: Model specification to load for generation.
        data: Base data loader configuration that is shared to each output
            generation task. Specifics for each output like the time(range),
            spatial extent, saved variables, and max_samples_per_gpu
            (effective batch size) are specified in each outputß.
        experiment_dir: Directory for saving generated zarr files and logs.
        outputs: List of output specifications. Each output generates a
            separate zarr file.
        logging: Logging configuration.
        patch: Default patch prediction configuration.

    Exclude following from autoclass documentation:
    Example YAML configuration::

        experiment_dir: /results
        model:
            checkpoint_path: /checkpoints/best_histogram_tail.ckpt
        data:
            topography: /climate-default/X-SHiELD-AMIP-downscaling/3km.zarr
            coarse:
            - data_path: /climate-default/X-SHiELD-AMIP-downscaling
              engine: zarr
              file_pattern: 100km.zarr
            batch_size: 4  # Value is overidden by each output
            num_data_workers: 0
            strict_ensemble: False
        patch:
            divide_generation: true
            composite_prediction: true
            coarse_horizontal_overlap: 0
        outputs:
          - name: "WA_AR_20230206"
            save_vars: ["PRATEsfc"]
            n_ens: 128
            max_samples_per_gpu: 8
            event_time: "2023-02-06T06:00:00"
            lat_extent:
                start: 36.0
                stop:  52.0
            lon_extent:
                start: 228.0
                stop: 244.0
          - name: "CONUS_2023"
            save_vars: ["PRATEsfc"]
            n_ens: 8
            max_samples_per_gpu: 8
            time_range:
               start_time: "2023-01-01T00:00:00"
               end_time: "2023-12-31T18:00:00"
            lat_extent:
                start: 22.0
                stop:  50.0
            lon_extent:
                start: 230.0
                stop: 295.0
        logging:
            log_to_screen: true
            log_to_wandb: false
            log_to_file: true
            project: downscaling
            entity: my_organization
    """

    model: CheckpointModelConfig | CascadePredictorConfig
    data: DataLoaderConfig
    experiment_dir: str
    outputs: list[EventConfig | TimeRangeConfig]
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
        model = self.model.build()
        outputs = [
            output_cfg.build(
                loader_config=self.data,
                requirements=self.model.data_requirements,
                patch=self.patch,
                static_inputs_from_checkpoint=model.static_inputs,
            )
            for output_cfg in self.outputs
        ]
        return Downscaler(model=model, outputs=outputs, output_dir=self.experiment_dir)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    generation_config: InferenceConfig = dacite.from_dict(
        data_class=InferenceConfig,
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

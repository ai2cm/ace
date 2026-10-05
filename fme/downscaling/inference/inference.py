import dataclasses
import logging
import time
from dataclasses import dataclass, field

import dacite
import numpy as np
import torch
import yaml

from fme.core.cli import prepare_directory
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.generics.trainer import count_parameters
from fme.core.logging_utils import LoggingConfig
from fme.core.wandb import WandB

from ..data import DataLoaderConfig
from ..models import CheckpointModelConfig, DiffusionModel
from ..predictors import (
    DenoisingMoEBundledConfig,
    DenoisingMoEConfig,
    DenoisingMoEPredictor,
    PatchPredictionConfig,
    PatchPredictor,
    check_input_shape_supported,
)
from .output import DownscalingOutput, EventConfig, TimeRangeConfig
from .work_items import LoadedSliceWorkItem


def _synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Downscaler:
    """
    Orchestrates downscaling generation across multiple outputs.

    Each output can have different spatial extents, time ranges, and ensemble sizes.
    Generation is performed sequentially across outputs.
    """

    def __init__(
        self,
        model: DiffusionModel | DenoisingMoEPredictor,
        outputs: list[DownscalingOutput],
        output_dir: str = ".",
    ):
        self.model = model
        self.outputs = outputs
        self.output_dir = output_dir
        # wandb steps must increase monotonically across all outputs
        self._wandb_step = 0

    def _log_batch_timing(
        self, output_name: str, generate_seconds: float, n_samples: int
    ):
        """Log the slowest rank's model-call time for this batch to wandb."""
        max_seconds = Distributed.get_instance().reduce_max(
            torch.tensor(generate_seconds, device=get_device())
        )
        logging.info(
            f"[{output_name}] generate: {generate_seconds:.3f}s this rank, "
            f"{max_seconds.item():.3f}s max across ranks, {n_samples} samples"
        )
        WandB.get_instance().log(
            {
                "timing/generate_seconds": max_seconds.item(),
                "timing/generate_seconds_per_sample": max_seconds.item() / n_samples,
            },
            step=self._wandb_step,
        )
        self._wandb_step += 1

    def _log_output_timing_summary(self, output_name: str, generate_times: list[float]):
        """
        Log total and steady-state model-call time for an output to wandb.
        Totals are maxed across ranks, since ranks run in parallel and the
        slowest one sets wall time. The first batch is excluded from the
        steady-state mean because it includes CUDA warmup.
        """
        dist = Distributed.get_instance()
        device = get_device()
        total = dist.reduce_max(torch.tensor(sum(generate_times), device=device))
        steady = generate_times[1:] if len(generate_times) > 1 else generate_times
        mean_steady = dist.reduce_max(
            torch.tensor(float(np.mean(steady)), device=device)
        )
        summary = {
            "timing/total_generate_seconds": total.item(),
            "timing/mean_generate_seconds_excl_first_batch": (mean_steady.item()),
            "timing/n_batches_per_rank": len(generate_times),
        }
        for key, value in summary.items():
            logging.info(f"{key}: {value}")
        WandB.get_instance().log(summary, step=self._wandb_step)
        self._wandb_step += 1

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
        output: DownscalingOutput,
    ) -> DiffusionModel | DenoisingMoEPredictor | PatchPredictor:
        """
        Set up the model for inference: roll coordinates to match the coarse lon
        convention, then wrap with PatchPredictor if needed.  While models are
        probably capable of generating any domain size, we haven't tested for domains
        smaller than the model patch size, so we raise an error in that case, and prompt
        the user to use patching for larger domains because that provides better
        generations.

        The coarse coordinates covering the output's full extent live on
        ``output.data`` (fixed at build time, independent of the per-batch work-item
        slices), so both the input shape and the longitude convention to roll to are
        derived from there.
        """
        coarse_coords = output.data.coarse_extent_latlon_coords
        input_shape = (len(coarse_coords.lat), len(coarse_coords.lon))
        # No-op when coarse_lon does not cross the prime meridian.
        base_model = self.model.with_rolled_lon(coarse_coords.lon)
        check_input_shape_supported(
            base_model.coarse_shape,
            input_shape,
            output.patch,
            name=f"output {output.name}",
        )
        if base_model.coarse_shape == input_shape:
            return base_model

        logging.info(f"Using PatchPredictor for output: {output.name}")
        return PatchPredictor(
            model=base_model,
            coarse_horizontal_overlap=output.patch.coarse_horizontal_overlap,
        )

    def _on_device_generator(self, loader):
        for loaded_item in loader:
            yield loaded_item.to_device()

    def run_output_generation(self, output: DownscalingOutput):
        """Execute the generation loop for this output."""
        logging.info(f"Generating downscaled outputs for output: {output.name}")

        model = self._get_generation_model(output=output)

        writer = None
        total_batches = len(output.data.loader)
        generate_times: list[float] = []

        loaded_item: LoadedSliceWorkItem
        for i, loaded_item in enumerate(output.data.get_generator()):
            if writer is None:
                fine_latlon_coords = model.get_fine_coords_for_batch(loaded_item.batch)
                writer = output.get_writer(
                    latlon_coords=fine_latlon_coords,
                    output_dir=self.output_dir,
                )
                writer.initialize_store(np.float32)

            logging.info(
                f"[{output.name}] Batch {i+1}/{total_batches}, "
                f"generating work slice {loaded_item.dim_insert_slices} "
            )

            # Synchronize so the timer covers only the model call, not pending
            # host-to-device copies or the device-to-host copy below.
            _synchronize()
            start = time.perf_counter()
            output_data = model.generate_on_batch_no_target(
                loaded_item.batch,
                n_samples=loaded_item.n_ens,
            )
            _synchronize()
            generate_seconds = time.perf_counter() - start
            n_samples = (
                loaded_item.time_slice.stop - loaded_item.time_slice.start
            ) * loaded_item.n_ens
            self._log_batch_timing(output.name, generate_seconds, n_samples)
            generate_times.append(generate_seconds)

            output_np = {key: value.cpu().numpy() for key, value in output_data.items()}
            insert_slices = loaded_item.dim_insert_slices

            if not loaded_item.is_padding:
                writer.record_batch(output_np, position_slices=insert_slices)
            else:
                logging.info("Skipping padding work item. No data will be written.")

        if generate_times:
            self._log_output_timing_summary(output.name, generate_times)
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

    model: DenoisingMoEConfig | DenoisingMoEBundledConfig | CheckpointModelConfig
    data: DataLoaderConfig
    experiment_dir: str
    outputs: list[EventConfig | TimeRangeConfig]
    logging: LoggingConfig
    patch: PatchPredictionConfig = field(default_factory=PatchPredictionConfig)

    def configure_logging(self, log_filename: str):
        config = dataclasses.asdict(self)
        self.logging.configure_logging(
            self.experiment_dir, log_filename, config=config, resumable=True
        )

    def build(self) -> Downscaler:
        model = self.model.build()
        outputs = [
            output_cfg.build(
                loader_config=self.data,
                requirements=self.model.data_requirements,
                patch=self.patch,
                fine_shape=model.fine_shape,
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

    logging.info("Starting downscaling generation...")
    downscaler = generation_config.build()
    logging.info(f"Number of parameters: {count_parameters(downscaler.model.modules)}")
    downscaler.run_all()

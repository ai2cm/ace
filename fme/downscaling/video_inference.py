# SPDX-FileCopyrightText: Copyright (c) 2026, Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""Test-set inference for the endpoint-conditioned video diffusion model.

Given a trained checkpoint, runs an ensemble of samples over a test split and
writes a single zarr store matching the input dataset's format -- dims
(time, ensemble, latitude, longitude) -- covering the full test period with
no gaps: endpoint frames are the observed ground truth broadcast across the
ensemble dimension, interior frames are the generated ensemble. A
``frame_source`` time-coordinate flags observed (0) vs. generated (1) frames.

Run (mirrors video_train.py's invocation):
    torchrun --nproc_per_node N -m fme.downscaling.video_inference <config.yaml>
"""

import argparse
import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import dacite
import numpy as np
import torch
import xarray as xr
import yaml

from fme.core.cli import prepare_directory
from fme.core.distributed import Distributed
from fme.core.ema import EMATracker
from fme.core.generics.trainer import count_parameters
from fme.core.logging_utils import LoggingConfig
from fme.core.writer import ZarrWriter
from fme.downscaling.data import PairedVideoGriddedData, PairedVideoLoaderConfig
from fme.downscaling.inference.zarr_utils import determine_zarr_chunks
from fme.downscaling.video_models import VideoDiffusionModel, VideoDiffusionModelConfig

logger = logging.getLogger(__name__)

TIME_NAME = "time"
ENSEMBLE_NAME = "ensemble"
LAT_NAME = "latitude"
LON_NAME = "longitude"
DIMS = (TIME_NAME, ENSEMBLE_NAME, LAT_NAME, LON_NAME)


def _bare_module(wrapped: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a ``DummyWrapper``/``DistributedDataParallel`` down to the raw
    net it wraps under ``.module``.
    """
    bare = getattr(wrapped, "module", None)
    if bare is None:
        raise RuntimeError(
            f"Expected {type(wrapped).__name__} to wrap a raw module under "
            "`.module` (DummyWrapper or DistributedDataParallel); checkpoint "
            "loading assumptions not met."
        )
    return bare


def _splice_observed_endpoints(
    generated: dict[str, torch.Tensor],
    truth: Mapping[str, torch.Tensor],
    n_ensemble: int,
) -> dict[str, torch.Tensor]:
    """Overwrite each clip's first/last frame with the observed ground truth,
    broadcast across the ensemble dimension.

    ``generated``/``truth`` are keyed by variable name, shaped
    (B, n_ensemble, T, H, W) / (B, T, H, W) respectively. Mutates and returns
    ``generated``.
    """
    for name, clip in generated.items():
        gt = truth[name]
        clip[:, :, 0] = gt[:, None, 0].expand(-1, n_ensemble, -1, -1)
        clip[:, :, -1] = gt[:, None, -1].expand(-1, n_ensemble, -1, -1)
    return generated


def _clip_write_slice(
    clip_start_time,
    frame_times: np.ndarray,
    last_clip_start_idx: int,
    n_timesteps: int,
) -> slice:
    """Time-axis slice a clip should write to, given its start time.

    Tumbling clips share their boundary frame with the next clip, so every
    clip but the last writes only its first ``n_timesteps - 1`` frames; the
    last clip also writes its own trailing endpoint.
    """
    n_time = len(frame_times)
    start_idx = int(np.searchsorted(frame_times, clip_start_time))
    if start_idx >= n_time or frame_times[start_idx] != clip_start_time:
        raise ValueError(
            f"Clip start time {clip_start_time} not found in the reference "
            "test time axis; data/config mismatch."
        )
    is_last_clip = start_idx == last_clip_start_idx
    n_frames = n_timesteps if is_last_clip else n_timesteps - 1
    return slice(start_idx, start_idx + n_frames)


def _validate_contiguous_clips(
    clip_start_indices: np.ndarray, n_time: int, n_timesteps: int
) -> None:
    """Guards against non-contiguous inputs (e.g. a multi-entry
    ``data.fine``/``data.coarse``, or a strided subset): tumbling clips must
    tile the frame axis in exact ``n_timesteps - 1`` steps with no gaps.
    """
    expected_starts = np.arange(0, n_time - n_timesteps + 1, n_timesteps - 1)
    if not np.array_equal(clip_start_indices, expected_starts):
        raise ValueError(
            f"Clip starts {clip_start_indices.tolist()} do not tile the "
            f"{n_time}-frame time axis in steps of {n_timesteps - 1}; some "
            "frames would be left unwritten. Check for multiple entries in "
            "data.fine/data.coarse or a strided data subset."
        )


def _warn_if_idle_ranks(n_clips: int, world_size: int) -> None:
    if n_clips < world_size:
        logger.warning(
            f"Only {n_clips} clip(s) available but {world_size} rank(s) "
            "requested; all but one rank will sit idle."
        )


@dataclass
class VideoInferenceConfig:
    """Config for running test-set inference with a trained video PMD model.

    ``model`` and ``data`` come from the training config's ``model:`` and
    ``test_data:`` blocks -- no new schema for those.

    Args:
        checkpoint_path: Path to the trained checkpoint to load.
        model: Model config, from training's ``model:`` block.
        data: Data config, from training's ``test_data:`` block.
        output_path: Path to write the output zarr store to.
        experiment_dir: Directory for logs.
        n_ensemble: Number of ensemble members to generate per clip.
        ensemble_chunk_size: Number of ensemble members to generate at once.
        use_ema: Whether to load the EMA (vs. raw) weights.
        overwrite: If True, overwrite an existing store at output_path
            instead of failing. Use while iterating on a run that keeps
            failing/retrying; leave False once a run is expected to succeed.
    """

    checkpoint_path: str
    model: VideoDiffusionModelConfig
    data: PairedVideoLoaderConfig
    output_path: str
    experiment_dir: str
    logging: LoggingConfig
    n_ensemble: int = 32
    ensemble_chunk_size: int = 8
    use_ema: bool = True
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.data.n_timesteps != self.model.n_timesteps:
            raise ValueError(
                f"data.n_timesteps ({self.data.n_timesteps}) must equal "
                f"model.n_timesteps ({self.model.n_timesteps})."
            )
        if self.data.clip_start_stride != self.model.n_timesteps - 1:
            raise ValueError(
                "Test-set inference requires tumbling clips (clip_start_stride "
                f"== n_timesteps - 1); got clip_start_stride="
                f"{self.data.clip_start_stride} for n_timesteps="
                f"{self.model.n_timesteps}. A different stride would leave "
                "gaps or overlaps in the output's write coverage."
            )
        if self.data.repeat != 1:
            raise ValueError(
                "data.repeat must be 1 for test-set inference (no repeated "
                f"clips), got {self.data.repeat}."
            )
        if self.data.sample_with_replacement is not None:
            raise ValueError(
                "data.sample_with_replacement must be unset for test-set "
                "inference (every clip must be visited exactly once)."
            )
        if self.ensemble_chunk_size <= 0:
            raise ValueError(
                f"ensemble_chunk_size must be > 0, got {self.ensemble_chunk_size}."
            )

    def configure_logging(self, log_filename: str) -> None:
        config = dataclasses.asdict(self)
        self.logging.configure_logging(
            self.experiment_dir, log_filename, config=config, resumable=True
        )

    def build_model(self) -> VideoDiffusionModel:
        """Build the model from config and load trained weights.

        The training config used ``validate_using_ema: true``, so the checkpoint
        that was selected as "best" was evaluated with EMA-swapped weights. To
        reproduce that quality at inference time, load the raw state dict first
        (establishes buffers) and then overwrite the trainable params with the
        EMA shadow, exactly mirroring ``VideoTrainer._ema_context()``.
        """
        model = self.model.build()
        # Checkpoint tensors don't need to be on GPU just to load into model.
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = {
            (key[len("module.") :] if key.startswith("module.") else key): value
            for key, value in ckpt["module"].items()
        }
        _bare_module(model.module).load_state_dict(state_dict)
        if self.use_ema:
            if "ema" not in ckpt:
                raise ValueError(
                    f"use_ema=True but checkpoint {self.checkpoint_path} has no "
                    "'ema' state."
                )
            ema = EMATracker.from_state(ckpt["ema"], model.modules)
            ema.copy_to(model.modules)
            logger.info("Loaded EMA weights for inference.")
        else:
            logger.info("Loaded raw (non-EMA) weights for inference.")
        model.module.eval()
        logger.info(
            f"Loaded checkpoint {self.checkpoint_path} "
            f"(epoch {ckpt.get('startEpoch')}, "
            f"best_valid_loss {ckpt.get('best_valid_loss')})"
        )
        return model

    def build(self) -> "VideoInferenceRunner":
        """Build the model, load its checkpoint, and set up the test-set loader."""
        model = self.build_model()
        griddata = self.data.build_video(
            train=False, requirements=self.model.data_requirements, drop_last=False
        )
        var_attrs = {
            name: griddata.variable_metadata[name].as_attrs()
            for name in model.out_names
            if name in griddata.variable_metadata
        }
        return VideoInferenceRunner(
            model=model,
            griddata=griddata,
            var_attrs=var_attrs,
            output_path=self.output_path,
            n_ensemble=self.n_ensemble,
            ensemble_chunk_size=self.ensemble_chunk_size,
            checkpoint_path=self.checkpoint_path,
            use_ema=self.use_ema,
            overwrite=self.overwrite,
        )


class VideoInferenceRunner:
    """Runs test-set inference and writes the output zarr store.

    Takes only built collaborators and plain values -- never a config -- so
    it has no knowledge of ``VideoInferenceConfig``'s structure.
    """

    def __init__(
        self,
        model: VideoDiffusionModel,
        griddata: PairedVideoGriddedData,
        var_attrs: dict[str, dict[str, str]],
        output_path: str,
        n_ensemble: int,
        ensemble_chunk_size: int,
        checkpoint_path: str,
        use_ema: bool,
        overwrite: bool,
    ):
        self.model = model
        self.griddata = griddata
        self.var_attrs = var_attrs
        self.output_path = output_path
        self.n_ensemble = n_ensemble
        self.ensemble_chunk_size = ensemble_chunk_size
        self.checkpoint_path = checkpoint_path
        self.use_ema = use_ema
        self.overwrite = overwrite

    def run(self) -> None:
        dist = Distributed.get_instance()
        model, griddata = self.model, self.griddata
        frame_times = griddata.frame_times
        last_clip_start_idx = griddata.clip_start_indices[-1]

        _warn_if_idle_ranks(len(griddata.clip_start_times), dist.world_size)

        logger.info(f"Number of parameters: {count_parameters(model.modules)}")

        n_time = len(frame_times)
        n_timesteps = model.n_timesteps
        _validate_contiguous_clips(griddata.clip_start_indices, n_time, n_timesteps)

        frame_source = np.ones(n_time, dtype=np.int8)
        frame_source[griddata.clip_start_indices] = 0
        frame_source[-1] = 0  # final frame is an observed endpoint, not a clip start

        lat = griddata.fine_extent_latlon_coords.lat.cpu().numpy()
        lon = griddata.fine_extent_latlon_coords.lon.cpu().numpy()

        coords = {
            TIME_NAME: np.asarray(frame_times),
            ENSEMBLE_NAME: np.arange(self.n_ensemble),
            LAT_NAME: lat,
            LON_NAME: lon,
        }
        chunks = determine_zarr_chunks(
            dims=DIMS,
            data_shape=(n_time, self.n_ensemble, len(lat), len(lon)),
            bytes_per_element=4,
        )
        # Shard to one clip's full write, avoiding an object per frame/member.
        shards = {TIME_NAME: n_timesteps - 1, ENSEMBLE_NAME: self.n_ensemble}

        writer = ZarrWriter(
            path=self.output_path,
            dims=DIMS,
            coords=coords,
            data_vars=model.out_names,
            chunks=chunks,
            shards=shards,
            array_attributes=self.var_attrs,
            group_attributes={
                "description": (
                    "Test-set inference: endpoint-conditioned video diffusion "
                    "infilling, ensemble of independent noise draws."
                ),
                "checkpoint_path": self.checkpoint_path,
                "n_ensemble": str(self.n_ensemble),
                "use_ema": str(self.use_ema),
            },
            nondim_coords={
                "frame_source": xr.DataArray(frame_source, dims=[TIME_NAME]),
            },
            mode="w" if self.overwrite else "w-",
            time_calendar=frame_times.calendar,
        )
        writer.initialize_store(data_dtype=np.float32)

        n_batches = len(griddata.loader)
        for i, batch in enumerate(griddata.loader):
            remaining = self.n_ensemble
            ensemble_chunks: dict[str, list[torch.Tensor]] = {
                name: [] for name in model.out_names
            }
            while remaining > 0:
                n = min(self.ensemble_chunk_size, remaining)
                generated = model.generate(batch, n_samples=n)
                for name in model.out_names:
                    ensemble_chunks[name].append(generated[name].cpu())
                remaining -= n
            # (B, n_ensemble, T, H, W), on CPU
            full = {
                name: torch.cat(chunks_, dim=1)
                for name, chunks_ in ensemble_chunks.items()
            }
            truth = {name: tensor.cpu() for name, tensor in batch.fine.data.items()}
            full = _splice_observed_endpoints(full, truth, self.n_ensemble)

            clip_times = batch.fine.time.values  # (B, T) cftime
            for b in range(clip_times.shape[0]):
                time_slice = _clip_write_slice(
                    clip_times[b, 0], frame_times, last_clip_start_idx, n_timesteps
                )
                n_frames_to_write = time_slice.stop - time_slice.start

                write_data = {
                    name: full[name][b, :, :n_frames_to_write]
                    .permute(
                        1, 0, 2, 3
                    )  # (n_ensemble, T', H, W) -> (T', n_ensemble, H, W)
                    .to(torch.float32)
                    .numpy()
                    for name in model.out_names
                }
                writer.record_batch(
                    write_data,
                    position_slices={
                        TIME_NAME: time_slice,
                        ENSEMBLE_NAME: slice(0, self.n_ensemble),
                    },
                )

            logger.info(f"Rank {dist.rank}: batch {i + 1}/{n_batches} written")

        if dist.is_distributed():
            dist.barrier()
        logger.info(f"Completed inference. Output: {self.output_path}")


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    inference_config: VideoInferenceConfig = dacite.from_dict(
        data_class=VideoInferenceConfig,
        data=config,
        config=dacite.Config(strict=True),
    )
    prepare_directory(inference_config.experiment_dir, config)
    inference_config.configure_logging(log_filename="out.log")
    logging.info("Starting video diffusion test-set inference")
    inference_config.build().run()


def parse_args():
    parser = argparse.ArgumentParser(description="Video PMD test-set inference")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Distributed.context():
        main(args.config_path)

# SPDX-FileCopyrightText: Copyright (c) 2026, Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""Test-set inference for the endpoint-conditioned video diffusion model.

Given a trained checkpoint, runs an ensemble of samples over a test split:
observed daily endpoint snapshots (0h, 24h) in, infilled 3-hourly interior
frames (3h..21h) out. Writes a single zarr store matching the input dataset's
format -- dims (time, ensemble, latitude, longitude) -- covering the full test
period with no gaps: endpoint frames are the observed ground truth broadcast
across the ensemble dimension, interior frames are the generated ensemble. A
``frame_source`` time-coordinate flags observed (0) vs. generated (1) frames.

Run (mirrors video_train.py's invocation):
    torchrun --nproc_per_node N -m fme.downscaling.video_inference <config.yaml>
"""

import argparse
import dataclasses
import itertools
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

import dacite
import numpy as np
import torch
import xarray as xr
import yaml

from fme.core.cli import prepare_directory
from fme.core.dataset.time import TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.device import get_device
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


def _reference_time_and_attrs(
    data_config: PairedVideoLoaderConfig, out_names: list[str]
) -> tuple[np.ndarray, dict[str, dict[str, str]]]:
    """Read the input zarr's own time coordinate + variable attrs for the test
    subset, so the output's timestamps and metadata match the input exactly.
    """
    src = data_config.fine[0]
    if not isinstance(src, XarrayDataConfig):
        raise ValueError(
            "Expected an XarrayDataConfig for data.fine[0] for video inference, "
            f"got {type(src)}."
        )
    ds = xr.open_zarr(os.path.join(src.data_path, src.file_pattern))
    subset = src.subset
    if not isinstance(subset, TimeSlice):
        raise ValueError(
            "Expected a TimeSlice subset on data.fine[0] for video inference, "
            f"got {type(subset)}."
        )
    ds = ds.sel(time=slice(subset.start_time, subset.stop_time))
    time = ds["time"].values
    attrs = {name: dict(ds[name].attrs) for name in out_names if name in ds}
    return time, attrs


def _post_crop_latlon(batch) -> tuple[np.ndarray, np.ndarray]:
    """Lat/lon actually fed to the model, i.e. post lat_extent/lon_extent crop
    (unlike ``griddata.fine_coords``, which reflects the pre-crop domain).
    """
    lat = batch.fine.latlon_coordinates.lat[0].cpu().numpy()
    lon = batch.fine.latlon_coordinates.lon[0].cpu().numpy()
    return lat, lon


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


def _clip_write_slice(clip_start_time, time: np.ndarray, n_timesteps: int) -> slice:
    """Time-axis slice a clip should write to, given its start time.

    Tumbling clips share their boundary frame with the next clip, so every
    clip but the last writes only its first ``n_timesteps - 1`` frames; the
    last clip also writes its own trailing endpoint.
    """
    n_time = len(time)
    start_idx = int(np.searchsorted(time, clip_start_time))
    if start_idx >= n_time or time[start_idx] != clip_start_time:
        raise ValueError(
            f"Clip start time {clip_start_time} not found in the reference "
            "test time axis; data/config mismatch."
        )
    is_last_clip = start_idx + (n_timesteps - 1) == n_time - 1
    n_frames = n_timesteps if is_last_clip else n_timesteps - 1
    return slice(start_idx, start_idx + n_frames)


@dataclass
class VideoInferenceConfig:
    """Config for running test-set inference with a trained video PMD model.

    ``model`` and ``data`` are pasted verbatim from the training config's
    ``model:`` and ``test_data:`` blocks -- no new schema for those.

    Args:
        checkpoint_path: Path to the trained checkpoint to load.
        model: Model config, pasted verbatim from training's ``model:`` block.
        data: Data config, pasted verbatim from training's ``test_data:`` block.
        output_path: Path to write the output zarr store to.
        experiment_dir: Directory for logs.
        n_ensemble: Number of ensemble members to generate per clip.
        ensemble_chunk_size: Number of ensemble members to generate at once.
        use_ema: Whether to load the EMA (vs. raw) weights.
        max_batches: Cap on batches processed per rank; for smoke tests.
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
    max_batches: int | None = None
    overwrite: bool = False

    def configure_logging(self, log_filename: str) -> None:
        config = dataclasses.asdict(self)
        self.logging.configure_logging(
            self.experiment_dir, log_filename, config=config, resumable=True
        )

    def build_model(self, device: torch.device) -> VideoDiffusionModel:
        """Build the model from config and load trained weights.

        The training config used ``validate_using_ema: true``, so the checkpoint
        that was selected as "best" was evaluated with EMA-swapped weights. To
        reproduce that quality at inference time, load the raw state dict first
        (establishes buffers) and then overwrite the trainable params with the
        EMA shadow, exactly mirroring ``VideoTrainer._ema_context()``.
        """
        model = self.model.build()
        ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
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
        model = self.build_model(get_device())
        griddata = self.data.build_video(
            train=False, requirements=self.model.data_requirements, drop_last=False
        )
        return VideoInferenceRunner(model, griddata, self)


class VideoInferenceRunner:
    def __init__(
        self,
        model: VideoDiffusionModel,
        griddata: PairedVideoGriddedData,
        config: VideoInferenceConfig,
    ):
        self.model = model
        self.griddata = griddata
        self.config = config

    def run(self) -> None:
        dist = Distributed.get_instance()
        model, griddata, config = self.model, self.griddata, self.config
        logger.info(f"Number of parameters: {count_parameters(model.modules)}")

        time, var_attrs = _reference_time_and_attrs(config.data, model.out_names)
        n_time = len(time)
        n_timesteps = config.model.n_timesteps
        clip_stride = n_timesteps - 1  # tumbling clips share only their boundary frame

        # Observed at every clip boundary (0, clip_stride, 2*clip_stride, ...),
        # generated everywhere else.
        frame_source = np.ones(n_time, dtype=np.int8)
        frame_source[0::clip_stride] = 0

        batch_iterator = griddata.get_generator()
        first_batch = next(batch_iterator)
        lat, lon = _post_crop_latlon(first_batch)
        n_lat, n_lon = len(lat), len(lon)

        coords = {
            TIME_NAME: time,
            ENSEMBLE_NAME: np.arange(config.n_ensemble),
            LAT_NAME: lat,
            LON_NAME: lon,
        }
        chunks = determine_zarr_chunks(
            dims=DIMS,
            data_shape=(n_time, config.n_ensemble, n_lat, n_lon),
            bytes_per_element=4,
        )

        writer = ZarrWriter(
            path=config.output_path,
            dims=DIMS,
            coords=coords,
            data_vars=model.out_names,
            chunks=chunks,
            array_attributes=var_attrs,
            group_attributes={
                "description": (
                    "Test-set inference: endpoint-conditioned video diffusion "
                    "infilling, ensemble of independent noise draws."
                ),
                "checkpoint_path": config.checkpoint_path,
                "n_ensemble": str(config.n_ensemble),
                "use_ema": str(config.use_ema),
            },
            nondim_coords={
                "frame_source": xr.DataArray(frame_source, dims=[TIME_NAME]),
            },
            mode="w" if config.overwrite else "w-",
            time_calendar="julian",
            # Unwritten cells (e.g. a killed/retried run) stay NaN rather than
            # a silently-plausible-looking 0.
            fill_value=np.nan,
        )
        writer.initialize_store(data_dtype=np.float32)

        n_batches = len(griddata.loader)
        for i, batch in enumerate(itertools.chain([first_batch], batch_iterator)):
            if config.max_batches is not None and i >= config.max_batches:
                break

            remaining = config.n_ensemble
            ensemble_chunks: dict[str, list[torch.Tensor]] = {
                name: [] for name in model.out_names
            }
            while remaining > 0:
                n = min(config.ensemble_chunk_size, remaining)
                generated = model.generate(batch, n_samples=n)
                for name in model.out_names:
                    ensemble_chunks[name].append(generated[name])
                remaining -= n
            # (B, n_ensemble, T, H, W)
            full = {
                name: torch.cat(chunks_, dim=1)
                for name, chunks_ in ensemble_chunks.items()
            }
            full = _splice_observed_endpoints(full, batch.fine.data, config.n_ensemble)

            clip_times = batch.fine.time.values  # (B, T) cftime
            for b in range(clip_times.shape[0]):
                time_slice = _clip_write_slice(clip_times[b, 0], time, n_timesteps)
                n_frames_to_write = time_slice.stop - time_slice.start

                write_data = {
                    name: full[name][b, :, :n_frames_to_write]
                    .permute(
                        1, 0, 2, 3
                    )  # (n_ensemble, T', H, W) -> (T', n_ensemble, H, W)
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                    for name in model.out_names
                }
                writer.record_batch(
                    write_data,
                    position_slices={
                        TIME_NAME: time_slice,
                        ENSEMBLE_NAME: slice(0, config.n_ensemble),
                    },
                )

            logger.info(f"Rank {dist.rank}: batch {i + 1}/{n_batches} written")

        if dist.is_distributed():
            dist.barrier()
        logger.info(f"Completed inference. Output: {config.output_path}")


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

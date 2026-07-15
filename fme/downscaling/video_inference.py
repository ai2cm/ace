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
import logging
import os
from dataclasses import dataclass

import dacite
import numpy as np
import torch
import xarray as xr
import yaml

from fme.core.cli import prepare_directory
from fme.core.dataset.time import TimeSlice
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.ema import EMATracker
from fme.core.generics.trainer import count_parameters
from fme.core.logging_utils import LoggingConfig
from fme.core.writer import ZarrWriter
from fme.downscaling.data import PairedDataLoaderConfig
from fme.downscaling.inference.zarr_utils import determine_zarr_chunks
from fme.downscaling.video_models import VideoDiffusionModel, VideoDiffusionModelConfig

logger = logging.getLogger(__name__)

TIME_NAME = "time"
ENSEMBLE_NAME = "ensemble"
LAT_NAME = "latitude"
LON_NAME = "longitude"
DIMS = (TIME_NAME, ENSEMBLE_NAME, LAT_NAME, LON_NAME)


def load_video_model(
    model_config: VideoDiffusionModelConfig,
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> VideoDiffusionModel:
    """Build the model from config and load trained weights.

    The training config used ``validate_using_ema: true``, so the checkpoint
    that was selected as "best" was evaluated with EMA-swapped weights. To
    reproduce that quality at inference time, load the raw state dict first
    (establishes buffers) and then overwrite the trainable params with the
    EMA shadow, exactly mirroring ``VideoTrainer._ema_context()``.

    ``model.module`` is always wrapped (``DummyWrapper`` outside torchrun,
    ``DistributedDataParallel`` under it), both under the attribute name
    ``module``. DDP's own ``state_dict()``/``load_state_dict()`` transparently
    forward to that inner module (no prefix), which is what training's saved
    checkpoint contains -- but ``DummyWrapper`` does *not* override those
    methods, so loading directly into ``model.module`` only works under DDP.
    Loading into ``getattr(model.module, "module", model.module)`` instead
    reaches the raw net directly in both cases (mirrors the proven pattern in
    ``toy/eval_video_ckpt.py``).
    """
    model = model_config.build()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in ckpt["module"].items()
    }
    bare = getattr(model.module, "module", model.module)
    bare.load_state_dict(state_dict)
    if use_ema:
        if "ema" not in ckpt:
            raise ValueError(
                f"use_ema=True but checkpoint {checkpoint_path} has no 'ema' state."
            )
        ema = EMATracker.from_state(ckpt["ema"], model.modules)
        ema.copy_to(model.modules)
        logger.info("Loaded EMA weights for inference.")
    else:
        logger.info("Loaded raw (non-EMA) weights for inference.")
    model.module.eval()
    logger.info(
        f"Loaded checkpoint {checkpoint_path} "
        f"(epoch {ckpt.get('startEpoch')}, "
        f"best_valid_loss {ckpt.get('best_valid_loss')})"
    )
    return model


def _reference_time_and_attrs(
    data_config: PairedDataLoaderConfig, out_names: list[str]
) -> tuple[np.ndarray, dict[str, dict[str, str]]]:
    """Read the input zarr's own time coordinate + variable attrs for the test
    subset, so the output's timestamps and metadata match the input exactly.
    """
    src = data_config.fine[0]
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


@dataclass
class VideoInferenceConfig:
    """Config for running test-set inference with a trained video PMD model.

    ``model`` and ``data`` are pasted verbatim from the training config's
    ``model:`` and ``test_data:`` blocks -- no new schema for those.
    """

    checkpoint_path: str
    model: VideoDiffusionModelConfig
    data: PairedDataLoaderConfig
    output_path: str
    experiment_dir: str
    logging: LoggingConfig
    n_ensemble: int = 32
    ensemble_chunk_size: int = 8
    use_ema: bool = True
    # Cap the number of batches processed per rank; for smoke tests.
    max_batches: int | None = None
    # If True, overwrite an existing store at output_path (mode="w") instead
    # of the safe default (mode="w-", fail if it already exists). Use this
    # while iterating on a run that keeps failing/retrying; leave False once
    # a run is expected to succeed, so a completed store can't be clobbered
    # by accident.
    overwrite: bool = False

    def configure_logging(self, log_filename: str) -> None:
        config = dataclasses.asdict(self)
        self.logging.configure_logging(
            self.experiment_dir, log_filename, config=config, resumable=True
        )


def run_inference(config: VideoInferenceConfig) -> None:
    dist = Distributed.get_instance()
    device = get_device()

    model = load_video_model(
        config.model, config.checkpoint_path, device, use_ema=config.use_ema
    )
    logger.info(f"Number of parameters: {count_parameters(model.modules)}")

    griddata = config.data.build_video(
        train=False, requirements=config.model.data_requirements
    )

    time, var_attrs = _reference_time_and_attrs(config.data, model.out_names)
    n_time = len(time)
    n_timesteps = config.model.n_timesteps
    clip_stride = n_timesteps - 1  # tumbling clips share only their boundary frame

    # Observed at every clip boundary (0, clip_stride, 2*clip_stride, ...),
    # generated everywhere else. If the dataloader's drop_last truncates the
    # final partial batch, the tail of this array is simply never written and
    # will show up as a gap (fill value) in the output -- see run.sh / README
    # verification notes.
    frame_source = np.ones(n_time, dtype=np.int8)
    frame_source[0::clip_stride] = 0

    lat = griddata.fine_coords.coords["lat"]
    lon = griddata.fine_coords.coords["lon"]
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
    )
    writer.initialize_store(data_dtype=np.float32)

    n_batches = len(griddata.loader)
    for i, batch in enumerate(griddata.get_generator()):
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
        # (B, n_ensemble, T, H, W); splice in exact observed endpoints.
        full = {name: torch.cat(chunks_, dim=1) for name, chunks_ in ensemble_chunks.items()}
        for name in model.out_names:
            gt = batch.fine.data[name]  # (B, T, H, W)
            full[name][:, :, 0] = gt[:, None, 0].expand(-1, config.n_ensemble, -1, -1)
            full[name][:, :, -1] = gt[:, None, -1].expand(-1, config.n_ensemble, -1, -1)

        clip_times = batch.fine.time.values  # (B, T) cftime
        batch_size = clip_times.shape[0]
        for b in range(batch_size):
            start_idx = int(np.searchsorted(time, clip_times[b, 0]))
            if time[start_idx] != clip_times[b, 0]:
                raise ValueError(
                    f"Clip start time {clip_times[b, 0]} not found in the "
                    "reference test time axis; data/config mismatch."
                )
            is_last_clip = start_idx + (n_timesteps - 1) == n_time - 1
            n_frames_to_write = n_timesteps if is_last_clip else n_timesteps - 1
            time_slice = slice(start_idx, start_idx + n_frames_to_write)

            write_data = {
                name: full[name][b, :, :n_frames_to_write]
                .permute(1, 0, 2, 3)  # (n_ensemble, T', H, W) -> (T', n_ensemble, H, W)
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
    run_inference(inference_config)


def parse_args():
    parser = argparse.ArgumentParser(description="Video PMD test-set inference")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Distributed.context():
        main(args.config_path)

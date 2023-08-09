from typing import Dict, Mapping, Optional, Tuple

import torch
import numpy as np

from fme.core.distributed import Distributed
from fme.core.wandb import WandB

wandb = WandB.get_instance()


def get_gen_shape(gen_data: Mapping[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class _VideoData:
    """
    Record batches of video data and compute the mean.
    """

    def __init__(self, n_timesteps: int, dist: Optional[Distributed] = None):
        self._target_data: Optional[Dict[str, torch.Tensor]] = None
        self._gen_data: Optional[Dict[str, torch.Tensor]] = None
        self._n_timesteps = n_timesteps
        self._n_batches = torch.zeros([n_timesteps], dtype=torch.int32).cpu()
        if dist is None:
            dist = Distributed.get_instance()
        self._dist = dist

    @torch.no_grad()
    def record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        i_time_start: int,
    ):
        """
        Record a batch of data.

        Args:
            target_data: Dict of tensors of shape (n_samples, n_timesteps, ...)
            gen_data: Dict of tensors of shape (n_samples, n_timesteps, ...)
            i_time_start: Index of the first timestep in the batch.
        """
        if self._target_data is None:
            self._target_data = initialize_zeros_video_from_batch(
                target_data, self._n_timesteps
            )
        if self._gen_data is None:
            self._gen_data = initialize_zeros_video_from_batch(
                gen_data, self._n_timesteps
            )

        window_steps = next(iter(target_data.values())).shape[1]
        time_slice = slice(i_time_start, i_time_start + window_steps)
        for name, tensor in target_data.items():
            self._target_data[name][:, time_slice, ...] += tensor.cpu()
        for name, tensor in gen_data.items():
            self._gen_data[name][:, time_slice, ...] += tensor.cpu()

        self._n_batches[time_slice] += 1

    @torch.no_grad()
    def get(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self._gen_data is None or self._target_data is None:
            raise RuntimeError("No data recorded")
        target_data = {}
        gen_data = {}
        for name, tensor in self._target_data.items():
            target_data[name] = (tensor / self._n_batches[None, :, None, None]).mean(
                dim=0
            )
            target_data[name] = self._dist.reduce_mean(target_data[name])
        for name, tensor in self._gen_data.items():
            gen_data[name] = (tensor / self._n_batches[None, :, None, None]).mean(dim=0)
            gen_data[name] = self._dist.reduce_mean(gen_data[name])
        return gen_data, target_data


def initialize_zeros_video_from_batch(
    batch: Mapping[str, torch.Tensor], n_timesteps: int
):
    """
    Initialize a video of the same shape as the batch, but with all zeros and
    with n_timesteps timesteps.
    """
    video = {}
    for name, value in batch.items():
        shape = list(value.shape)
        shape[1] = n_timesteps
        video[name] = torch.zeros(shape, dtype=value.dtype).cpu()
    return video


class VideoAggregator:
    """Videos of state evolution."""

    def __init__(self, n_timesteps: int, dist: Optional[Distributed] = None):
        # This class exists instead of directly using VideoData
        # because we may want to record different kinds
        # of videos, e.g. for the best/worst series and for the mean.
        self._data = _VideoData(n_timesteps=n_timesteps, dist=dist)

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        i_time_start: int = 0,
    ):
        self._data.record_batch(
            target_data=target_data,
            gen_data=gen_data,
            i_time_start=i_time_start,
        )

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        gen_data, target_data = self._data.get()
        videos = {}
        for name in gen_data:
            videos[f"{label}/{name}"] = make_video(
                caption=(
                    f"Autoregressive (left) prediction and (right) target for {name}"
                ),
                gen=gen_data[name],
                target=target_data[name],
            )
        return videos


def make_video(
    caption: str,
    gen: torch.Tensor,
    target: Optional[torch.Tensor] = None,
):
    if target is None:
        video_data = np.expand_dims(gen.cpu().numpy(), axis=1)
    else:
        gen = np.expand_dims(gen.cpu().numpy(), axis=1)
        target = np.expand_dims(target.cpu().numpy(), axis=1)
        gap = np.zeros([gen.shape[0], 1, gen.shape[2], 10])
        video_data = np.concatenate([gen, gap, target], axis=-1)
    if target is None:
        data_min = video_data.min()
        data_max = video_data.max()
    else:
        # use target data to set the color scale
        data_min = target.min()
        data_max = target.max()
    # video data is brightness values on a 0-255 scale
    video_data = 255 * (video_data - data_min) / (data_max - data_min)
    video_data = np.minimum(video_data, 255)
    video_data = np.maximum(video_data, 0)
    caption += f"; vmin={data_min:.4g}, vmax={data_max:.4g}"
    return wandb.Video(
        video_data,
        caption=caption,
        fps=4,
    )

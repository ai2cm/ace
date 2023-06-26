from typing import Dict, Mapping, Optional, Tuple

import torch
import xarray as xr
import numpy as np

from fme.core.distributed import Distributed
from fme.core.wandb import WandB

wandb = WandB.get_instance()


def get_gen_shape(gen_data: Mapping[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class VideoData:
    """
    Record batches of video data and compute the mean.
    """

    def __init__(self, max_batches: Optional[int] = None):
        """
        Args:
            max_batches: Maximum number of batches to record.
                If None, record all batches.
        """
        if max_batches is not None and max_batches < 1:
            raise ValueError(f"max_batches must be at least 1, got {max_batches}")
        self._target_data: Optional[Dict[str, torch.Tensor]] = None
        self._gen_data: Optional[Dict[str, torch.Tensor]] = None
        self._n_batches = 0
        self._max_batches = max_batches

    @torch.no_grad()
    def record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
    ):
        if self._max_batches is None or self._n_batches < self._max_batches:
            if self._target_data is None:
                # make a copy of the data
                self._target_data = {
                    name: value.clone() for name, value in target_data.items()
                }
            else:
                for name, tensor in target_data.items():
                    self._target_data[name] += tensor
            if self._gen_data is None:
                self._gen_data = {
                    name: value.clone() for name, value in gen_data.items()
                }
            else:
                for name, tensor in gen_data.items():
                    self._gen_data[name] += tensor
        self._n_batches += 1

    def _get_data(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self._gen_data is None or self._target_data is None:
            raise RuntimeError("No data recorded")
        target_data = {}
        gen_data = {}
        dist = Distributed.get_instance()
        for name, tensor in self._target_data.items():
            target_data[name] = (tensor / self._n_batches).mean(dim=0)
            target_data[name] = dist.reduce_mean(target_data[name])
        for name, tensor in self._gen_data.items():
            gen_data[name] = (tensor / self._n_batches).mean(dim=0)
            gen_data[name] = dist.reduce_mean(gen_data[name])
        return gen_data, target_data

    @torch.no_grad()
    def get(self, label: str):
        gen_data, target_data = self._get_data()
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

    @torch.no_grad()
    def get_dataset(self, label: str = ""):
        if len(label) > 0:
            label = f"{label}_"
        gen_data, target_data = self._get_data()
        data_vars = {}
        all_names = set(gen_data.keys()).union(set(target_data.keys()))
        for name in all_names:
            data = []
            if name in gen_data:
                data.append(
                    xr.DataArray(
                        gen_data[name].cpu().numpy()[None, :],
                        dims=("source", "forecast_step", "lat", "lon"),
                        coords={"source": ["prediction"]},
                    )
                )
            if name in target_data:
                data.append(
                    xr.DataArray(
                        target_data[name].cpu().numpy()[None, :],
                        dims=("source", "forecast_step", "lat", "lon"),
                        coords={"source": ["target"]},
                    )
                )
            data_vars[f"{label}{name}"] = xr.concat(data, dim="source")
        return xr.Dataset(data_vars=data_vars)


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
    caption += f"; vmin={data_min:.2f}, vmax={data_max:.2f}"
    return wandb.Video(
        video_data,
        caption=caption,
        fps=4,
    )


class VideoAggregator:
    """Videos of state evolution."""

    def __init__(self):
        # This class exists instead of directly using VideoData
        # because we may want to record different kinds
        # of videos, e.g. for the best/worst series and for the mean.
        self._data = VideoData()

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
    ):
        self._data.record_batch(
            target_data=target_data,
            gen_data=gen_data,
        )

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        return self._data.get(label=label)

    def get_dataset(self, label: str):
        if len(label) > 0:
            label = f"{label}_"
        return self._data.get_dataset(label=f"{label}mean")

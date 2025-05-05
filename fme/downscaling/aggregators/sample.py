from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorMapping
from fme.core.wandb import Image, WandB

from ..metrics_and_maths import compute_rank
from ..typing_ import FineResCoarseResPair
from .main import ensure_trailing_slash


def _plot_spatial(data: torch.Tensor, coords: LatLonCoordinates, **kwargs):
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = plt.axes(projection=ccrs.PlateCarree())
    cmap = kwargs.get("cmap", None)

    if cmap is not None and cmap == "RdBu_r":
        coastline_color = "black"
    else:
        coastline_color = "white"
    ax.coastlines(color=coastline_color)

    lat, lon = coords.meshgrid
    res = ax.pcolormesh(
        lon.cpu().numpy(),
        lat.cpu().numpy(),
        data.detach().numpy(),
        transform=ccrs.PlateCarree(),
        **kwargs,
    )
    plt.colorbar(res, ax=ax, orientation="vertical")
    return fig


def _get_cmap_and_limits(data: torch.Tensor) -> tuple[str, float, float]:
    """
    Returns the colormap and limits for the data.
    """
    # cmap = "RdBu_r" if data.min() < 0 and data.max() > 0 else None
    cmap = "viridis"
    vmin, vmax = (
        np.percentile(data.cpu().detach().numpy(), 1),
        np.percentile(data.cpu().detach().numpy(), 99),
    )
    return cmap, vmin, vmax


def _get_caption(key: str, data: torch.Tensor) -> str:
    avg = data.mean()
    vmin, vmax = data.min(), data.max()
    caption = f"{key},  mean: {avg:.4g}, min: {vmin:.4g}, max: {vmax:.4g}"
    return caption


class SampleAggregator:
    """
    Aggregates samples generated in parallel across GPUs for visualization.
    This aggregator collects many samples into CPU memory, so keep that in
    mind when using large sample generation / batch sizes.  Intended for
    single-event evaluation over a single patch, not for large-scale
    evaluation.
    """

    def __init__(
        self,
        target: TensorMapping,
        coarse: TensorMapping,
        latlon_coordinates: FineResCoarseResPair[LatLonCoordinates],
        num_plot_samples: int = 8,
    ) -> None:
        self._target = target
        self._coarse = coarse
        self._num_plot_samples = num_plot_samples
        self._samples: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._latlon_coordinates = latlon_coordinates
        self._sample_dim = 1
        self._dist = Distributed.get_instance()

    @torch.no_grad()
    def record_batch(self, samples: TensorMapping) -> None:
        for k, v in samples.items():
            self._samples[k].append(v)

    def _get_wandb_image(
        self, data: torch.Tensor, key: str, coords: LatLonCoordinates, **plot_kwargs
    ) -> Image:
        fig = _plot_spatial(data, coords=coords, **plot_kwargs)
        caption = _get_caption(key, data)
        wandb = WandB.get_instance()
        image = wandb.Image(fig, caption=caption)
        plt.close(fig)
        return image

    def _get_sample_plots(
        self, samples: torch.Tensor, key: str, **plot_kwargs
    ) -> list[Any]:
        # batch, sample, lat, lon
        samples = samples[0, : self._num_plot_samples]
        if not plot_kwargs:
            cmap, vmin, vmax = _get_cmap_and_limits(samples)
            plot_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)

        images = [
            self._get_wandb_image(
                sample, key, self._latlon_coordinates.fine, **plot_kwargs
            )
            for sample in samples
        ]
        return images

    def _get_target_and_coarse_plots(
        self, key: str, **plot_kwargs
    ) -> Mapping[str, Any]:
        target = self._target[key]
        coarse = self._coarse[key]

        target_image = self._get_wandb_image(
            target.squeeze().cpu(),
            f"fine {key}",
            self._latlon_coordinates.fine,
            **plot_kwargs,
        )
        coarse_image = self._get_wandb_image(
            coarse.squeeze().cpu(),
            f"coarse {key}",
            self._latlon_coordinates.coarse,
            **plot_kwargs,
        )

        return {
            f"fine_target/{key}": target_image,
            f"coarse/{key}": coarse_image,
        }

    def _get_sample_rank_histogram(self, samples: torch.Tensor, key: str) -> Image:
        rank = compute_rank(self._target[key].cpu(), samples)
        nsamples = samples.shape[1]
        bins = np.arange(-0.5, nsamples + 0.6, 1)
        fig = plt.figure()
        plt.hist(rank.flatten(), bins=bins)
        plt.xlabel("Rank")
        wandb = WandB.get_instance()
        image = wandb.Image(fig, caption=f"{key} rank histogram")
        plt.close(fig)
        return image

    def _get_sample_ensemble_plots(
        self, samples: torch.Tensor, key: str, **plot_kwargs
    ) -> Mapping[str, Any]:
        ensemble_mean = samples.mean(axis=self._sample_dim).squeeze()
        mean_image = self._get_wandb_image(
            ensemble_mean,
            f"ensemble_mean/{key}",
            self._latlon_coordinates.fine,
            **plot_kwargs,
        )

        ensemble_std = samples.std(axis=self._sample_dim).squeeze()
        _, vmin, vmax = _get_cmap_and_limits(ensemble_std)
        std_image = self._get_wandb_image(
            ensemble_std,
            f"ensemble_std/{key}",
            self._latlon_coordinates.fine,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )

        rank_histogram = self._get_sample_rank_histogram(samples, key)

        return {
            f"ensemble_mean/{key}": mean_image,
            f"ensemble_std/{key}": std_image,
            f"rank_histogram/{key}": rank_histogram,
        }

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        ret = {}
        for k, v in self._samples.items():
            samples = torch.concat(v, dim=self._sample_dim)
            gathered_samples = self._dist.gather(samples)
            cmap, vmin, vmax = _get_cmap_and_limits(self._target[k])
            plot_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)

            # subselection of image plots
            if gathered_samples is not None:
                gathered_samples = torch.concat(
                    gathered_samples, dim=self._sample_dim
                ).cpu()
                ret[f"generation_samples/{k}"] = self._get_sample_plots(
                    gathered_samples, k, **plot_kwargs
                )
                ret.update(
                    self._get_sample_ensemble_plots(gathered_samples, k, **plot_kwargs)
                )
                ret.update(self._get_target_and_coarse_plots(k, **plot_kwargs))
        prefix = ensure_trailing_slash(prefix)
        ret = {f"{prefix}{k}": v for k, v in ret.items()}
        return ret

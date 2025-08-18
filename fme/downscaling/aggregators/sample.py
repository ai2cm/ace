from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

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
    Aggregates generated samples generated in parallel across GPUs for visualization.
    This aggregator collects many samples into CPU memory, so keep that in
    mind when using large sample generation / batch sizes.  Intended for
    single-event evaluation over a single patch, not for large-scale
    evaluation.
    This aggregator is designed to be used on data with batch size 1.
    """

    def __init__(
        self,
        coarse: TensorMapping,
        latlon_coordinates: FineResCoarseResPair[LatLonCoordinates],
        num_plot_samples: int = 8,
    ) -> None:
        self._coarse = coarse
        self._num_plot_samples = num_plot_samples
        self._samples: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._latlon_coordinates = latlon_coordinates
        self._sample_dim = 1
        self._dist = Distributed.get_instance()
        self._gathered_samples: dict[str, torch.Tensor] | None = None

    def _validate_batch_size(self, samples: TensorMapping) -> None:
        """
        Validates that the batch size of the samples is 1.
        """
        for k, v in samples.items():
            if len(v.shape) != 4:
                raise ValueError(
                    f"Expected 4d tensor for {k} with dims (batch, sample, lat, lon), "
                    f"got tensor with shape {v.shape}."
                )
            if v.shape[0] != 1:
                raise ValueError(f"Expected batch size of 1 for {k}, got {v.shape[0]}")

    @torch.no_grad()
    def record_batch(self, samples: TensorMapping) -> None:
        if self._gathered_samples is not None:
            raise RuntimeError("Cannot record new samples after gathering samples. ")
        self._validate_batch_size(samples)
        for k, v in samples.items():
            self._samples[k].append(v)

    def get_wandb_image(
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
            self.get_wandb_image(
                sample, key, self._latlon_coordinates.fine, **plot_kwargs
            )
            for sample in samples
        ]
        return images

    @property
    def gathered_samples(self):
        """
        Gathers all samples from all GPUs into CPU memory.
        """
        if self._gathered_samples is not None:
            return self._gathered_samples
        else:
            gathered_samples = {}
            for k, v in self._samples.items():
                samples = torch.concat(v, dim=self._sample_dim)
                gathered_tensor = self._dist.gather(samples)
                if gathered_tensor is not None:
                    gathered_samples[k] = torch.concat(
                        gathered_tensor, dim=self._sample_dim
                    ).cpu()
            self._gathered_samples = gathered_samples
            return self._gathered_samples

    def _get_sample_ensemble_plots(
        self, samples: torch.Tensor, key: str, **plot_kwargs
    ) -> Mapping[str, Any]:
        ensemble_mean = samples.mean(axis=self._sample_dim).squeeze()
        mean_image = self.get_wandb_image(
            ensemble_mean,
            f"ensemble_mean/{key}",
            self._latlon_coordinates.fine,
            **plot_kwargs,
        )

        ensemble_std = samples.std(axis=self._sample_dim).squeeze()
        _, vmin, vmax = _get_cmap_and_limits(ensemble_std)
        std_image = self.get_wandb_image(
            ensemble_std,
            f"ensemble_std/{key}",
            self._latlon_coordinates.fine,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        return {
            f"ensemble_mean/{key}": mean_image,
            f"ensemble_std/{key}": std_image,
        }

    def get_wandb_single_var(
        self, key: str, samples: torch.Tensor, **plot_kwargs
    ) -> dict[str, Any]:
        ret: dict[str, Any] = {}
        # subselection of image plots
        ret[f"generation_samples/{key}"] = self._get_sample_plots(
            samples, key, **plot_kwargs
        )
        ret.update(self._get_sample_ensemble_plots(samples, key, **plot_kwargs))
        coarse_image = self.get_wandb_image(
            self._coarse[key].squeeze().cpu(),
            f"coarse {key}",
            self._latlon_coordinates.coarse,
            **plot_kwargs,
        )
        ret.update({f"coarse/{key}": coarse_image})
        return ret

    def get_wandb(self, prefix: str = "") -> dict[str, Any]:
        ret: dict[str, Any] = {}
        for k, tensor in self.gathered_samples.items():
            cmap, vmin, vmax = _get_cmap_and_limits(tensor)
            plot_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            ret.update(self.get_wandb_single_var(k, tensor, **plot_kwargs))
        prefix = ensure_trailing_slash(prefix)
        ret = {f"{prefix}{k}": v for k, v in ret.items()}
        return ret

    def get_dataset(self) -> xr.Dataset:
        latgrid, longrid = self._latlon_coordinates.fine.meshgrid
        lat, lon = latgrid.cpu().numpy()[:, 0], longrid.cpu().numpy()[0, :]
        ds = xr.Dataset()

        for k, v in self.gathered_samples.items():
            nsamples = v.shape[self._sample_dim]
            # Remove batch dimension before saving
            data = v.squeeze()
            ds[f"{k}_predicted"] = xr.DataArray(
                data,
                dims=["sample", "lat", "lon"],
                coords={
                    "lat": lat,
                    "lon": lon,
                    "sample": np.arange(nsamples),
                },
            )
        return ds


class PairedSampleAggregator:
    """
    Aggregates generated samples and paired targets generated in parallel
    `across GPUs for visualization.
    This aggregator collects many samples into CPU memory, so keep that in
    mind when using large sample generation / batch sizes.  Intended for
    single-event evaluation over a single patch, not for large-scale
    evaluation.
    This aggregator is designed to be used on data with batch size 1.
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
        self._gathered_samples: dict[str, torch.Tensor] | None = None
        self._sample_agg = SampleAggregator(
            coarse=coarse,
            latlon_coordinates=latlon_coordinates,
            num_plot_samples=num_plot_samples,
        )

    @torch.no_grad()
    def record_batch(self, samples: TensorMapping) -> None:
        self._sample_agg.record_batch(samples)

    def _get_sample_rank_histogram(self, samples: torch.Tensor, key: str) -> Image:
        rank = compute_rank(self._target[key].cpu(), samples)
        nsamples = samples.shape[self._sample_dim]
        bins = np.arange(-0.5, nsamples + 0.6, 1)
        fig = plt.figure()
        plt.hist(rank.flatten(), bins=bins)
        plt.xlabel("Rank")
        wandb = WandB.get_instance()
        image = wandb.Image(fig, caption=f"{key} rank histogram")
        plt.close(fig)
        return image

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        ret = {}
        for k, tensor in self._sample_agg.gathered_samples.items():
            cmap, vmin, vmax = _get_cmap_and_limits(self._target[k])
            plot_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            ret.update(
                {f"rank_histogram/{k}": self._get_sample_rank_histogram(tensor, k)}
            )
            target_image = self._sample_agg.get_wandb_image(
                self._target[k].squeeze().cpu(),
                f"fine {k}",
                self._latlon_coordinates.fine,
                **plot_kwargs,
            )
            ret.update({f"fine_target/{k}": target_image})
            unpaired_ret = self._sample_agg.get_wandb_single_var(
                k, tensor, **plot_kwargs
            )
            ret.update(unpaired_ret)
        prefix = ensure_trailing_slash(prefix)
        ret = {f"{prefix}{k}": v for k, v in ret.items()}

        return ret

    def get_dataset(self) -> xr.Dataset:
        latgrid, longrid = self._latlon_coordinates.fine.meshgrid
        lat, lon = latgrid.cpu().numpy()[:, 0], longrid.cpu().numpy()[0, :]
        ds = self._sample_agg.get_dataset()

        for k in self._sample_agg.gathered_samples:
            ds[f"{k}_target"] = xr.DataArray(
                self._target[k].squeeze().cpu().numpy(),
                dims=["lat", "lon"],
                coords={"lat": lat, "lon": lon},
            )
        return ds

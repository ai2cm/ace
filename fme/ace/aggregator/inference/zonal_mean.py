import dataclasses
from collections.abc import Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import Image, WandB

from ..plotting import get_cmap_limits, plot_imshow


@dataclasses.dataclass
class _RawData:
    datum: torch.Tensor
    caption: str
    metadata: VariableMetadata
    vmin: float | None = None
    vmax: float | None = None
    cmap: str | None = None

    def get_image(self) -> Image:
        # images are y, x from upper left corner
        # data is time, lat
        # we want lat on y-axis (increasing upward) and time on x-axis
        # so transpose and flip along lat axis
        datum = np.flip(self.datum.transpose(), axis=0)
        fig = plot_imshow(
            datum, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, flip_lat=False
        )
        wandb = WandB.get_instance()
        wandb_image = wandb.Image(fig, caption=self.caption)
        plt.close(fig)
        return wandb_image


class ZonalMeanAggregator:
    """Images of the zonal-mean state as a function of latitude and time.

    This aggregator keeps track of the generated and target zonal-mean state,
    then generates zonal-mean (Hovmoller) images when logs are retrieved.
    The zonal-mean images are averaged across the sample dimension.
    """

    _captions = {
        "error": (
            "{name} zonal-mean error (generated - target) [{units}], "
            "x-axis is time increasing to right, y-axis is latitude increasing upward"
        ),
        "gen": (
            "{name} zonal-mean generated [{units}], "
            "x-axis is time increasing to right, y-axis is latitude increasing upward"
        ),
    }

    def __init__(
        self,
        zonal_mean: Callable[[torch.Tensor], torch.Tensor],
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        """
        Args:
            zonal_mean: Function that computes the zonal mean of a tensor.
            n_timesteps: Number of timesteps of inference that will be run.
            variable_metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
        """
        self._n_timesteps = n_timesteps
        self._dist = Distributed.get_instance()
        if variable_metadata is None:
            self._variable_metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._variable_metadata = variable_metadata

        self._target_data: TensorDict | None = None
        self._gen_data: TensorDict | None = None
        self._n_batches = torch.zeros(
            n_timesteps, dtype=torch.int32, device=get_device()
        )[None, :, None]  # sample, time, lat
        self._zonal_mean = zonal_mean

    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
        i_time_start: int,
    ):
        if self._target_data is None:
            self._target_data = self._initialize_zeros_zonal_mean_from_batch(
                target_data, self._n_timesteps
            )
        if self._gen_data is None:
            self._gen_data = self._initialize_zeros_zonal_mean_from_batch(
                gen_data, self._n_timesteps
            )

        window_steps = next(iter(target_data.values())).shape[1]
        time_slice = slice(i_time_start, i_time_start + window_steps)
        # we can average along longitude without area weighting
        for name, tensor in target_data.items():
            if name in self._target_data:
                self._target_data[name][:, time_slice, :] += self._zonal_mean(tensor)
        for name, tensor in gen_data.items():
            if name in self._gen_data:
                self._gen_data[name][:, time_slice, :] += self._zonal_mean(tensor)
        self._n_batches[:, time_slice, :] += 1

    def _get_data(self) -> dict[str, _RawData]:
        if self._gen_data is None or self._target_data is None:
            raise RuntimeError("No data recorded")
        sample_dim = 0
        data: dict[str, _RawData] = {}
        sorted_names = sorted(list(self._gen_data.keys()))
        for name in sorted_names:
            gen = (
                self._dist.reduce_mean(self._gen_data[name] / self._n_batches)
                .mean(sample_dim)
                .cpu()
                .numpy()
            )
            error = (
                self._dist.reduce_mean(
                    (self._gen_data[name] - self._target_data[name]) / self._n_batches
                )
                .mean(sample_dim)
                .cpu()
                .numpy()
            )

            metadata = self._variable_metadata.get(
                name, VariableMetadata("unknown_units", name)
            )
            vmin, vmax = get_cmap_limits(gen)
            data[f"gen/{name}"] = _RawData(
                datum=gen,
                caption=self._get_caption("gen", name, vmin, vmax),
                # generated data is not considered to have units
                metadata=VariableMetadata(units="", long_name=metadata.long_name),
            )
            vmin, vmax = get_cmap_limits(error, diverging=True)
            data[f"error/{name}"] = _RawData(
                datum=error,
                caption=self._get_caption("error", name, vmin, vmax),
                metadata=metadata,
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r",
            )

        return data

    def get_logs(self, label: str) -> dict[str, Image]:
        logs = {}
        data = self._get_data()
        for key, datum in data.items():
            logs[f"{label}/{key}"] = datum.get_image()
        return logs

    def get_dataset(self) -> xr.Dataset:
        data = {
            k.replace("/", "-"): xr.DataArray(
                v.datum, dims=("forecast_step", "lat"), attrs=v.metadata._asdict()
            )
            for k, v in self._get_data().items()
        }

        ret = xr.Dataset(data)
        return ret

    def _get_caption(self, key: str, varname: str, vmin: float, vmax: float) -> str:
        if varname in self._variable_metadata:
            caption_name = self._variable_metadata[varname].long_name
            units = self._variable_metadata[varname].units
        else:
            caption_name, units = varname, "unknown_units"
        caption = self._captions[key].format(name=caption_name, units=units)
        caption += f" vmin={vmin:.4g}, vmax={vmax:.4g}."
        return caption

    @staticmethod
    def _initialize_zeros_zonal_mean_from_batch(
        data: TensorMapping, n_timesteps: int, lat_dim: int = 2
    ) -> TensorDict:
        return {
            name: torch.zeros(
                (tensor.shape[0], n_timesteps, tensor.shape[lat_dim]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for name, tensor in data.items()
        }

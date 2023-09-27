import dataclasses
from typing import Dict, Mapping, Optional

import numpy as np
import torch
from wandb import Image

from fme.core.data_loading.typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.wandb import WandB

from .image_scaling import scale_image

wandb = WandB.get_instance()

import xarray as xr


@dataclasses.dataclass
class _RawData:
    datum: np.ndarray
    caption: str
    metadata: VariableMetadata

    def get_image(self):
        # images are y, x from upper left corner
        # data is time, lat
        # we want lat on y-axis (increasing upward) and time on x-axis
        # so transpose and flip along lat axis
        datum = np.flip(self.datum.transpose(), axis=0)
        datum = scale_image(datum)
        return wandb.Image(datum, caption=self.caption)


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
        n_timesteps: int,
        dist: Optional[Distributed] = None,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ):
        """
        Args:
            n_timesteps: Number of timesteps of inference that will be run.
            dist: Distributed object to use for communication.
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
        """
        self._n_timesteps = n_timesteps
        if dist is None:
            self._dist = Distributed.get_instance()
        else:
            self._dist = dist
        if metadata is None:
            self._metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._metadata = metadata

        self._target_data: Optional[Dict[str, torch.Tensor]] = None
        self._gen_data: Optional[Dict[str, torch.Tensor]] = None
        self._n_batches = torch.zeros(
            n_timesteps, dtype=torch.int32, device=get_device()
        )[
            None, :, None
        ]  # sample, time, lat

    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        i_time_start: int,
    ):
        lon_dim = 3
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
            self._target_data[name][:, time_slice, :] += tensor.mean(dim=lon_dim)
        for name, tensor in gen_data.items():
            self._gen_data[name][:, time_slice, :] += tensor.mean(dim=lon_dim)
        self._n_batches[:, time_slice, :] += 1

    def _get_data(self) -> Dict[str, _RawData]:
        if self._gen_data is None or self._target_data is None:
            raise RuntimeError("No data recorded")
        sample_dim = 0
        data: Dict[str, _RawData] = {}
        for name in self._gen_data.keys():
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

            metadata = self._metadata.get(name, VariableMetadata("unknown_units", name))
            data[f"gen/{name}"] = _RawData(
                datum=gen,
                caption=self._get_caption("gen", name, gen),
                # generated data is not considered to have units
                metadata=VariableMetadata(units="", long_name=metadata.long_name),
            )
            data[f"error/{name}"] = _RawData(
                datum=error,
                caption=self._get_caption("error", name, error),
                metadata=metadata,
            )

        return data

    def get_logs(self, label: str) -> Dict[str, Image]:
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

    def _get_caption(self, caption_key: str, varname: str, data: torch.Tensor) -> str:
        if varname in self._metadata:
            caption_name = self._metadata[varname].long_name
            units = self._metadata[varname].units
        else:
            caption_name, units = varname, "unknown_units"
        caption = self._captions[caption_key].format(name=caption_name, units=units)
        caption += f" vmin={data.min():.4g}, vmax={data.max():.4g}."
        return caption

    @staticmethod
    def _initialize_zeros_zonal_mean_from_batch(
        data: Mapping[str, torch.Tensor], n_timesteps: int, lat_dim: int = 2
    ) -> Dict[str, torch.Tensor]:
        return {
            name: torch.zeros(
                (tensor.shape[0], n_timesteps, tensor.shape[lat_dim]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for name, tensor in data.items()
        }

import dataclasses
import logging
from collections.abc import Callable, Mapping

import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import Image

from ..plotting import plot_paneled_data


@dataclasses.dataclass
class _RawData:
    datum: torch.Tensor
    caption: str
    metadata: VariableMetadata
    target_datum: torch.Tensor | None = None
    diverging: bool = False

    def get_image(self) -> Image:
        # images are y, x from upper left corner
        # data is time, lat
        # we want lat on y-axis and time on x-axis
        datum = self.datum.transpose()
        if self.target_datum is not None:
            target_datum = self.target_datum.transpose()
            return plot_paneled_data(
                [[datum], [target_datum]],
                diverging=self.diverging,
                caption=self.caption,
            )
        else:
            return plot_paneled_data(
                [[datum]],
                diverging=self.diverging,
                caption=self.caption,
            )


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
            "{name} zonal-mean; "
            "(top) generated and (bottom) target [{units}], "
            "x-axis is time increasing to right, y-axis is latitude increasing upward"
        ),
    }
    _max_matplotlib_size = 2**15  # max number of timesteps before breaking matplotlib
    _time_dim = 1

    def __init__(
        self,
        zonal_mean: Callable[[torch.Tensor], torch.Tensor],
        n_timesteps: int,
        zonal_mean_max_size: int,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        """
        Args:
            zonal_mean: Function that computes the zonal mean of a tensor.
            n_timesteps: Number of timesteps of inference that will be run.
            variable_metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            zonal_mean_max_size: Maximum number of timesteps to log zonal mean images.
                If zonal_mean_max_size > _max_matplotlib_size, it will be set to
                _max_matplotlib_size.
        """
        if zonal_mean_max_size > self._max_matplotlib_size:
            logging.warning(
                "Zonal mean images support images with a maximum of "
                f"{self._max_matplotlib_size} timesteps. "
                f"Coarsening to {self._max_matplotlib_size} timesteps."
            )
            zonal_mean_max_size = self._max_matplotlib_size

        if zonal_mean_max_size > n_timesteps:
            logging.warning(
                f"Zonal mean max size {zonal_mean_max_size} is greater than "
                f"the number of timesteps {n_timesteps}. Setting zonal mean max size "
                "to the number of timesteps."
            )
            zonal_mean_max_size = n_timesteps

        self._max_size = zonal_mean_max_size

        if n_timesteps > self._max_size:
            # warn users that we are automatically coarsening the time dimension of
            # zonal mean images to avoid memory issues
            logging.info(
                f"Automatic time coarsening is enabled for zonal mean images due to "
                f"timesteps being greater than {self._max_size}. If you want to "
                f"override this, set log_zonal_mean_images to a value up to "
                f"{self._max_matplotlib_size}."
            )
            self.time_coarsening_factor = np.ceil(n_timesteps / self._max_size).astype(
                int
            )
            self._n_timesteps = n_timesteps // self.time_coarsening_factor

            logging.info(
                f"Zonal mean time coarsening factor is  "
                f"{self.time_coarsening_factor} with image size of {self._n_timesteps}."
            )
        else:
            self.time_coarsening_factor = 1
            self._n_timesteps = n_timesteps

        self._dist = Distributed.get_instance()

        if variable_metadata is None:
            self._variable_metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._variable_metadata = variable_metadata

        self._target_data: TensorDict | None = None
        self._gen_data: TensorDict | None = None
        self._n_batches = torch.zeros(
            self._n_timesteps, dtype=torch.int32, device=get_device()
        )[None, :, None]  # sample, time, lat
        self._zonal_mean = zonal_mean
        self.last_step = 0
        self.logged_batch_skip = False
        self._buffer_target: TensorDict | None = None
        self._buffer_gen: TensorDict | None = None

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

        if window_steps < self.time_coarsening_factor:
            # skip this batch if its not possible to coarsen
            if not self.logged_batch_skip:
                logging.info(
                    f"Skipping batch with {window_steps} steps because it is less than "
                    f"the zonal mean time coarsening factor of "
                    f"{self.time_coarsening_factor}."
                )
                self.logged_batch_skip = True
            return

        # if we have a buffer that means we didnt record the last batch
        if self._buffer_gen:
            start_idx = self.last_step
        else:
            start_idx = i_time_start // self.time_coarsening_factor

        time_slice = slice(
            start_idx,
            (i_time_start + window_steps) // self.time_coarsening_factor,
        )

        self.last_step = (i_time_start + window_steps) // self.time_coarsening_factor

        buffer_size = (
            i_time_start + window_steps
        ) - self.last_step * self.time_coarsening_factor

        buffer = {}
        for name, tensor in target_data.items():
            if name in self._target_data:
                if self._buffer_target:
                    tensor = torch.cat(
                        [
                            self._buffer_target[name],
                            tensor[:, 0 : window_steps - buffer_size, :],
                        ],
                        dim=self._time_dim,
                    )
                self._target_data[name][:, time_slice, :] += self._coarsen_tensor(
                    self._zonal_mean(tensor)
                )
                if buffer_size > 0:
                    buffer[name] = tensor[:, -buffer_size:, :]
        self._buffer_target = buffer

        buffer = {}
        for name, tensor in gen_data.items():
            if name in self._gen_data:
                if self._buffer_gen:
                    tensor = torch.cat(
                        [
                            self._buffer_gen[name],
                            tensor[:, 0 : window_steps - buffer_size, :],
                        ],
                        dim=self._time_dim,
                    )
                self._gen_data[name][:, time_slice, :] += self._coarsen_tensor(
                    self._zonal_mean(tensor)
                )
                if buffer_size > 0:
                    buffer[name] = tensor[:, -buffer_size:, :]
        self._buffer_gen = buffer

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
            target = (
                self._dist.reduce_mean(self._target_data[name] / self._n_batches)
                .mean(sample_dim)
                .cpu()
                .numpy()
            )
            error = gen - target

            metadata = self._variable_metadata.get(
                name, VariableMetadata("unknown_units", name)
            )
            data[f"gen/{name}"] = _RawData(
                datum=gen,
                target_datum=target,
                caption=self._get_caption("gen", name),
                metadata=metadata,
                diverging=False,
            )
            data[f"error/{name}"] = _RawData(
                datum=error,
                caption=self._get_caption("error", name),
                metadata=metadata,
                diverging=True,
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

    def _get_caption(self, key: str, varname: str) -> str:
        if varname in self._variable_metadata:
            caption_name = self._variable_metadata[varname].long_name
            units = self._variable_metadata[varname].units
        else:
            caption_name, units = varname, "unknown_units"
        caption = self._captions[key].format(name=caption_name, units=units)
        return caption

    def _coarsen_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Coarsen tensor along a given axis by a given factor."""
        return tensor.unfold(
            dimension=self._time_dim,
            size=self.time_coarsening_factor,
            step=self.time_coarsening_factor,
        ).mean(dim=-1)

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

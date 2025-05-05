import dataclasses
from collections.abc import Mapping

import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import WandB

wandb = WandB.get_instance()


def _get_gen_shape(gen_data: TensorMapping):
    for name in gen_data:
        return gen_data[name].shape
    raise ValueError("No data in gen_data")


@dataclasses.dataclass
class _ErrorData:
    rmse: TensorDict
    min_err: TensorDict
    max_err: TensorDict


class _ErrorVideoData:
    """
    Record batches of video data and compute statistics on the error.
    """

    def __init__(self, n_timesteps: int):
        self._mse_data: TensorDict | None = None
        self._min_err_data: TensorDict | None = None
        self._max_err_data: TensorDict | None = None
        self._n_timesteps = n_timesteps
        self._n_batches = torch.zeros([n_timesteps], dtype=torch.int32).cpu()
        self._dist = Distributed.get_instance()

    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        i_time_start: int,
    ):
        """
        Record a batch of data.

        Args:
            target_data: Dict of tensors of shape (n_samples, n_timesteps, ...)
            gen_data: Dict of tensors of shape (n_samples, n_timesteps, ...)
            i_time_start: Index of the first timestep in the batch.
        """
        if self._mse_data is None:
            self._mse_data = _initialize_video_from_batch(gen_data, self._n_timesteps)
        if self._min_err_data is None:
            self._min_err_data = _initialize_video_from_batch(
                gen_data, self._n_timesteps, fill_value=np.inf
            )
        if self._max_err_data is None:
            self._max_err_data = _initialize_video_from_batch(
                gen_data, self._n_timesteps, fill_value=-np.inf
            )

        window_steps = next(iter(target_data.values())).shape[1]
        time_slice = slice(i_time_start, i_time_start + window_steps)
        for name, gen_tensor in gen_data.items():
            target_tensor = target_data[name]
            error_tensor = (gen_tensor - target_tensor).cpu()
            self._mse_data[name][time_slice, ...] += torch.mean(
                torch.square(error_tensor), dim=0
            )
            self._min_err_data[name][time_slice, ...] = torch.minimum(
                self._min_err_data[name][time_slice, ...], error_tensor.min(dim=0)[0]
            )
            self._max_err_data[name][time_slice, ...] = torch.maximum(
                self._max_err_data[name][time_slice, ...], error_tensor.max(dim=0)[0]
            )

        self._n_batches[time_slice] += 1

    @torch.no_grad()
    def get(
        self,
    ) -> _ErrorData:
        if (
            self._mse_data is None
            or self._min_err_data is None
            or self._max_err_data is None
        ):
            raise RuntimeError("No data recorded")
        rmse_data = {}
        min_err_data = {}
        max_err_data = {}
        for name in sorted(self._mse_data):
            tensor = self._mse_data[name]
            mse = (tensor / self._n_batches[None, :, None, None]).mean(dim=0)
            mse = self._dist.reduce_mean(mse)
            rmse_data[name] = torch.sqrt(mse)
        for name in sorted(self._min_err_data):
            min_err_data[name] = self._dist.reduce_min(self._min_err_data[name])
        for name in sorted(self._max_err_data):
            max_err_data[name] = self._dist.reduce_max(self._max_err_data[name])
        return _ErrorData(rmse_data, min_err_data, max_err_data)


class _MeanVideoData:
    """
    Record batches of video data and compute the mean.
    """

    def __init__(self, n_timesteps: int):
        self._target_data: TensorDict | None = None
        self._gen_data: TensorDict | None = None
        self._n_timesteps = n_timesteps
        self._n_batches = torch.zeros([n_timesteps], dtype=torch.int32).cpu()
        self._dist = Distributed.get_instance()

    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
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
            self._target_data = _initialize_video_from_batch(
                target_data, self._n_timesteps
            )
        if self._gen_data is None:
            self._gen_data = _initialize_video_from_batch(gen_data, self._n_timesteps)

        window_steps = next(iter(target_data.values())).shape[1]
        time_slice = slice(i_time_start, i_time_start + window_steps)
        for name, tensor in target_data.items():
            self._target_data[name][time_slice, ...] += tensor.mean(dim=0).cpu()
        for name, tensor in gen_data.items():
            self._gen_data[name][time_slice, ...] += tensor.mean(dim=0).cpu()

        self._n_batches[time_slice] += 1

    @torch.no_grad()
    def get(self) -> tuple[TensorDict, TensorDict]:
        if self._gen_data is None or self._target_data is None:
            raise RuntimeError("No data recorded")
        target_data = {}
        gen_data = {}
        for name in sorted(self._target_data):
            tensor = self._target_data[name]
            target_data[name] = tensor / self._n_batches[:, None, None]
            target_data[name] = self._dist.reduce_mean(target_data[name])
        for name in sorted(self._gen_data):
            tensor = self._gen_data[name]
            gen_data[name] = tensor / self._n_batches[:, None, None]
            gen_data[name] = self._dist.reduce_mean(gen_data[name])
        return gen_data, target_data


class _VarianceVideoData:
    """
    Record batches of video data and compute the variance.
    """

    def __init__(self, n_timesteps: int):
        self._target_means: TensorDict | None = None
        self._gen_means: TensorDict | None = None
        self._target_squares: TensorDict | None = None
        self._gen_squares: TensorDict | None = None
        self._n_timesteps = n_timesteps
        self._n_batches = torch.zeros([n_timesteps], dtype=torch.int32).cpu()
        self._dist = Distributed.get_instance()

    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        i_time_start: int,
    ):
        """
        Record a batch of data.

        Args:
            target_data: Dict of tensors of shape (n_samples, n_timesteps, ...)
            gen_data: Dict of tensors of shape (n_samples, n_timesteps, ...)
            i_time_start: Index of the first timestep in the batch.
        """
        if self._target_means is None:
            self._target_means = _initialize_video_from_batch(
                target_data, self._n_timesteps
            )
        if self._gen_means is None:
            self._gen_means = _initialize_video_from_batch(gen_data, self._n_timesteps)
        if self._target_squares is None:
            self._target_squares = _initialize_video_from_batch(
                target_data, self._n_timesteps
            )

        if self._gen_squares is None:
            self._gen_squares = _initialize_video_from_batch(
                gen_data, self._n_timesteps
            )

        window_steps = next(iter(target_data.values())).shape[1]
        time_slice = slice(i_time_start, i_time_start + window_steps)
        for name, tensor in target_data.items():
            self._target_means[name][time_slice, ...] += tensor.mean(dim=0).cpu()
            self._target_squares[name][time_slice, ...] += (tensor**2).mean(dim=0).cpu()
        for name, tensor in gen_data.items():
            self._gen_means[name][time_slice, ...] += tensor.mean(dim=0).cpu()
            self._gen_squares[name][time_slice, ...] += (tensor**2).mean(dim=0).cpu()
        self._n_batches[time_slice] += 1

    @torch.no_grad()
    def get(self) -> tuple[TensorDict, TensorDict]:
        if (
            self._gen_means is None
            or self._target_means is None
            or self._gen_squares is None
            or self._target_squares is None
        ):
            raise RuntimeError("No data recorded")
        target_data = {}
        gen_data = {}
        # calculate variance as E[X^2] - E[X]^2
        for name in sorted(self._target_means):
            tensor = self._target_means[name]
            mean = tensor / self._n_batches[:, None, None]
            mean = self._dist.reduce_mean(mean)
            square = self._target_squares[name] / self._n_batches[:, None, None]
            square = self._dist.reduce_mean(square)
            target_data[name] = square - mean**2
        for name in sorted(self._gen_means):
            tensor = self._gen_means[name]
            mean = tensor / self._n_batches[:, None, None]
            mean = self._dist.reduce_mean(mean)
            square = self._gen_squares[name] / self._n_batches[:, None, None]
            square = self._dist.reduce_mean(square)
            gen_data[name] = square - mean**2
        return gen_data, target_data


def _initialize_video_from_batch(
    batch: TensorMapping, n_timesteps: int, fill_value: float = 0.0
):
    """
    Initialize a video of the same shape as the batch, but with all valeus equal
    to fill_value and with n_timesteps timesteps.
    """
    video = {}
    for name, value in batch.items():
        shape = list(value.shape[1:])
        shape[0] = n_timesteps
        video[name] = torch.zeros(shape, dtype=torch.double).cpu()
        video[name][:, ...] = fill_value
    return video


@dataclasses.dataclass
class _MaybePairedVideoData:
    caption: str
    gen: torch.Tensor
    units: str | None
    long_name: str | None
    target: torch.Tensor | None = None

    def make_video(self):
        return _make_video(
            caption=self.caption,
            gen=self.gen,
            target=self.target,
        )


class VideoAggregator:
    """Videos of state evolution."""

    def __init__(
        self,
        n_timesteps: int,
        enable_extended_videos: bool,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        """
        Args:
            n_timesteps: Number of timesteps of inference that will be run.
            enable_extended_videos: Whether to log videos of statistical
                metrics of state evolution
            variable_metadata: Mapping of variable names their metadata that will
                used in generating logged video captions.
        """
        if variable_metadata is None:
            self._variable_metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._variable_metadata = variable_metadata
        self._mean_data = _MeanVideoData(n_timesteps=n_timesteps)
        if enable_extended_videos:
            self._error_data: _ErrorVideoData | None = _ErrorVideoData(
                n_timesteps=n_timesteps
            )
            self._variance_data: _VarianceVideoData | None = _VarianceVideoData(
                n_timesteps=n_timesteps
            )
            self._enable_extended_videos = True
        else:
            self._error_data = None
            self._variance_data = None
            self._enable_extended_videos = False

    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping | None = None,
        gen_data_norm: TensorMapping | None = None,
        i_time_start: int = 0,
    ):
        del target_data_norm, gen_data_norm  # intentionally unused
        self._mean_data.record_batch(
            target_data=target_data,
            gen_data=gen_data,
            i_time_start=i_time_start,
        )
        if self._error_data is not None:
            self._error_data.record_batch(
                target_data=target_data,
                gen_data=gen_data,
                i_time_start=i_time_start,
            )
        if self._variance_data is not None:
            self._variance_data.record_batch(
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
        data = self._get_data()
        videos = {}
        for sub_label, d in data.items():
            videos[f"{label}/{sub_label}"] = d.make_video()
        return videos

    @torch.no_grad()
    def _get_data(self) -> Mapping[str, _MaybePairedVideoData]:
        """
        Returns video data as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        gen_data, target_data = self._mean_data.get()
        video_data = {}

        def get_units(name: str) -> str | None:
            if name in self._variable_metadata:
                return self._variable_metadata[name].units
            else:
                return None

        def get_long_name(name: str) -> str | None:
            if name in self._variable_metadata:
                return self._variable_metadata[name].long_name
            else:
                return None

        for name in gen_data:
            video_data[name] = _MaybePairedVideoData(
                caption=self._get_caption(name),
                gen=gen_data[name],
                target=target_data[name],
                units=get_units(name),
                long_name=f"ensemble mean of {get_long_name(name)}",
            )
            if self._enable_extended_videos:
                video_data[f"bias/{name}"] = _MaybePairedVideoData(
                    caption=(f"prediction - target for {name}"),
                    gen=gen_data[name] - target_data[name],
                    units=get_units(name),
                    long_name=f"bias of {get_long_name(name)}",
                )
        if self._error_data is not None:
            data = self._error_data.get()
            for name in data.rmse:
                video_data[f"rmse/{name}"] = _MaybePairedVideoData(
                    caption=f"RMSE over ensemble for {name}",
                    gen=data.rmse[name],
                    units=get_units(name),
                    long_name=f"root mean squared error of {get_long_name(name)}",
                )
            for name in data.min_err:
                video_data[f"min_err/{name}"] = _MaybePairedVideoData(
                    caption=f"Min across ensemble members of min error for {name}",
                    gen=data.min_err[name],
                    units=get_units(name),
                    long_name=(
                        f"min error of {get_long_name(name)} across ensemble members"
                    ),
                )
            for name in data.max_err:
                video_data[f"max_err/{name}"] = _MaybePairedVideoData(
                    caption=f"Max across ensemble members of max error for {name}",
                    gen=data.max_err[name],
                    units=get_units(name),
                    long_name=(
                        f"max error of {get_long_name(name)} across ensemble members"
                    ),
                )
        if self._variance_data is not None:
            gen_data, target_data = self._variance_data.get()
            for name in gen_data:
                video_data[f"gen_var/{name}"] = _MaybePairedVideoData(
                    caption=(
                        f"Variance of gen data for {name} "
                        "as fraction of target variance"
                    ),
                    gen=gen_data[name] / target_data[name],
                    units="",
                    long_name=(
                        f"prediction variance of {get_long_name(name)} "
                        "as fraction of target variance"
                    ),
                )
        return video_data

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        """
        Return video data as an xarray Dataset.
        """
        data = self._get_data()
        video_data = {}
        for label, d in data.items():
            label = label.strip("/").replace("/", "_")  # remove leading slash
            attrs = {}
            if d.units is not None:
                attrs["units"] = d.units
            if d.long_name is not None:
                attrs["long_name"] = d.long_name
            if d.target is not None:
                video_data[label] = xr.DataArray(
                    data=np.concatenate(
                        [d.gen.cpu().numpy()[None, :], d.target.cpu().numpy()[None, :]],
                        axis=0,
                    ),
                    dims=("source", "timestep", "lat", "lon"),
                    attrs=attrs,
                )
            else:
                video_data[label] = xr.DataArray(
                    data=d.gen.cpu().numpy(),
                    dims=("timestep", "lat", "lon"),
                    attrs=attrs,
                )
        return xr.Dataset(video_data)

    def _get_caption(self, name: str) -> str:
        caption = (
            "Autoregressive (left) prediction and (right) target for {name} [{units}]"
        )
        if name in self._variable_metadata:
            caption_name = self._variable_metadata[name].long_name
            units = self._variable_metadata[name].units
        else:
            caption_name, units = name, "unknown units"
        return caption.format(name=caption_name, units=units)


def _make_video(
    caption: str,
    gen: torch.Tensor,
    target: torch.Tensor | None = None,
):
    if target is None:
        video_data = np.expand_dims(gen.cpu().numpy(), axis=1)
    else:
        gen = np.expand_dims(gen.cpu().numpy(), axis=1)
        target = np.expand_dims(target.cpu().numpy(), axis=1)
        gap = np.zeros([gen.shape[0], 1, gen.shape[2], 10])
        video_data = np.concatenate([gen, gap, target], axis=-1)
    if target is None:
        data_min = np.nanmin(video_data)
        data_max = np.nanmax(video_data)
    else:
        # use target data to set the color scale
        data_min = np.nanmin(target)
        data_max = np.nanmax(target)
    # video data is brightness values on a 0-255 scale
    video_data = 255 * (video_data - data_min) / (data_max - data_min)
    video_data = np.minimum(video_data, 255)
    video_data = np.maximum(video_data, 0)
    video_data[np.isnan(video_data)] = 0
    caption += f"; vmin={data_min:.4g}, vmax={data_max:.4g}"
    return wandb.Video(
        np.flip(video_data, axis=-2),
        caption=caption,
        fps=4,
    )

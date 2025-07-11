from collections.abc import Mapping

import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.typing_ import TensorMapping
from fme.core.wandb import Image

from ..plotting import plot_paneled_data


class SnapshotAggregator:
    """
    An aggregator that records the first sample of the last batch of data.
    """

    _captions = {
        "full-field": (
            "{name} one step full field for first sample in last batch; "
            "(top) generated and (bottom) target [{units}]"
        ),
        "residual": (
            "{name} one step residual (prediction - previous time) for first sample in "
            "last batch; (top) generated and (bottom) target [{units}]"
        ),
        "error": (
            "{name} one step full field error (generated - target) "
            "for first sample in last batch [{units}]"
        ),
    }

    def __init__(
        self, dims: list[str], metadata: Mapping[str, VariableMetadata] | None = None
    ):
        """
        Args:
            dims: Dimensions of the data.
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
        """
        self._dims = dims
        if metadata is None:
            self._metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._metadata = metadata

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
    ):
        self._loss = loss
        self._target_data = target_data
        self._gen_data = gen_data
        self._target_data_norm = target_data_norm
        self._gen_data_norm = gen_data_norm

    def _get_data(self) -> tuple[TensorMapping, TensorMapping, TensorMapping]:
        time_dim = 1
        input_time = 0
        target_time = 1
        gen, target, input = {}, {}, {}
        for name in self._gen_data.keys():
            # use first sample in batch
            gen[name] = (
                self._gen_data[name]
                .select(dim=time_dim, index=target_time)[0]
                .cpu()
                .numpy()
            )
            target[name] = (
                self._target_data[name]
                .select(dim=time_dim, index=target_time)[0]
                .cpu()
                .numpy()
            )
            input[name] = (
                self._target_data[name]
                .select(dim=time_dim, index=input_time)[0]
                .cpu()
                .numpy()
            )
        return gen, target, input

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, Image]:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        image_logs = {}
        gen, target, input = self._get_data()
        for name in gen:
            images = {}
            images["error"] = [[(gen[name] - target[name])]]
            images["full-field"] = [[gen[name]], [target[name]]]
            images["residual"] = [
                [(gen[name] - input[name])],
                [(target[name] - input[name])],
            ]
            for key, data in images.items():
                if key == "error" or key == "residual":
                    diverging = True
                else:
                    diverging = False
                caption = self._get_caption(key, name)
                wandb_image = plot_paneled_data(data, diverging, caption=caption)
                image_logs[f"image-{key}/{name}"] = wandb_image
        image_logs = {f"{label}/{key}": image_logs[key] for key in image_logs}
        return image_logs

    def _get_caption(self, key: str, name: str) -> str:
        if name in self._metadata:
            caption_name = self._metadata[name].long_name
            units = self._metadata[name].units
        else:
            caption_name, units = name, "unknown_units"
        caption = self._captions[key].format(name=caption_name, units=units)
        return caption

    def get_dataset(self) -> xr.Dataset:
        gen, target, input = self._get_data()
        ds = xr.Dataset()
        for name in gen:
            if name in self._metadata:
                long_name = self._metadata[name].long_name
                units = self._metadata[name].units
            else:
                long_name = name
                units = "unknown_units"
            metadata_attrs = {"long_name": long_name, "units": units}
            ds[f"error_map-{name}"] = xr.DataArray(
                data=(gen[name] - target[name]), dims=self._dims, attrs=metadata_attrs
            )
            ds[f"gen_full_field_map-{name}"] = xr.DataArray(
                data=gen[name],
                dims=self._dims,
                attrs=metadata_attrs,
            )
            ds[f"gen_residual_map-{name}"] = xr.DataArray(
                data=gen[name] - input[name],
                dims=self._dims,
                attrs=metadata_attrs,
            )
            ds[f"target_residual_map-{name}"] = xr.DataArray(
                data=target[name] - input[name],
                dims=self._dims,
                attrs=metadata_attrs,
            )
        return ds

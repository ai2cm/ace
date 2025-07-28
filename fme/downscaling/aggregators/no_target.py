from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import torch
import xarray as xr

from fme.ace.aggregator.plotting import get_cmap_limits, plot_imshow
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import WandB
from fme.downscaling.aggregators.main import Mean, batch_mean, ensure_trailing_slash
from fme.downscaling.aggregators.shape_helpers import (
    get_data_dim,
    subselect_and_squeeze,
    upsample_tensor,
)

from ..metrics_and_maths import filter_tensor_mapping


class NoTargetAggregator:
    def __init__(
        self,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        ensemble_dim: int = 1,
    ):
        self.ensemble_dim = ensemble_dim
        self.single_sample_mean_map = _MapAggregator(
            name="single_sample_time_mean",
            variable_metadata=variable_metadata,
        )

    @torch.no_grad()
    def record_batch(
        self,
        prediction: TensorDict,
        coarse: TensorDict,
    ) -> None:
        self.single_sample_mean_map.record_batch(
            prediction=subselect_and_squeeze(prediction, self.ensemble_dim),
            coarse=subselect_and_squeeze(coarse, self.ensemble_dim),
        )

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        ret = {}
        ret.update(self.single_sample_mean_map.get_wandb(prefix))
        return ret

    def get_dataset(self) -> xr.Dataset:
        """
        Get the dataset from all sub aggregators.
        """
        ds = self.single_sample_mean_map.get_dataset()
        return ds


class _MapAggregator:
    def __init__(
        self,
        name: str = "",
        gap_width: int = 4,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._mean_prediction = Mean(batch_mean, name="prediction")
        self._mean_coarse = Mean(batch_mean, name="coarse")
        self.gap_width = gap_width
        self._name = ensure_trailing_slash(name)
        if variable_metadata is None:
            self._variable_metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._variable_metadata = variable_metadata
        self._expected_ndims = 3

    def _get_downscale_factor(self, prediction: TensorMapping, coarse: TensorMapping):
        k = list(prediction.keys())[0]
        return prediction[k].shape[-1] // coarse[k].shape[-1]

    @torch.no_grad()
    def record_batch(
        self,
        prediction: TensorMapping,
        coarse: TensorMapping,
    ) -> None:
        coarse = filter_tensor_mapping(coarse, prediction.keys())
        downscale_factor = self._get_downscale_factor(prediction, coarse)
        coarse = {k: upsample_tensor(v, downscale_factor) for k, v in coarse.items()}
        for data in [prediction, coarse]:
            ndim = get_data_dim(data)
            if ndim != self._expected_ndims:
                raise ValueError(
                    "Data passed to _MapAggregator must be 3D, i.e. any sample dim "
                    "is already folded into batch dim, or subselected and squeezed."
                )

        self._mean_prediction.record_batch(prediction)
        self._mean_coarse.record_batch(coarse)

    def _get_maps(self) -> Mapping[str, Any]:
        coarse = self._mean_coarse.get()
        prediction = self._mean_prediction.get()

        maps = {}
        for var_name in prediction.keys():
            gap = torch.full(
                (prediction[var_name].shape[-2], self.gap_width),
                float(prediction[var_name].min()),
                device=prediction[var_name].device,
            )
            maps[f"maps/{self._name}full-field/{var_name}"] = torch.cat(
                (prediction[var_name], gap, coarse[var_name]), dim=1
            )
        return maps

    def _get_caption(self, key: str, name: str, vmin: float, vmax: float) -> str:
        _caption = (
            "{name}  mean full field; (left) generated and " "(right) coarse [{units}]"
        )

        if name in self._variable_metadata:
            caption_name = self._variable_metadata[name].long_name
            units = self._variable_metadata[name].units
        else:
            caption_name, units = name, "unknown_units"
        caption = _caption.format(name=caption_name, units=units)
        caption += f" vmin={vmin:.4g}, vmax={vmax:.4g}."
        return caption

    def get_wandb(self, prefix: str = ""):
        prefix = ensure_trailing_slash(prefix)
        ret = {}
        wandb = WandB.get_instance()
        maps = self._get_maps()
        for key, data in maps.items():
            if "error" in key:
                diverging, cmap = True, "RdBu_r"
            else:
                diverging, cmap = False, None
            data = data.cpu().numpy()
            vmin, vmax = get_cmap_limits(data, diverging=diverging)
            map_name, var_name = key.split("/")[-2:]
            caption = self._get_caption(map_name, var_name, vmin, vmax)
            fig = plot_imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
            ret[f"{prefix}{key}"] = wandb.Image(fig, caption=caption)
            plt.close(fig)

        return ret

    def get_dataset(self) -> xr.Dataset:
        """
        Get the time mean maps dataset.
        """
        coarse = self._mean_coarse.get()
        prediction = self._mean_prediction.get()
        data = {}
        for key in prediction:
            data[f"{self._name}coarse.{key}"] = coarse[key].cpu().numpy()
            data[f"{self._name}prediction.{key}"] = prediction[key].cpu().numpy()
        ds = xr.Dataset({k: (("lat", "lon"), v) for k, v in data.items()})
        return ds

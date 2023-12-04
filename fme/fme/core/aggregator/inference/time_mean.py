import dataclasses
from typing import Dict, List, Mapping, MutableMapping, Optional, Union

import numpy as np
import torch
import xarray as xr

from fme.core import metrics
from fme.core.data_loading.typing import VariableMetadata
from fme.core.distributed import Distributed
from fme.core.wandb import WandB

wandb = WandB.get_instance()


@dataclasses.dataclass
class _TargetGenPair:
    name: str
    target: torch.Tensor
    gen: torch.Tensor

    def bias(self):
        return self.gen - self.target

    def rmse(self, weights: torch.Tensor) -> float:
        ret = float(
            metrics.root_mean_squared_error(
                predicted=self.gen,
                truth=self.target,
                weights=weights,
            )
            .cpu()
            .numpy()
        )
        return ret

    def weighted_mean_bias(self, weights: torch.Tensor) -> float:
        return float(
            metrics.weighted_mean_bias(
                predicted=self.gen, truth=self.target, weights=weights
            )
            .cpu()
            .numpy()
        )


def get_gen_shape(gen_data: Mapping[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class TimeMeanAggregator:
    """Statistics and images on the time-mean state.

    This aggregator keeps track of the time-mean state, then computes
    statistics and images on that time-mean state when logs are retrieved.
    """

    _image_captions = {
        "bias_map": "{name} time-mean bias (generated - target) [{units}]",
        "gen_map": "{name} time-mean generated [{units}]",
    }

    def __init__(
        self,
        area_weights: torch.Tensor,
        dist: Optional[Distributed] = None,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ):
        """
        Args:
            area_weights: Area weights for each grid cell.
            dist: Distributed object to use for communication.
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
        """
        self._area_weights = area_weights
        if dist is None:
            self._dist = Distributed.get_instance()
        else:
            self._dist = dist
        if metadata is None:
            self._metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._metadata = metadata

        # Dictionaries of tensors of shape [n_lat, n_lon] represnting time means
        self._target_data: Optional[Dict[str, torch.Tensor]] = None
        self._gen_data: Optional[Dict[str, torch.Tensor]] = None
        self._target_data_norm = None
        self._gen_data_norm = None
        self._n_batches = 0

    @staticmethod
    def _add_or_initialize_time_mean(
        maybe_dict: Optional[MutableMapping[str, torch.Tensor]],
        new_data: Mapping[str, torch.Tensor],
        ignore_initial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        sample_dim = 0
        time_dim = 1
        if ignore_initial:
            time_slice = slice(1, None)
        else:
            time_slice = slice(0, None)
        if maybe_dict is None:
            d: Dict[str, torch.Tensor] = {
                name: tensor[:, time_slice].mean(dim=time_dim).mean(dim=sample_dim)
                for name, tensor in new_data.items()
            }
        else:
            d = dict(maybe_dict)
            for name, tensor in new_data.items():
                d[name] += tensor[:, time_slice].mean(dim=time_dim).mean(dim=sample_dim)
        return d

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
        ignore_initial = i_time_start == 0
        self._target_data = self._add_or_initialize_time_mean(
            self._target_data, target_data, ignore_initial
        )
        self._gen_data = self._add_or_initialize_time_mean(
            self._gen_data, gen_data, ignore_initial
        )

        # we can ignore time slicing and just treat segments as though they're
        # different batches, because we can assume all time segments have the
        # same length
        self._n_batches += 1

    def _get_target_gen_pairs(self) -> List[_TargetGenPair]:
        if self._n_batches == 0 or self._gen_data is None or self._target_data is None:
            raise ValueError("No data recorded.")

        ret = []
        for name in self._gen_data.keys():
            gen = self._dist.reduce_mean(self._gen_data[name] / self._n_batches)
            target = self._dist.reduce_mean(self._target_data[name] / self._n_batches)
            ret.append(_TargetGenPair(gen=gen, target=target, name=name))
        return ret

    @torch.no_grad()
    def get_logs(self, label: str) -> Dict[str, Union[float, torch.Tensor]]:
        logs = {}
        preds = self._get_target_gen_pairs()
        bias_map_key, gen_map_key = "bias_map", "gen_map"
        for pred in preds:
            logs.update(
                {
                    f"{bias_map_key}/{pred.name}": _make_image(
                        self._get_caption(bias_map_key, pred.name, pred.bias()),
                        pred.bias(),
                    ),
                    f"{gen_map_key}/{pred.name}": _make_image(
                        self._get_caption(gen_map_key, pred.name, pred.gen), pred.gen
                    ),
                    f"rmse/{pred.name}": pred.rmse(weights=self._area_weights),
                    f"bias/{pred.name}": pred.weighted_mean_bias(
                        weights=self._area_weights
                    ),
                }
            )

        if len(label) != 0:
            return {f"{label}/{key}": logs[key] for key in logs}
        return logs

    def _get_caption(self, key: str, name: str, data: torch.Tensor) -> str:
        if name in self._metadata:
            caption_name = self._metadata[name].long_name
            units = self._metadata[name].units
        else:
            caption_name, units = name, "unknown_units"
        caption = self._image_captions[key].format(name=caption_name, units=units)
        caption += f" vmin={data.min():.4g}, vmax={data.max():.4g}."
        return caption

    def get_dataset(self) -> xr.Dataset:
        data = {}
        preds = self._get_target_gen_pairs()
        dims = ("lat", "lon")
        for pred in preds:
            bias_metadata = self._metadata.get(
                pred.name, VariableMetadata(units="unknown_units", long_name=pred.name)
            )._asdict()
            gen_metadata = VariableMetadata(units="", long_name=pred.name)._asdict()
            data.update(
                {
                    f"bias_map-{pred.name}": xr.DataArray(
                        pred.bias().cpu(), dims=dims, attrs=bias_metadata
                    ),
                    f"gen_map-{pred.name}": xr.DataArray(
                        pred.gen.cpu(),
                        dims=dims,
                        attrs=gen_metadata,
                    ),
                }
            )
        return xr.Dataset(data)


def _make_image(
    caption: str,
    data: torch.Tensor,
):
    image_data = data.cpu().numpy()
    lat_dim = -2
    return wandb.Image(
        np.flip(image_data, axis=lat_dim),
        caption=caption,
    )

from typing import Dict, Mapping, Optional

import matplotlib.pyplot as plt
import torch

from fme.core.data_loading.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.wandb import Image, WandB

from ..plotting import get_cmap_limits, plot_imshow


class MapAggregator:
    """
    An aggregator that records the average over batches as function of lat and lon.
    """

    _captions = {
        "full-field": (
            "{name} one step mean full field; "
            "(left) generated and (right) target [{units}]"
        ),
        "error": (
            "{name} one step mean full field error (generated - target) " "[{units}]"
        ),
    }

    def __init__(self, metadata: Optional[Mapping[str, VariableMetadata]] = None):
        """
        Args:
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
        """
        if metadata is None:
            self._metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._metadata = metadata
        self._n_batches = 0
        self._target_data: Dict[str, torch.Tensor] = {}
        self._gen_data: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
    ):
        self._loss = loss
        for name in target_data:
            if name in self._target_data:
                self._target_data[name] += target_data[name].mean(dim=0)
            else:
                self._target_data[name] = target_data[name].mean(dim=0)
        for name in gen_data:
            if name in self._gen_data:
                self._gen_data[name] += gen_data[name].mean(dim=0)
            else:
                self._gen_data[name] = gen_data[name].mean(dim=0)
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str) -> Dict[str, Image]:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        dist = Distributed.get_instance()
        time_dim = 0
        target_time = 1
        image_logs = {}
        wandb = WandB.get_instance()
        for name in self._gen_data.keys():
            # use first sample in batch
            gen = (
                dist.reduce_mean(
                    self._gen_data[name].select(dim=time_dim, index=target_time)
                )
                / self._n_batches
            )
            target = (
                dist.reduce_mean(
                    self._target_data[name].select(dim=time_dim, index=target_time)
                )
                / self._n_batches
            )
            gap_shape = (target.shape[-2], 4)
            gap = torch.full(gap_shape, target.min()).to(get_device())
            images = {}
            images["error"] = (gen - target).cpu().numpy()
            images["full-field"] = torch.cat((gen, gap, target), axis=1).cpu().numpy()
            for key, data in images.items():
                if key == "error":
                    diverging = True
                    cmap = "RdBu_r"
                else:
                    diverging = False
                    cmap = None
                vmin, vmax = get_cmap_limits(data, diverging=diverging)
                caption = self._get_caption(key, name, vmin, vmax)
                fig = plot_imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
                wandb_image = wandb.Image(fig, caption=caption)
                plt.close(fig)
                image_logs[f"image-{key}/{name}"] = wandb_image
        image_logs = {f"{label}/{key}": image_logs[key] for key in image_logs}
        return image_logs

    def _get_caption(self, key: str, name: str, vmin: float, vmax: float) -> str:
        if name in self._metadata:
            caption_name = self._metadata[name].long_name
            units = self._metadata[name].units
        else:
            caption_name, units = name, "unknown_units"
        caption = self._captions[key].format(name=caption_name, units=units)
        caption += f" vmin={vmin:.4g}, vmax={vmax:.4g}."
        return caption

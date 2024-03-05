from typing import Dict, Mapping, Optional

import matplotlib.pyplot as plt
import torch

from fme.core.data_loading.data_typing import VariableMetadata
from fme.core.wandb import Image, WandB

from ..plotting import get_cmap_limits, plot_imshow


class SnapshotAggregator:
    """
    An aggregator that records the first sample of the last batch of data.
    """

    _captions = {
        "full-field": (
            "{name} one step full field for last sample; "
            "(left) generated and (right) target [{units}]"
        ),
        "residual": (
            "{name} one step residual (prediction - previous time) for last sample; "
            "(left) generated and (right) target [{units}]"
        ),
        "error": (
            "{name} one step full field error (generated - target) "
            "for last sample [{units}]"
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
        self._target_data = target_data
        self._gen_data = gen_data
        self._target_data_norm = target_data_norm
        self._gen_data_norm = gen_data_norm

    @torch.no_grad()
    def get_logs(self, label: str) -> Dict[str, Image]:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        time_dim = 1
        input_time = 0
        target_time = 1
        image_logs = {}
        wandb = WandB.get_instance()
        for name in self._gen_data.keys():
            # use first sample in batch
            gen = self._gen_data[name].select(dim=time_dim, index=target_time)[0].cpu()
            target = (
                self._target_data[name].select(dim=time_dim, index=target_time)[0].cpu()
            )
            input = (
                self._target_data[name].select(dim=time_dim, index=input_time)[0].cpu()
            )
            gap_shape = (input.shape[-2], 4)
            gap = torch.full(gap_shape, target.min())
            gap_res = torch.full(gap_shape, (target - input).min())
            images = {}
            images["error"] = (gen - target).numpy()
            images["full-field"] = torch.cat((gen, gap, target), axis=1).numpy()
            images["residual"] = torch.cat(
                (
                    gen - input,
                    gap_res,
                    target - input,
                ),
                axis=1,
            ).numpy()
            for key, data in images.items():
                if key == "error" or key == "residual":
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

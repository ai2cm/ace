from typing import Mapping

import torch
from fme.core.device import get_device
from fme.core.wandb import WandB

wandb = WandB.get_instance()


class SnapshotAggregator:
    """
    An aggregator that records the first sample of the last batch of data.
    """

    _captions = {
        "full-field": "{name} one step full field for last sample; (left) generated and (right) target.",  # noqa: E501
        "residual": "{name} one step residual for last sample; (left) generated and (right) target.",  # noqa: E501
        "error": "{name} one step error (generated - target) for last sample.",
    }

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
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        time_dim = 1
        input_time = 0
        target_time = 1
        image_logs = {}
        for name in self._gen_data.keys():
            # use first sample in batch
            gen = self._gen_data[name].select(dim=time_dim, index=target_time)[0]
            target = self._target_data[name].select(dim=time_dim, index=target_time)[0]
            input = self._target_data[name].select(dim=time_dim, index=input_time)[0]
            gap_shape = (input.shape[-2], 4)
            gap = torch.full(gap_shape, target.min()).to(get_device())
            gap_res = torch.full(gap_shape, (target - input).min()).to(get_device())
            images = {}
            images["error"] = gen - target
            images["full-field"] = torch.cat((gen, gap, target), axis=1)
            images["residual"] = torch.cat(
                (
                    gen - input,
                    gap_res,
                    target - input,
                ),
                axis=1,
            )
            for key, data in images.items():
                caption = self._captions[key].format(name=name)
                caption += f" vmin={data.min():.4g}, vmax={data.max():.4g}."
                wandb_image = wandb.Image(data, caption=caption)
                image_logs[f"image-{key}/{name}"] = wandb_image
        image_logs = {f"{label}/{key}": image_logs[key] for key in image_logs}
        return image_logs

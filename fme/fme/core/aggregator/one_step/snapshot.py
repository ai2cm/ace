from typing import Mapping

import torch
from fme.core.device import get_device
from fme.core.wandb import WandB

wandb = WandB.get_instance()


class SnapshotAggregator:
    """
    An aggregator that records the first sample of the last batch of data.
    """

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
            gen_for_image = self._gen_data_norm[name].select(
                dim=time_dim, index=target_time
            )[
                0
            ]  # first sample in batch
            target_for_image = self._target_data_norm[name].select(
                dim=time_dim, index=target_time
            )[0]
            input_for_image = self._target_data_norm[name].select(
                dim=time_dim, index=input_time
            )[0]
            gap = torch.zeros((input_for_image.shape[-2], 4)).to(
                get_device(), dtype=torch.float
            )
            image_error = gen_for_image - target_for_image
            image_full_field = torch.cat((gen_for_image, gap, target_for_image), axis=1)
            image_residual = torch.cat(
                (
                    gen_for_image - input_for_image,
                    gap,
                    target_for_image - input_for_image,
                ),
                axis=1,
            )
            caption = (
                f"{name} one step full field for "
                "last sample; (left) generated and (right) target."
            )
            wandb_image = wandb.Image(image_full_field, caption=caption)
            image_logs[f"image-full-field/{name}"] = wandb_image
            caption = (
                f"{name} one step residual for "
                "last sample; (left) generated and (right) target."
            )
            wandb_image = wandb.Image(image_residual, caption=caption)
            image_logs[f"image-residual/{name}"] = wandb_image
            caption = f"{name} one step error " "(generated - target) for last sample."
            wandb_image = wandb.Image(image_error, caption=caption)
            image_logs[f"image-error/{name}"] = wandb_image
        image_logs = {f"{label}/{key}": image_logs[key] for key in image_logs}
        return image_logs

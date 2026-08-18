import contextlib
import dataclasses
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.labels import BatchLabels
from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer
from fme.core.registry import ModuleSelector
from fme.core.typing_ import TensorDict, TensorMapping


class StepDiscriminator:
    """
    Judges whether an (input, output) timestep pair comes from the training
    data or was produced by the model.

    Operates at the same API level as StepABC: callers pass denormalized
    variable dicts and receive per-pixel logits; normalization, packing, and
    module concerns stay internal. Positive logits mean "judged real".
    """

    CHANNEL_DIM = -3

    def __init__(
        self,
        builder: ModuleSelector,
        in_names: list[str],
        out_names: list[str],
        normalizer: StandardNormalizer,
        dataset_info: DatasetInfo,
    ):
        """
        Args:
            builder: Builder for the discriminator module.
            in_names: Names of the input-timestep variables the discriminator
                is conditioned on.
            out_names: Names of the output-timestep variables it judges.
            normalizer: Normalizer applied to both timesteps before packing.
            dataset_info: Information about the dataset.
        """
        self._in_names = list(in_names)
        self._out_names = list(out_names)
        self._in_packer = Packer(self._in_names)
        self._out_packer = Packer(self._out_names)
        self._normalizer = normalizer
        module = builder.build(
            n_in_channels=len(self._in_names) + len(self._out_names),
            n_out_channels=1,
            dataset_info=dataset_info,
        )
        dist = Distributed.get_instance()
        self._module = module.to(get_device()).wrap_module(dist.wrap_module)

    @property
    def in_names(self) -> list[str]:
        """Names of the input-timestep (conditioning) variables."""
        return list(self._in_names)

    @property
    def out_names(self) -> list[str]:
        """Names of the output-timestep variables being judged."""
        return list(self._out_names)

    @property
    def modules(self) -> nn.ModuleList:
        return nn.ModuleList([self._module.torch_module])

    @property
    def training(self) -> bool:
        return self._module.torch_module.training

    def train(self, mode: bool = True) -> "StepDiscriminator":
        self._module.torch_module.train(mode)
        return self

    @contextlib.contextmanager
    def gradient_sync_disabled(self):
        """Context under which forward passes do not sync gradients across
        ranks on backward.

        Used for forward passes whose gradients with respect to discriminator
        parameters will be discarded (the generator's adversarial term), so
        the single synchronized backward per batch is the discriminator's own
        loss. ``no_sync`` only exists on DistributedDataParallel, hence the
        isinstance check; other wrappers have nothing to disable.
        """
        torch_module = self._module.torch_module
        if isinstance(torch_module, DistributedDataParallel):
            with torch_module.no_sync():
                yield
        else:
            yield

    def forward(
        self,
        input: TensorMapping,
        output: TensorMapping,
        labels: BatchLabels | None = None,
    ) -> torch.Tensor:
        """
        Judge (input, output) timestep pairs.

        Args:
            input: Denormalized input-timestep data, containing at least
                ``in_names``. Tensors of shape [n_batch, n_lat, n_lon].
            output: Denormalized output-timestep data, containing at least
                ``out_names``. Tensors of shape [n_batch, n_lat, n_lon].
            labels: Labels for each batch member.

        Returns:
            Per-pixel logits of shape [n_batch, 1, n_lat, n_lon]; positive
            means "judged real".
        """
        input_norm = self._normalizer.normalize(
            {name: input[name] for name in self._in_names}
        )
        output_norm = self._normalizer.normalize(
            {name: output[name] for name in self._out_names}
        )
        input_tensor = self._in_packer.pack(input_norm, axis=self.CHANNEL_DIM)
        output_tensor = self._out_packer.pack(output_norm, axis=self.CHANNEL_DIM)
        pair_tensor = torch.cat([input_tensor, output_tensor], dim=self.CHANNEL_DIM)
        return self._module(pair_tensor, labels=labels)

    def get_state(self) -> dict[str, Any]:
        return {"module": self._module.get_state()}

    def load_state(self, state: dict[str, Any]) -> None:
        self._module.load_state(state["module"])


@dataclasses.dataclass
class GanStepLosses:
    """Adversarial losses and diagnostics for one forward step.

    Parameters:
        generator_loss: Non-saturating generator term (how strongly the
            discriminator rejects the generated pair), with gradients flowing
            into the generator. Unweighted.
        discriminator_loss: The discriminator's training loss (real + fake
            sides), with gradients flowing only into the discriminator.
        discriminator_loss_real: Detached real-side component.
        discriminator_loss_fake: Detached fake-side component.
        score_real: Detached mean sigmoid score on real pairs (1 = confidently
            real; 0.5 at equilibrium).
        score_fake: Detached mean sigmoid score on generated pairs.
    """

    generator_loss: torch.Tensor
    discriminator_loss: torch.Tensor
    discriminator_loss_real: torch.Tensor
    discriminator_loss_fake: torch.Tensor
    score_real: torch.Tensor
    score_fake: torch.Tensor


def _area_weighted_bce(
    logits: torch.Tensor,
    target_value: float,
    gridded_operations: GriddedOperations,
) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.full_like(logits, target_value), reduction="none"
    )
    return gridded_operations.area_weighted_mean(bce).mean()


def _area_weighted_score(
    logits: torch.Tensor, gridded_operations: GriddedOperations
) -> torch.Tensor:
    return gridded_operations.area_weighted_mean(torch.sigmoid(logits)).mean().detach()


def _detached(data: TensorMapping) -> TensorDict:
    return {k: v.detach() for k, v in data.items()}


def compute_gan_step_losses(
    discriminator: StepDiscriminator,
    gridded_operations: GriddedOperations,
    real_input: TensorMapping,
    real_output: TensorMapping,
    fake_input: TensorMapping,
    fake_output: TensorMapping,
    real_labels: BatchLabels | None = None,
    fake_labels: BatchLabels | None = None,
) -> GanStepLosses:
    """
    Compute non-saturating GAN losses for one forward step.

    The generated pair is judged twice: once with gradients flowing into the
    generator (the generator's adversarial term, computed with gradient sync
    disabled since its discriminator-parameter gradients are discarded), and
    once fully detached for the discriminator's own loss. Real and fake sides
    of the discriminator loss are averaged separately, so differing batch
    sizes (e.g. ensemble members on the fake side only) stay balanced.

    Args:
        discriminator: The discriminator to evaluate and train.
        gridded_operations: Provides the area-weighted spherical mean reducing
            per-pixel binary cross-entropy to a scalar.
        real_input: Denormalized input-timestep data from the dataset.
        real_output: Denormalized output-timestep data from the dataset.
        fake_input: Denormalized input the model consumed this step (already
            detached from previous steps when the host detaches between
            steps).
        fake_output: Denormalized model output for this step.
        real_labels: Labels for the real pairs' batch members.
        fake_labels: Labels for the generated pairs' batch members.

    Returns:
        The step's adversarial losses and detached diagnostics.
    """
    with discriminator.gradient_sync_disabled():
        generator_logits = discriminator.forward(
            fake_input, fake_output, labels=fake_labels
        )
    generator_loss = _area_weighted_bce(generator_logits, 1.0, gridded_operations)
    fake_logits = discriminator.forward(
        _detached(fake_input), _detached(fake_output), labels=fake_labels
    )
    real_logits = discriminator.forward(real_input, real_output, labels=real_labels)
    loss_real = _area_weighted_bce(real_logits, 1.0, gridded_operations)
    loss_fake = _area_weighted_bce(fake_logits, 0.0, gridded_operations)
    return GanStepLosses(
        generator_loss=generator_loss,
        discriminator_loss=loss_real + loss_fake,
        discriminator_loss_real=loss_real.detach(),
        discriminator_loss_fake=loss_fake.detach(),
        score_real=_area_weighted_score(real_logits, gridded_operations),
        score_fake=_area_weighted_score(fake_logits, gridded_operations),
    )

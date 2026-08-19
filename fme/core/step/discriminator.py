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
        """Context under which forward passes do not arm gradient syncing.

        Used for forward passes whose gradients with respect to discriminator
        parameters will be discarded (the generator's adversarial term).
        ``no_sync`` marks the forwards it wraps, not a graph: a forward under
        it does not call the DDP reducer's ``prepare_for_backward``, so a
        later backward through that graph fires no reduction (grads still
        accumulate locally and must be discarded by the caller). The
        discriminator's own loss must therefore run its forwards *outside*
        this context and immediately precede its backward, so that backward
        is the one the armed reducer services. ``no_sync`` only exists on
        DistributedDataParallel, hence the isinstance check; other wrappers
        have nothing to disable.
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


def compute_generator_adversarial_loss(
    discriminator: StepDiscriminator,
    gridded_operations: GriddedOperations,
    fake_input: TensorMapping,
    fake_output: TensorMapping,
    fake_labels: BatchLabels | None = None,
) -> torch.Tensor:
    """
    Compute the non-saturating generator adversarial term for one step.

    Gradients flow through the discriminator into the generator; the
    discriminator's own parameter gradients from this term are discarded by
    the caller, so the forward runs under ``gradient_sync_disabled`` (no DDP
    reduction is armed for the generator's backward).

    Args:
        discriminator: The discriminator judging the pair.
        gridded_operations: Provides the area-weighted spherical mean reducing
            per-pixel binary cross-entropy to a scalar.
        fake_input: Denormalized input the model consumed this step (already
            detached from previous steps when the host detaches between
            steps).
        fake_output: Denormalized model output for this step.
        fake_labels: Labels for the generated pairs' batch members.

    Returns:
        Unweighted generator adversarial loss (how strongly the discriminator
        rejects the generated pair).
    """
    with discriminator.gradient_sync_disabled():
        generator_logits = discriminator.forward(
            fake_input, fake_output, labels=fake_labels
        )
    return _area_weighted_bce(generator_logits, 1.0, gridded_operations)


@dataclasses.dataclass
class GanStepPair:
    """Real and generated (input, output) pairs for one optimized step,
    cached so the discriminator's own loss can run after the generator's
    optimizer step (its forwards must immediately precede its backward for
    DDP gradient reduction to service it; see
    ``StepDiscriminator.gradient_sync_disabled``).

    The fake side is detached so the discriminator loss cannot backpropagate
    into the generator.
    """

    real_input: TensorDict
    real_output: TensorDict
    fake_input: TensorDict
    fake_output: TensorDict
    real_labels: BatchLabels | None
    fake_labels: BatchLabels | None

    @classmethod
    def from_step_data(
        cls,
        real_input: TensorMapping,
        real_output: TensorMapping,
        fake_input: TensorMapping,
        fake_output: TensorMapping,
        real_labels: BatchLabels | None,
        fake_labels: BatchLabels | None,
    ) -> "GanStepPair":
        return cls(
            real_input=dict(real_input),
            real_output=dict(real_output),
            fake_input=_detached(fake_input),
            fake_output=_detached(fake_output),
            real_labels=real_labels,
            fake_labels=fake_labels,
        )


@dataclasses.dataclass
class DiscriminatorLosses:
    """The discriminator's training loss and detached diagnostics, aggregated
    over the batch's optimized steps.

    Parameters:
        loss: The discriminator's training loss (real + fake sides + R1 if
            enabled, summed over steps), with gradients flowing only into the
            discriminator.
        loss_real: Detached real-side component, averaged over steps.
        loss_fake: Detached fake-side component, averaged over steps.
        r1_penalty: Detached R1 gradient penalty, averaged over steps
            (0 when R1 is not enabled).
        score_real: Detached mean sigmoid score on real pairs, averaged over
            steps (1 = confidently real; 0.5 at equilibrium).
        score_fake: Detached mean sigmoid score on generated pairs, averaged
            over steps.
    """

    loss: torch.Tensor
    loss_real: torch.Tensor
    loss_fake: torch.Tensor
    r1_penalty: torch.Tensor
    score_real: torch.Tensor
    score_fake: torch.Tensor


def _r1_gradient_penalty(
    real_logits: torch.Tensor,
    real_inputs: list[torch.Tensor],
    gridded_operations: GriddedOperations,
) -> torch.Tensor:
    """R1 gradient penalty (Mescheder et al. 2018) on real data.

    Returns the area-weighted mean of ``||∇_x D(x)||²`` over the real
    inputs — the unweighted penalty; the caller multiplies by ``λ/2``.
    ``create_graph=True`` so the penalty's own gradients flow into the
    discriminator's optimizer step.
    """
    (grads,) = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_inputs,
        create_graph=True,
    )
    grad_sq = grads.square()
    return gridded_operations.area_weighted_mean(grad_sq).mean()


def compute_discriminator_losses(
    discriminator: StepDiscriminator,
    gridded_operations: GriddedOperations,
    pairs: list[GanStepPair],
    r1_penalty_coefficient: float = 0.0,
) -> DiscriminatorLosses:
    """
    Compute the discriminator's training loss over a batch's optimized steps.

    Real and fake sides are averaged separately per step, so differing batch
    sizes (e.g. ensemble members on the fake side only) stay balanced; steps
    are summed. All forwards run before the caller's single backward — under
    DDP they arm the reducer that backward services, so this must be the
    discriminator's final forward-backward of the batch (see
    ``StepDiscriminator.gradient_sync_disabled``).

    Args:
        discriminator: The discriminator to train.
        gridded_operations: Provides the area-weighted spherical mean reducing
            per-pixel binary cross-entropy to a scalar.
        pairs: One real/fake pair per optimized step.
        r1_penalty_coefficient: R1 gradient penalty coefficient (λ/2 in
            Mescheder et al. 2018). 0 disables the penalty. Applied to
            the real-side forward, per step.

    Returns:
        The discriminator's loss and detached diagnostics.
    """
    if not pairs:
        raise ValueError("pairs must be non-empty")
    losses_real = []
    losses_fake = []
    r1_penalties = []
    scores_real = []
    scores_fake = []
    use_r1 = r1_penalty_coefficient > 0
    for pair in pairs:
        fake_logits = discriminator.forward(
            pair.fake_input, pair.fake_output, labels=pair.fake_labels
        )
        if use_r1:
            real_packed = torch.cat(
                [
                    torch.stack(list(pair.real_input.values()), dim=-3),
                    torch.stack(list(pair.real_output.values()), dim=-3),
                ],
                dim=-3,
            )
            real_packed.requires_grad_(True)
            n_in = len(pair.real_input)
            real_input_r1 = dict(
                zip(pair.real_input.keys(), real_packed[:, :n_in].unbind(dim=1))
            )
            real_output_r1 = dict(
                zip(pair.real_output.keys(), real_packed[:, n_in:].unbind(dim=1))
            )
            real_logits = discriminator.forward(
                real_input_r1, real_output_r1, labels=pair.real_labels
            )
            r1 = _r1_gradient_penalty(real_logits, [real_packed], gridded_operations)
            r1_penalties.append(r1)
        else:
            real_logits = discriminator.forward(
                pair.real_input, pair.real_output, labels=pair.real_labels
            )
        losses_real.append(_area_weighted_bce(real_logits, 1.0, gridded_operations))
        losses_fake.append(_area_weighted_bce(fake_logits, 0.0, gridded_operations))
        scores_real.append(_area_weighted_score(real_logits, gridded_operations))
        scores_fake.append(_area_weighted_score(fake_logits, gridded_operations))
    loss_real = torch.stack(losses_real).sum()
    loss_fake = torch.stack(losses_fake).sum()
    if use_r1:
        r1_total = torch.stack(r1_penalties).sum()
        total_loss = loss_real + loss_fake + r1_penalty_coefficient * r1_total
    else:
        r1_total = torch.tensor(0.0, device=loss_real.device)
        total_loss = loss_real + loss_fake
    n_steps = len(pairs)
    return DiscriminatorLosses(
        loss=total_loss,
        loss_real=loss_real.detach() / n_steps,
        loss_fake=loss_fake.detach() / n_steps,
        r1_penalty=r1_total.detach() / n_steps,
        score_real=torch.stack(scores_real).mean(),
        score_fake=torch.stack(scores_fake).mean(),
    )

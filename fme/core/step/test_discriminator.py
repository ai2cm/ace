import math

import pytest
import torch

from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.normalizer import StandardNormalizer
from fme.core.registry import ModuleSelector
from fme.core.step.discriminator import StepDiscriminator, compute_gan_step_losses
from fme.core.testing.dataset_info import get_dataset_info

IN_NAMES = ["a", "b"]
OUT_NAMES = ["b", "c"]
IMG_SHAPE = (5, 5)


class _ChannelMeanLogits(torch.nn.Module):
    """Per-pixel logit head with a parameter so optimizers can build."""

    def __init__(self, weight: float = 0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((1,), weight))

    def forward(self, x):
        return x.mean(dim=-3, keepdim=True) * self.weight


def _get_normalizer(names: list[str], mean: float = 0.0) -> StandardNormalizer:
    return StandardNormalizer(
        means={name: torch.tensor(mean) for name in names},
        stds={name: torch.tensor(1.0) for name in names},
    )


def _get_discriminator(
    module: torch.nn.Module | None = None, mean: float = 0.0
) -> StepDiscriminator:
    if module is None:
        module = _ChannelMeanLogits()
    return StepDiscriminator(
        builder=ModuleSelector(type="prebuilt", config={"module": module}),
        in_names=IN_NAMES,
        out_names=OUT_NAMES,
        normalizer=_get_normalizer(sorted(set(IN_NAMES + OUT_NAMES)), mean=mean),
        dataset_info=get_dataset_info(img_shape=IMG_SHAPE),
    )


def _get_pair(n_batch: int = 3) -> tuple[dict, dict]:
    device = get_device()
    input = {name: torch.randn(n_batch, *IMG_SHAPE, device=device) for name in IN_NAMES}
    output = {
        name: torch.randn(n_batch, *IMG_SHAPE, device=device) for name in OUT_NAMES
    }
    return input, output


def test_forward_returns_per_pixel_logits():
    discriminator = _get_discriminator()
    input, output = _get_pair(n_batch=3)
    logits = discriminator.forward(input, output)
    assert logits.shape == (3, 1, *IMG_SHAPE)


def test_forward_ignores_extra_variables():
    """Callers may pass supersets (e.g. the model's full output dict)."""
    discriminator = _get_discriminator()
    input, output = _get_pair()
    extra = torch.randn_like(input["a"])
    logits = discriminator.forward(
        {**input, "extra": extra}, {**output, "extra": extra}
    )
    torch.testing.assert_close(logits, discriminator.forward(input, output))


def test_forward_normalizes_inputs():
    """A pair at the climatological mean packs to all-zero channels."""
    discriminator = _get_discriminator(module=_ChannelMeanLogits(weight=1.0), mean=2.0)
    device = get_device()
    input = {name: torch.full((2, *IMG_SHAPE), 2.0, device=device) for name in IN_NAMES}
    output = {
        name: torch.full((2, *IMG_SHAPE), 2.0, device=device) for name in OUT_NAMES
    }
    logits = discriminator.forward(input, output)
    torch.testing.assert_close(logits, torch.zeros_like(logits))


def test_state_round_trip():
    discriminator = _get_discriminator(module=_ChannelMeanLogits(weight=0.7))
    other = _get_discriminator(module=_ChannelMeanLogits(weight=0.1))
    other.load_state(discriminator.get_state())
    input, output = _get_pair()
    torch.testing.assert_close(
        other.forward(input, output), discriminator.forward(input, output)
    )


def _get_gridded_operations() -> LatLonOperations:
    return LatLonOperations(torch.ones(*IMG_SHAPE))


def test_gan_step_losses_at_chance_logits():
    """A zero-logit discriminator is at equilibrium: both sides score 0.5 and
    every BCE term is ln(2)."""
    discriminator = _get_discriminator(module=_ChannelMeanLogits(weight=0.0))
    real_input, real_output = _get_pair()
    fake_input, fake_output = _get_pair()
    losses = compute_gan_step_losses(
        discriminator=discriminator,
        gridded_operations=_get_gridded_operations(),
        real_input=real_input,
        real_output=real_output,
        fake_input=fake_input,
        fake_output=fake_output,
    )
    ln2 = torch.tensor(math.log(2.0), device=losses.generator_loss.device)
    torch.testing.assert_close(losses.generator_loss, ln2)
    torch.testing.assert_close(losses.discriminator_loss_real, ln2)
    torch.testing.assert_close(losses.discriminator_loss_fake, ln2)
    torch.testing.assert_close(losses.discriminator_loss, 2 * ln2)
    torch.testing.assert_close(
        losses.score_real, torch.full_like(losses.score_real, 0.5)
    )
    torch.testing.assert_close(
        losses.score_fake, torch.full_like(losses.score_fake, 0.5)
    )


def test_generator_loss_flows_into_fake_pair_only():
    discriminator = _get_discriminator(module=_ChannelMeanLogits(weight=0.5))
    real_input, real_output = _get_pair()
    fake_input, fake_output = _get_pair()
    for tensor in list(fake_input.values()) + list(fake_output.values()):
        tensor.requires_grad_(True)
    losses = compute_gan_step_losses(
        discriminator=discriminator,
        gridded_operations=_get_gridded_operations(),
        real_input=real_input,
        real_output=real_output,
        fake_input=fake_input,
        fake_output=fake_output,
    )
    losses.generator_loss.backward()
    assert fake_output["b"].grad is not None
    assert fake_input["a"].grad is not None


def test_discriminator_loss_does_not_flow_into_generator():
    discriminator = _get_discriminator(module=_ChannelMeanLogits(weight=0.5))
    real_input, real_output = _get_pair()
    fake_input, fake_output = _get_pair()
    for tensor in list(fake_input.values()) + list(fake_output.values()):
        tensor.requires_grad_(True)
    losses = compute_gan_step_losses(
        discriminator=discriminator,
        gridded_operations=_get_gridded_operations(),
        real_input=real_input,
        real_output=real_output,
        fake_input=fake_input,
        fake_output=fake_output,
    )
    losses.discriminator_loss.backward()
    assert all(tensor.grad is None for tensor in fake_input.values())
    assert all(tensor.grad is None for tensor in fake_output.values())
    (weight,) = discriminator.modules[0].parameters()
    assert weight.grad is not None


def test_detached_diagnostics_carry_no_graph():
    discriminator = _get_discriminator(module=_ChannelMeanLogits(weight=0.5))
    real_input, real_output = _get_pair()
    fake_input, fake_output = _get_pair()
    losses = compute_gan_step_losses(
        discriminator=discriminator,
        gridded_operations=_get_gridded_operations(),
        real_input=real_input,
        real_output=real_output,
        fake_input=fake_input,
        fake_output=fake_output,
    )
    for diagnostic in [
        losses.discriminator_loss_real,
        losses.discriminator_loss_fake,
        losses.score_real,
        losses.score_fake,
    ]:
        assert diagnostic.grad_fn is None


def test_train_mode_toggles():
    discriminator = _get_discriminator()
    assert discriminator.training
    discriminator.train(False)
    assert not discriminator.training
    discriminator.train()
    assert discriminator.training


def test_missing_variable_raises():
    discriminator = _get_discriminator()
    input, output = _get_pair()
    del input["a"]
    with pytest.raises(KeyError):
        discriminator.forward(input, output)

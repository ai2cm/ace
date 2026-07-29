"""Cut-point decomposition of the noise-conditioned SFNO.

The v2 noise-conditioned SFNO
(:class:`fme.ace.registry.stochastic_sfno.NoiseConditionedModel` wrapping a
:class:`SphericalFourierNeuralOperatorNet`) is a monolithic module: physical
input channels in, physical output channels out. Two pieces of planned work
need to open it up and use its three stages as separate pool components:

- the **latent-splice** transfer arm, which keeps a donor's *processor* frozen
  and trains new learned transforms on either side of it, replacing the
  donor's physical-variable interface; and
- the **multi-scale warm-start**, which initializes all three components of a
  multi-resolution composite from a plain ACE checkpoint.

This module registers one :class:`TransformSelector` entry, ``sfno_cut_point``,
whose ``part`` field selects which stage to build. All three parts are
configured with the same ``sfno:`` block (the donor's
:class:`NoiseConditionedSFNOBuilder` config), and each carries the *donor's*
``state_dict`` names for its own parameters, so any subset of the three can be
name-matched onto a donor ACE checkpoint via ``donor_checkpoint``.

Cut-point interface
-------------------

Both cut-points carry a single tensor, because that is what a pool component
consumes and produces. Its channels are the ``embed_dim`` latent followed by
the big-skip residual (``in_chans`` channels, absent when ``big_skip`` is
False), so a latent domain sitting at a cut-point declares
``embed_dim + in_chans`` channels.

Stage boundaries follow the monolithic ``forward`` with one deliberate
regrouping: the big-skip normalization (``norm_big_skip``, a
context-conditioned layer norm) moves from where the monolith computes it
(alongside the residual, before the encoder) into the **processor**. That
keeps every context-conditioned operation — all the blocks plus the skip
normalization — inside the one component that draws the noise, so composing
the three parts reproduces the monolithic net's output *exactly* rather than
approximately, for any configuration. ``test_cutpoint.py`` asserts that
equivalence; it is what pins this module against changes to the net it mirrors.

Bit-exactness also requires the encoder stage to consume no randomness, because
the composed path runs it *before* the processor's noise draw while the monolith
draws the noise first. That holds today: the only RNG-consuming op ahead of the
blocks is ``pos_drop``, which is an ``nn.Identity`` because
``NoiseConditionedSFNOBuilder`` does not expose ``drop_rate``. The equivalence
test runs the parts in training mode, so it fails if that stops being true.

The parts do not change resolution. Resolution-changing operators are separate
registry entries (``interpolate`` and its successors) that a config chains
around a cut-point part.
"""

import dataclasses
from typing import Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.ace.stepper.single_module import load_weights_and_history
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.labels import LabelEncoding
from fme.core.models.conditional_sfno.layers import Context
from fme.core.models.conditional_sfno.sfnonet import SphericalFourierNeuralOperatorNet
from fme.core.registry.module import Module
from fme.core.weight_ops import overwrite_weights, strip_leading_module

from .modules import TransformModuleConfig, TransformSelector

__all__ = ["SFNOCutPointConfig"]

CutPointPart = Literal["encoder", "processor", "decoder"]
_PARTS: tuple[CutPointPart, ...] = ("encoder", "processor", "decoder")


class _SFNOEncoder(nn.Module):
    """The monolith's input stage: encoder stack, position embedding, dropout.

    Emits the ``embed_dim`` latent concatenated with the (unnormalized)
    big-skip residual; the processor normalizes that residual, so that the
    noise draw conditioning it is the same one the blocks see.
    """

    def __init__(
        self,
        net: SphericalFourierNeuralOperatorNet,
        embed_dim: int,
        big_skip: bool,
        checkpointing: int,
        clip_latent_global_means: bool,
        img_shape: tuple[int, int],
    ):
        super().__init__()
        self.encoder = net.encoder
        self.pos_drop = net.pos_drop
        self.pos_embed = net.pos_embed
        if big_skip:
            self.residual_filter_down = net.residual_filter_down
            self.residual_filter_up = net.residual_filter_up
        self._big_skip = big_skip
        self._checkpointing = checkpointing
        self._clip_latent_global_means = clip_latent_global_means
        self._spatial_h_slice, self._spatial_w_slice = (
            Distributed.get_instance().get_local_slices(img_shape)
        )
        if clip_latent_global_means:
            self.register_buffer(
                "_gm_min", torch.full((1, embed_dim, 1, 1), float("inf"))
            )
            self.register_buffer(
                "_gm_max", torch.full((1, embed_dim, 1, 1), float("-inf"))
            )
            self._gm_reset_pending: bool = False

    def request_latent_global_mean_envelope_reset(self) -> None:
        if self._clip_latent_global_means:
            self._gm_reset_pending = True

    def _apply_global_mean_clip(self, x: torch.Tensor) -> torch.Tensor:
        global_means = x.mean(dim=(-2, -1), keepdim=True)
        if self.training:
            with torch.no_grad():
                if self._gm_reset_pending:
                    self._gm_min.fill_(float("inf"))
                    self._gm_max.fill_(float("-inf"))
                    self._gm_reset_pending = False
                batch_min = global_means.detach().amin(dim=0, keepdim=True)
                batch_max = global_means.detach().amax(dim=0, keepdim=True)
                dist = Distributed.get_instance()
                dist.reduce_min(batch_min)
                dist.reduce_max(batch_max)
                self._gm_min.copy_(torch.minimum(self._gm_min, batch_min))
                self._gm_max.copy_(torch.maximum(self._gm_max, batch_max))
        elif torch.isfinite(self._gm_max).all():
            clipped = torch.clamp(global_means, min=self._gm_min, max=self._gm_max)
            x = x + (clipped - global_means)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._big_skip:
            residual = self.residual_filter_up(self.residual_filter_down(x))
        if self._checkpointing >= 1:
            latent = checkpoint(self.encoder, x)
        else:
            latent = self.encoder(x)
        if self.pos_embed is not None:
            latent = (
                latent
                + self.pos_embed[..., self._spatial_h_slice, self._spatial_w_slice]
            )
        latent = self.pos_drop(latent)
        if self._clip_latent_global_means:
            latent = self._apply_global_mean_clip(latent)
        if self._big_skip:
            return torch.cat((latent, residual), dim=1)
        return latent


class _SFNOProcessor(nn.Module):
    """The monolith's FNO blocks, plus the big-skip normalization.

    Takes and returns the cut-point tensor. The residual channels pass through
    the blocks untouched; they are normalized here (not in the encoder) so the
    normalization and the blocks share one noise draw, as in the monolith.
    """

    def __init__(
        self,
        net: SphericalFourierNeuralOperatorNet,
        embed_dim: int,
        big_skip: bool,
        checkpointing: int,
    ):
        super().__init__()
        self.blocks = net.blocks
        if big_skip:
            self.norm_big_skip = net.norm_big_skip
        self._embed_dim = embed_dim
        self._big_skip = big_skip
        self._checkpointing = checkpointing

    def forward(self, x: torch.Tensor, context: Context) -> torch.Tensor:
        latent = x[:, : self._embed_dim]
        if self._big_skip:
            residual = self.norm_big_skip(x[:, self._embed_dim :], context=context)
        for block in self.blocks:
            if self._checkpointing >= 3:
                latent = checkpoint(block, latent, context)
            else:
                latent = block(latent, context)
        if self._big_skip:
            return torch.cat((latent, residual), dim=1)
        return latent


class _SFNODecoder(nn.Module):
    """The monolith's output stage: decoder stack and optional output filter."""

    def __init__(
        self,
        net: SphericalFourierNeuralOperatorNet,
        checkpointing: int,
        filter_output: bool,
    ):
        super().__init__()
        self.decoder = net.decoder
        if filter_output:
            self.filter_output_down = net.filter_output_down
            self.filter_output_up = net.filter_output_up
        self._checkpointing = checkpointing
        self._filter_output = filter_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._checkpointing >= 1:
            out = checkpoint(self.decoder, x)
        else:
            out = self.decoder(x)
        if self._filter_output:
            out = self.filter_output_up(self.filter_output_down(out))
        return out


class _UnconditionalCutPoint(nn.Module):
    """Holds a context-free part under the donor's ``conditional_model`` name.

    The encoder and decoder parts consume no context, so they need none of
    :class:`NoiseConditionedModel`'s noise machinery — but their parameters
    must keep the donor's ``state_dict`` names, which are nested under
    ``conditional_model``. The leading-dimension flattening matches
    ``NoiseConditionedModel.forward`` so all three parts accept the same
    shapes.
    """

    def __init__(self, conditional_model: nn.Module):
        super().__init__()
        self.conditional_model = conditional_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conditional_model(x.reshape(-1, *x.shape[-3:]))


@TransformSelector.register("sfno_cut_point")
@dataclasses.dataclass
class SFNOCutPointConfig(TransformModuleConfig):
    """One stage of a decomposed noise-conditioned SFNO, as a pool component.

    The ``sfno`` block is the *donor's* module config: all three parts of one
    decomposition are configured with the same block, and each builds only the
    parameters belonging to its stage. Channel counts come from the domains, so
    the config validates that they agree with ``sfno.embed_dim`` and
    ``sfno.big_skip`` (see the module docstring for the cut-point channel
    layout).

    Parameters:
        part: Which stage to build.
        sfno: The donor's noise-conditioned SFNO configuration.
        donor_checkpoint: Path to an ACE stepper checkpoint to name-match this
            part's parameters against. Every parameter of the built part must be
            present in the donor *and* have the donor's shape, which holds when
            ``sfno`` is the donor's own config and the cut-point domain declares
            ``embed_dim`` plus the donor's input channels; anything else raises
            rather than partly initializing the part. Applied before
            ``TransformConfig.parameter_init``, so an explicit ``weights_path``
            still wins over the donor. Only parameters are transferred, not
            buffers: an encoder configured with ``clip_latent_global_means``
            therefore starts with an empty envelope and relearns it over the
            first training epoch rather than inheriting the donor's.
        donor_module_index: Index into the donor stepper's ``modules`` list.
        conditional: Whether to pass batch labels through to the context.
            Only the processor consumes context.
    """

    part: CutPointPart
    sfno: NoiseConditionedSFNOBuilder = dataclasses.field(
        default_factory=NoiseConditionedSFNOBuilder
    )
    donor_checkpoint: str | None = None
    donor_module_index: int = 0
    conditional: bool = False

    def __post_init__(self):
        if self.part not in _PARTS:
            raise ValueError(
                f"Unknown sfno_cut_point part {self.part!r}; expected one of "
                f"{list(_PARTS)}."
            )
        if self.conditional and self.part != "processor":
            raise ValueError(
                "Only the processor part of an sfno_cut_point consumes context, "
                f"so conditional=True is not meaningful for the {self.part!r} "
                "part."
            )

    def _latent_channels(self, n_in_channels: int, n_out_channels: int) -> int:
        """The cut-point tensor's channel count, from this part's domains."""
        return n_out_channels if self.part == "encoder" else n_in_channels

    def _donor_in_channels(self, n_in_channels: int, n_out_channels: int) -> int:
        """The donor SFNO's input channel count, from this part's domains.

        Only the encoder sees it directly. The others recover it from the
        cut-point width, which carries the big-skip residual; when there is no
        big skip they keep nothing sized by it, so any positive placeholder does
        (``norm_big_skip`` is still *built* at that size when
        ``normalize_big_skip`` is set, but no part keeps it without a big skip).
        """
        if self.part == "encoder":
            return n_in_channels
        if not self.sfno.big_skip:
            return 0
        return n_in_channels - self.sfno.embed_dim

    def _validate_channels(self, n_in_channels: int, n_out_channels: int) -> None:
        embed_dim = self.sfno.embed_dim
        latent = self._latent_channels(n_in_channels, n_out_channels)
        if self.part == "processor" and n_in_channels != n_out_channels:
            raise ValueError(
                "An sfno_cut_point processor maps the cut-point to itself, so "
                "its input and output domains must have the same number of "
                f"channels, got {n_in_channels} and {n_out_channels}."
            )
        if self.sfno.big_skip:
            if latent <= embed_dim:
                raise ValueError(
                    f"An sfno_cut_point {self.part!r} with big_skip expects a "
                    f"cut-point of embed_dim ({embed_dim}) plus the donor's "
                    f"input channels, got {latent} channels."
                )
            if self.part == "encoder" and latent != embed_dim + n_in_channels:
                raise ValueError(
                    "An sfno_cut_point encoder with big_skip emits embed_dim "
                    f"({embed_dim}) plus its own input channels "
                    f"({n_in_channels}) = {embed_dim + n_in_channels} channels, "
                    f"but its output domain declares {latent}."
                )
        elif latent != embed_dim:
            raise ValueError(
                f"An sfno_cut_point {self.part!r} without big_skip expects a "
                f"cut-point of exactly embed_dim ({embed_dim}) channels, got "
                f"{latent}."
            )

    def _build_part(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        """Build the donor SFNO and keep only this part's submodules."""
        donor_in_channels = self._donor_in_channels(n_in_channels, n_out_channels)
        built = self.sfno.build(
            # Stages the part discards are still constructed here, so that the
            # ones it keeps are built exactly as the donor builds them. Only
            # the discarded stages' channel counts are free; they must be
            # positive.
            n_in_channels=max(donor_in_channels, 1),
            n_out_channels=n_out_channels if self.part == "decoder" else 1,
            dataset_info=dataset_info,
        )
        net = built.conditional_model
        if self.part == "encoder":
            return _UnconditionalCutPoint(
                _SFNOEncoder(
                    net,
                    embed_dim=self.sfno.embed_dim,
                    big_skip=self.sfno.big_skip,
                    checkpointing=self.sfno.checkpointing,
                    clip_latent_global_means=self.sfno.clip_latent_global_means,
                    img_shape=dataset_info.img_shape,
                )
            )
        if self.part == "decoder":
            return _UnconditionalCutPoint(
                _SFNODecoder(
                    net,
                    checkpointing=self.sfno.checkpointing,
                    filter_output=self.sfno.filter_output,
                )
            )
        # The processor keeps NoiseConditionedModel's noise machinery (and its
        # state_dict names) and swaps the net it wraps for the blocks alone.
        built.conditional_model = _SFNOProcessor(
            net,
            embed_dim=self.sfno.embed_dim,
            big_skip=self.sfno.big_skip,
            checkpointing=self.sfno.checkpointing,
        )
        return built

    def _apply_donor_weights(self, module: nn.Module) -> None:
        weights, _ = load_weights_and_history(self.donor_checkpoint)
        if weights is None:
            raise ValueError(
                f"Donor checkpoint {self.donor_checkpoint!r} contains no module "
                "weights."
            )
        if not 0 <= self.donor_module_index < len(weights):
            raise ValueError(
                f"donor_module_index {self.donor_module_index} is out of range "
                f"for a donor stepper with {len(weights)} module(s)."
            )
        donor = strip_leading_module(weights[self.donor_module_index])
        missing = []
        mismatched = []
        for name, parameter in module.named_parameters():
            if name not in donor:
                missing.append(name)
            elif donor[name].shape != parameter.shape:
                mismatched.append(
                    f"{name} (donor {tuple(donor[name].shape)}, "
                    f"part {tuple(parameter.shape)})"
                )
        if missing:
            raise ValueError(
                f"The donor checkpoint {self.donor_checkpoint!r} has no weights "
                f"for {len(missing)} parameter(s) of this sfno_cut_point "
                f"{self.part!r}, e.g. {sorted(missing)[:5]}. The 'sfno' block "
                "must be the donor's own module configuration."
            )
        if mismatched:
            # overwrite_weights would copy the leading slice of each of these and
            # leave the rest at its random initialization, silently producing a
            # part that is only partly the donor's.
            raise ValueError(
                f"The donor checkpoint {self.donor_checkpoint!r} has "
                f"differently-shaped weights for {len(mismatched)} parameter(s) "
                f"of this sfno_cut_point {self.part!r}: {sorted(mismatched)[:5]}. "
                "The usual cause is a cut-point domain whose channel count is "
                "not embed_dim plus the donor's own input channels. For a "
                "deliberately partial load, use parameter_init.weights_path "
                "instead of donor_checkpoint."
            )
        destination = set(module.state_dict())
        overwrite_weights(
            {name: value for name, value in donor.items() if name in destination},
            module,
        )

    def _build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        in_dataset_info: DatasetInfo,
        out_dataset_info: DatasetInfo,
        load_donor: bool,
    ) -> Module:
        if in_dataset_info.img_shape != out_dataset_info.img_shape:
            raise ValueError(
                "An sfno_cut_point part does not change resolution, but got "
                f"input grid {in_dataset_info.img_shape} and output grid "
                f"{out_dataset_info.img_shape}. Chain it with a "
                "resolution-changing transform instead."
            )
        if self.conditional and len(in_dataset_info.all_labels) == 0:
            raise ValueError("Conditional predictions require labels")
        self._validate_channels(n_in_channels, n_out_channels)
        module = self._build_part(n_in_channels, n_out_channels, in_dataset_info)
        if load_donor and self.donor_checkpoint is not None:
            self._apply_donor_weights(module)
        label_encoding = (
            LabelEncoding(sorted(in_dataset_info.all_labels))
            if self.conditional
            else None
        )
        return Module(module, label_encoding=label_encoding)

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        in_dataset_info: DatasetInfo,
        out_dataset_info: DatasetInfo,
    ) -> Module:
        return self._build(
            n_in_channels,
            n_out_channels,
            in_dataset_info,
            out_dataset_info,
            load_donor=True,
        )

    def build_for_load(
        self,
        n_in_channels: int,
        n_out_channels: int,
        in_dataset_info: DatasetInfo,
        out_dataset_info: DatasetInfo,
    ) -> Module:
        """Build without reading the donor checkpoint.

        The weights are about to be restored from a saved state, so the donor
        is not consulted — the component reloads without the donor checkpoint
        still existing.
        """
        return self._build(
            n_in_channels,
            n_out_channels,
            in_dataset_info,
            out_dataset_info,
            load_donor=False,
        )

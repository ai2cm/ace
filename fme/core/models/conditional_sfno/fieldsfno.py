"""Field-space spherical Fourier neural operator.

A variant of ``SphericalFourierNeuralOperatorNet`` whose residual stream is
the field itself rather than a learned latent, in the spirit of the
field-space attention network of FieldSpaceNN
(https://github.com/FREVA-CLINT/FieldSpaceNN). There is no encoder, decoder,
positional embedding, or big skip: each block's spectral filter and MLP
read the field, compute at their own widths (set by ``embed_dim``), and are
added back onto the field through learned gates initialized near zero, so
the network starts as the identity map on the prognostic fields.

The network starts as the identity map on the prognostic channels, so it
already predicts persistence at initialization. Use it with the step's
``residual_prediction`` off: a step that adds the network output onto the
input would predict twice the input at initialization.

Channel contract (not enforced on names; order channels in configs to match):

- input channels are ``[prognostic..., forcing...]``: the first
  ``prognostic_chans`` channels are prognostic fields, everything after
  (forcing fields, and any extra channels the step packs after its inputs)
  is forcing.
- output channels are ``[prognostic..., diagnostic...]``: the prognostic
  fields in the same order as the input, then the diagnostic fields.

Forcing channels are not part of the residual stream. They are encoded once
by a small pointwise network and condition every conditional layer
normalization through ``Context.embedding_forcing``. Diagnostic channels are
initialized from learned per-channel values (zero at init) and then evolved
by the blocks like the prognostic channels.
"""

import dataclasses
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from fme.core.benchmark.timer import NullTimer, Timer
from fme.core.distributed import Distributed

from .layers import MLP, ConditionalLayerNorm, Context, ContextConfig, DropPath
from .sfnonet import SpectralFilterLayer

ACTIVATION_FUNCTIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


@dataclasses.dataclass
class FieldSFNONetConfig:
    """Configuration parameters for FieldSpaceSphericalFourierNeuralOperatorNet.

    Attributes:
        filter_type: Type of spectral filter to use ('linear', 'makani-linear',
            'local').
        embed_dim: Base width of the block branches. The MLP hidden layer has
            ``int(embed_dim * mlp_ratio)`` channels and the spectral filter
            runs on ``round(embed_dim * spectral_ratio)`` channels; neither
            depends on the number of fields.
        spectral_ratio: Fraction of ``embed_dim`` the spectral filter runs
            on. Each block projects the field to
            ``round(embed_dim * spectral_ratio)`` channels with a 1x1
            convolution, applies the filter (and inner skip) at that width,
            and projects back to the field channels.
        global_layer_norm: Whether to reduce along the spatial domain when
            applying layer normalization.
        num_layers: Number of field-space FNO blocks.
        use_mlp: Whether to use an MLP in each block.
        mlp_ratio: Ratio of the MLP hidden dimension to ``embed_dim``.
        activation_function: Activation function name ('relu', 'gelu', 'silu').
        forcing_embed_dim: Width of the forcing embedding that conditions the
            layer norms. Ignored when the model has no forcing channels.
        forcing_encoder_layers: Number of 1x1 convolution layers in the
            forcing encoder (activations between layers, none after the last).
        drop_rate: Dropout rate inside the MLPs.
        drop_path_rate: Stochastic depth rate (drop path), applied to each
            block's filter and MLP branches.
        hard_thresholding_fraction: Fraction of spectral modes to retain.
        checkpointing: Gradient checkpointing level (0 = none, >=2 checkpoints
            the MLPs, >=3 checkpoints whole blocks).
        filter_num_groups: Number of groups in the spectral filter's grouped
            convolution. Must divide the spectral filter width
            ``round(embed_dim * spectral_ratio)``.
        filter_residual: Whether to pass each block's inner skip through a
            SHT round-trip.
        block_inner_skip: Whether each block adds a learned 1x1 convolution
            of the spectral filter's input to its output before the
            activation, as ``SphericalFourierNeuralOperatorNet`` always
            does. With the field itself as the residual stream this skip
            may be redundant with the block's gated residual.
        block_mlp_input: What each block's MLP reads. ``"field"``: the
            block makes two gated updates to the field, the MLP reading the
            field after the filter branch has updated it. ``"filter"``: the
            MLP reads the filter branch's output directly and the block
            makes one gated update at the end, so spectral changes need not
            be written into the field before the MLP can act on them.
            Ignored when ``use_mlp`` is False.
        filter_output: Whether to pass the network's update (output minus
            input state) through a SHT round-trip.
        local_blocks: Indices of blocks that use discrete-continuous
            convolution (DISCO) local filters instead of spectral filters.
        affine_norms: Whether to use element-wise affine parameters in the
            normalization layers.
        filter_preserves_global_mean: If True, the spectral filter preserves
            the l=0 (global mean) spherical harmonic coefficient of its
            (projected) input; the projections around the filter are
            pointwise, so no spectral mixing changes the global mean.
        gamma_init: Initial value of the residual gates. Each block computes
            ``x = (1 + gamma_res) * x + gamma * branch(x)``; the FieldSpaceNN
            default of 1e-12 makes every block the identity at init.
    """

    filter_type: Literal["linear", "makani-linear", "local"] = "linear"
    embed_dim: int = 256
    spectral_ratio: float = 1.0
    global_layer_norm: bool = False
    num_layers: int = 12
    use_mlp: bool = True
    mlp_ratio: float = 2.0
    activation_function: str = "gelu"
    forcing_embed_dim: int = 32
    forcing_encoder_layers: int = 2
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    hard_thresholding_fraction: float = 1.0
    checkpointing: int = 0
    filter_num_groups: int = 1
    filter_residual: bool = False
    block_inner_skip: bool = True
    block_mlp_input: Literal["field", "filter"] = "field"
    filter_output: bool = False
    local_blocks: list[int] | None = None
    affine_norms: bool = False
    filter_preserves_global_mean: bool = False
    gamma_init: float = 1e-12

    def __post_init__(self):
        if self.block_mlp_input not in ("field", "filter"):
            raise ValueError(
                f"block_mlp_input must be 'field' or 'filter', "
                f"got {self.block_mlp_input!r}"
            )
        if self.activation_function not in ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unknown activation function {self.activation_function}, "
                f"expected one of {sorted(ACTIVATION_FUNCTIONS)}"
            )
        if self.forcing_encoder_layers < 1:
            raise ValueError("forcing_encoder_layers must be at least 1")
        if self.forcing_embed_dim < 1:
            raise ValueError("forcing_embed_dim must be at least 1")
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be at least 1")
        if not 0.0 < self.spectral_ratio <= 1.0:
            raise ValueError(
                f"spectral_ratio must be in (0, 1], got {self.spectral_ratio}"
            )
        if self.spectral_dim < 1:
            raise ValueError(
                f"spectral_ratio={self.spectral_ratio} with "
                f"embed_dim={self.embed_dim} yields no spectral channels"
            )
        if self.spectral_dim % self.filter_num_groups != 0:
            raise ValueError(
                f"filter_num_groups={self.filter_num_groups} must divide the "
                f"spectral filter width round(embed_dim * spectral_ratio)="
                f"{self.spectral_dim}"
            )
        if int(self.embed_dim * self.mlp_ratio) < 1:
            raise ValueError(
                f"mlp_ratio={self.mlp_ratio} with embed_dim={self.embed_dim} "
                "yields no MLP hidden channels"
            )
        if self.local_blocks is not None:
            bad = [i for i in self.local_blocks if not 0 <= i < self.num_layers]
            if bad:
                raise ValueError(
                    f"local_blocks {bad} out of range for num_layers={self.num_layers}"
                )

    @property
    def spectral_dim(self) -> int:
        """Width the spectral filter runs on."""
        return round(self.embed_dim * self.spectral_ratio)

    @property
    def mlp_hidden_dim(self) -> int:
        """Width of the MLP hidden layer."""
        return int(self.embed_dim * self.mlp_ratio)


class GatedResidual(nn.Module):
    """Computes ``(1 + gamma_res) * x + gamma * branch`` with per-channel gates.

    Mirrors the ``gamma``/``gamma_res`` gates of FieldSpaceNN's field-space
    attention block: both start at ``gamma_init`` so the update is the
    identity at initialization.
    """

    def __init__(self, n_channels: int, gamma_init: float):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, n_channels, 1, 1), gamma_init))
        self.gamma_res = nn.Parameter(torch.full((1, n_channels, 1, 1), gamma_init))

    def forward(self, x: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return (1.0 + self.gamma_res) * x + self.gamma * branch


class FieldSpaceFourierNeuralOperatorBlock(nn.Module):
    """A Fourier neural operator block acting directly on field channels.

    With ``mlp_input="field"``, two gated residual updates on the field ``x``:

    1. ``u = proj_in(norm0(x))``, ``y = proj_out(act(filter(u) + inner_skip(u)))``,
       then ``x = (1 + gamma_res) * x + gamma * y`` (the ``inner_skip`` term
       is omitted when ``block_inner_skip`` is False)
    2. ``z = mlp(norm1(x))``, then ``x = (1 + gamma_res') * x + gamma' * z``

    With ``mlp_input="filter"``, one gated update: ``y`` is computed as
    above, then ``z = mlp(norm1(y))`` and ``x = (1 + gamma_res) * x + gamma * z``.
    The filter branch's output reaches the field only through the MLP.
    Without an MLP (``use_mlp=False``) only update 1 runs, whatever
    ``mlp_input`` is.

    ``proj_in``/``proj_out`` are 1x1 convolutions between the ``n_channels``
    field channels and the ``spectral_dim`` channels the filter runs on; the
    MLP maps ``n_channels`` through ``mlp_hidden_dim`` back to ``n_channels``.
    """

    def __init__(
        self,
        forward_transform: nn.Module,
        inverse_transform: nn.Module,
        n_channels: int,
        spectral_dim: int,
        mlp_hidden_dim: int,
        img_shape: tuple[int, int],
        context_config: ContextConfig,
        filter_type: str = "linear",
        global_layer_norm: bool = False,
        drop_rate: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        use_mlp: bool = True,
        checkpointing: int = 0,
        filter_residual: bool = False,
        block_inner_skip: bool = True,
        mlp_input: Literal["field", "filter"] = "field",
        affine_norms: bool = False,
        filter_num_groups: int = 1,
        filter_preserves_global_mean: bool = False,
        gamma_init: float = 1e-12,
    ):
        super().__init__()
        if mlp_input not in ("field", "filter"):
            raise ValueError(
                f"mlp_input must be 'field' or 'filter', got {mlp_input!r}"
            )
        self.mlp_input = mlp_input
        self.norm0 = ConditionalLayerNorm(
            n_channels,
            img_shape=img_shape,
            global_layer_norm=global_layer_norm,
            context_config=context_config,
            elementwise_affine=affine_norms,
        )
        self.proj_in = nn.Conv2d(n_channels, spectral_dim, 1)
        self.filter = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            spectral_dim,
            filter_type,
            filter_residual=filter_residual,
            num_groups=filter_num_groups,
            preserve_global_mean=filter_preserves_global_mean,
        )
        self.inner_skip: nn.Module | None = (
            nn.Conv2d(spectral_dim, spectral_dim, 1) if block_inner_skip else None
        )
        self.act_layer = act_layer()
        self.proj_out = nn.Conv2d(spectral_dim, n_channels, 1)
        self.gate_filter = GatedResidual(n_channels, gamma_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if use_mlp:
            self.norm1: nn.Module | None = ConditionalLayerNorm(
                n_channels,
                img_shape=img_shape,
                global_layer_norm=global_layer_norm,
                context_config=context_config,
                elementwise_affine=affine_norms,
            )
            self.mlp: nn.Module | None = MLP(
                in_features=n_channels,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing=checkpointing,
            )
            self.gate_mlp: GatedResidual | None = (
                GatedResidual(n_channels, gamma_init) if mlp_input == "field" else None
            )
        else:
            self.norm1 = None
            self.mlp = None
            self.gate_mlp = None

    def forward(
        self, x: torch.Tensor, context: Context, timer: Timer = NullTimer()
    ) -> torch.Tensor:
        with timer.child("norm0") as norm0_timer:
            x_norm = self.norm0(x, context, timer=norm0_timer)
        with timer.child("proj_in"):
            u = self.proj_in(x_norm)
        with timer.child("filter") as filter_timer:
            y, residual = self.filter(u, timer=filter_timer)
        if self.inner_skip is not None:
            with timer.child("inner_skip"):
                y = y + self.inner_skip(residual)
        with timer.child("activation"):
            y = self.act_layer(y)
        with timer.child("proj_out"):
            y = self.proj_out(y)
        if self.mlp is None:
            x = self.gate_filter(x, self.drop_path(y))
        elif self.mlp_input == "filter":
            # The MLP consumes the filter branch; the field is updated once,
            # at the end of the block.
            assert self.norm1 is not None
            with timer.child("norm1") as norm1_timer:
                y_norm = self.norm1(y, context, timer=norm1_timer)
            with timer.child("mlp"):
                z = self.mlp(y_norm)
            x = self.gate_filter(x, self.drop_path(z))
        else:
            # The filter branch updates the field, then the MLP reads the
            # updated field and updates it again.
            assert self.norm1 is not None and self.gate_mlp is not None
            x = self.gate_filter(x, self.drop_path(y))
            with timer.child("norm1") as norm1_timer:
                x_norm = self.norm1(x, context, timer=norm1_timer)
            with timer.child("mlp"):
                z = self.mlp(x_norm)
            x = self.gate_mlp(x, self.drop_path(z))
        return x


class FieldSpaceSphericalFourierNeuralOperatorNet(nn.Module):
    """Spherical Fourier neural operator whose residual stream is the field.

    See the module docstring for the channel contract.

    Args:
        params: Model configuration. See ``FieldSFNONetConfig``.
        img_shape: Global spatial shape (lat, lon) of the data.
        trans_down: Spherical harmonic transform on the data grid.
        itrans_up: Inverse spherical harmonic transform on the data grid.
        prognostic_chans: Number of prognostic channels (in both input and
            output).
        forcing_chans: Number of forcing channels (input only).
        diagnostic_chans: Number of diagnostic channels (output only).
        context_config: Dimensions of the externally provided context
            embeddings. ``embed_dim_forcing`` is set by this class from
            ``params.forcing_embed_dim`` and must be left at 0.
    """

    def __init__(
        self,
        params: FieldSFNONetConfig,
        img_shape: tuple[int, int],
        trans_down: nn.Module,
        itrans_up: nn.Module,
        prognostic_chans: int,
        forcing_chans: int,
        diagnostic_chans: int,
        context_config: ContextConfig = ContextConfig(
            embed_dim_scalar=0,
            embed_dim_labels=0,
            embed_dim_noise=0,
            embed_dim_pos=0,
        ),
    ):
        super().__init__()
        if prognostic_chans < 1:
            raise ValueError("prognostic_chans must be at least 1")
        if forcing_chans < 0 or diagnostic_chans < 0:
            raise ValueError("forcing_chans and diagnostic_chans must be >= 0")
        if context_config.embed_dim_forcing != 0:
            raise ValueError(
                "context_config.embed_dim_forcing is set by the network from "
                "params.forcing_embed_dim; pass it as 0"
            )
        self.prognostic_chans = prognostic_chans
        self.forcing_chans = forcing_chans
        self.diagnostic_chans = diagnostic_chans
        self.in_chans = prognostic_chans + forcing_chans
        self.out_chans = prognostic_chans + diagnostic_chans
        self.checkpointing = params.checkpointing
        self.filter_output = params.filter_output
        self.trans_down = trans_down
        self.itrans_up = itrans_up

        act_layer = ACTIVATION_FUNCTIONS[params.activation_function]

        if diagnostic_chans > 0:
            self.diagnostic_init: nn.Parameter | None = nn.Parameter(
                torch.zeros(1, diagnostic_chans, 1, 1)
            )
        else:
            self.diagnostic_init = None

        if forcing_chans > 0:
            layers: list[nn.Module] = []
            width_in = forcing_chans
            for i in range(params.forcing_encoder_layers):
                layers.append(nn.Conv2d(width_in, params.forcing_embed_dim, 1))
                if i < params.forcing_encoder_layers - 1:
                    layers.append(act_layer())
                width_in = params.forcing_embed_dim
            self.forcing_encoder: nn.Module | None = nn.Sequential(*layers)
            context_config = dataclasses.replace(
                context_config, embed_dim_forcing=params.forcing_embed_dim
            )
        else:
            self.forcing_encoder = None

        local_blocks = set(params.local_blocks or [])
        dpr = [
            x.item()
            for x in torch.linspace(0, params.drop_path_rate, params.num_layers)
        ]
        self.blocks = nn.ModuleList(
            [
                FieldSpaceFourierNeuralOperatorBlock(
                    forward_transform=trans_down,
                    inverse_transform=itrans_up,
                    n_channels=self.out_chans,
                    spectral_dim=params.spectral_dim,
                    mlp_hidden_dim=params.mlp_hidden_dim,
                    img_shape=img_shape,
                    context_config=context_config,
                    filter_type="local" if i in local_blocks else params.filter_type,
                    global_layer_norm=params.global_layer_norm,
                    drop_rate=params.drop_rate,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    use_mlp=params.use_mlp,
                    checkpointing=params.checkpointing,
                    filter_residual=params.filter_residual,
                    block_inner_skip=params.block_inner_skip,
                    mlp_input=params.block_mlp_input,
                    affine_norms=params.affine_norms,
                    filter_num_groups=params.filter_num_groups,
                    filter_preserves_global_mean=params.filter_preserves_global_mean,
                    gamma_init=params.gamma_init,
                )
                for i in range(params.num_layers)
            ]
        )

    def _initial_state(self, x: torch.Tensor) -> torch.Tensor:
        prognostic = x[:, : self.prognostic_chans]
        if self.diagnostic_init is None:
            return prognostic
        diagnostic = self.diagnostic_init.to(dtype=x.dtype).expand(
            x.shape[0], self.diagnostic_chans, *x.shape[-2:]
        )
        return torch.cat((prognostic, diagnostic), dim=1)

    def forward(self, x: torch.Tensor, context: Context) -> torch.Tensor:
        if x.shape[-3] != self.in_chans:
            raise ValueError(
                f"Expected {self.in_chans} input channels "
                f"({self.prognostic_chans} prognostic + {self.forcing_chans} "
                f"forcing), got {x.shape[-3]}"
            )
        if self.forcing_encoder is not None:
            forcing = x[:, self.prognostic_chans :]
            context = dataclasses.replace(
                context, embedding_forcing=self.forcing_encoder(forcing)
            )
        state_in = self._initial_state(x)
        state = state_in
        for blk in self.blocks:
            if self.checkpointing >= 3:
                state = checkpoint(blk, state, context, use_reentrant=False)
            else:
                state = blk(state, context)
        if self.filter_output:
            update = state - state_in
            state = state_in + self.itrans_up(self.trans_down(update))
        return state


def get_lat_lon_fieldsfno(
    params: FieldSFNONetConfig,
    prognostic_chans: int,
    forcing_chans: int,
    diagnostic_chans: int,
    img_shape: tuple[int, int],
    data_grid: str = "equiangular",
    context_config: ContextConfig = ContextConfig(
        embed_dim_scalar=0,
        embed_dim_noise=0,
        embed_dim_labels=0,
        embed_dim_pos=0,
    ),
) -> FieldSpaceSphericalFourierNeuralOperatorNet:
    h, w = img_shape
    modes_lat = int(h * params.hard_thresholding_fraction)
    modes_lon = int((w // 2 + 1) * params.hard_thresholding_fraction)
    dist = Distributed.get_instance()
    trans_down = dist.get_sht(h, w, lmax=modes_lat, mmax=modes_lon, grid=data_grid)
    itrans_up = dist.get_isht(h, w, lmax=modes_lat, mmax=modes_lon, grid=data_grid)
    return FieldSpaceSphericalFourierNeuralOperatorNet(
        params,
        img_shape=img_shape,
        trans_down=trans_down,
        itrans_up=itrans_up,
        prognostic_chans=prognostic_chans,
        forcing_chans=forcing_chans,
        diagnostic_chans=diagnostic_chans,
        context_config=context_config,
    )

"""A two-track (global / local latent) spherical Fourier neural operator.

This extends the conditional SFNO (see ``sfnonet.py``) so that only a subset
of the latent channels -- the *global* track -- passes through the spherical
harmonic transform and the per-mode spectral weight. The remaining *local*
track sees only pointwise operations (conditional norm, activation, MLP, skip
connections and, optionally, a dense ``conv1x1``), so its horizontal gradients
are never distorted by the spatial mixing. The motivation is a more physical
prior for out-of-sample climates and support for column-local
variable-transformation bases (see the sibling investigation).

The network is deliberately semantics-agnostic: it takes a single input tensor
whose leading ``global_in_channels`` channels are the global-track inputs and
whose trailing channels are the local-track inputs, and returns a single
output tensor ordered ``[global_out | local_out]``. The step owns the
variable-to-track assignment and the packing/unpacking.

Design invariant (pins the regression test): when the local track has zero
width and all three options are off, the module tree and every parameter name
are identical to the single-track ``SphericalFourierNeuralOperatorNet``, so an
old single-track checkpoint loads directly with no remap.

The *forward output* additionally reduces to the single-track net byte-for-byte
only when the spectral filter uses no round-trip residual. ``filter_residual``
is one way to turn that residual on; the other is ``data_grid='equiangular'``,
which gives the first and last blocks a forward/inverse transform on different
grids and so forces ``SpectralConvS2._round_trip_residual`` true for those
blocks. The single-track block then feeds its skip from the round-tripped
residual, whereas this two-track block always takes the skip from the full
normed input (see ``TwoTrackFourierNeuralOperatorBlock.forward``). Output
equivalence therefore holds for the default ``data_grid='legendre-gauss'`` (all
transforms share a grid, no round trip) but NOT for ``'equiangular'``. The
regression test exercises the legendre-gauss case; a checkpoint trained with
``'equiangular'`` loads without error but does not reproduce its output.
"""

import dataclasses
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from fme.core.benchmark.timer import NullTimer, Timer
from fme.core.distributed import Distributed

from .initialization import trunc_normal_
from .latents import Latents
from .layers import MLP, ConditionalLayerNorm, Context, ContextConfig, DropPath
from .lora import LoRAConv2d
from .s2convolutions import validate_spectral_ratio
from .sfnonet import NoLayerNorm, SpectralFilterLayer

ACTIVATION_FUNCTIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


@dataclasses.dataclass
class TwoTrackSFNONetConfig:
    """Configuration for ``TwoTrackSphericalFourierNeuralOperatorNet``.

    Attributes:
        embed_dim: Total latent width, i.e. global + local. The spectral filter
            is sized to the global width ``embed_dim - local_embed_dim``.
        local_embed_dim: Width of the local (pointwise-only) latent track. 0
            (default) recovers the single-track network.
        filter_type: Spectral filter type. Only "linear" is supported.
        scale_factor: Must be 1 (matches the single-track network).
        global_layer_norm: Whether the conditional norm reduces over the whole
            spatial domain rather than per-pixel over channels.
        num_layers: Number of Fourier neural operator blocks.
        use_mlp: Whether each block uses an MLP.
        mlp_ratio: MLP hidden-dim ratio.
        activation_function: One of "relu", "gelu", "silu".
        encoder_layers: Number of encoder/decoder hidden conv layers.
        pos_embed: Whether to add a learned positional embedding to the latent.
        drop_rate: Dropout rate.
        drop_path_rate: Stochastic-depth rate.
        hard_thresholding_fraction: Fraction of spectral modes retained.
        big_skip: Whether to use an input->decoder skip connection.
        checkpointing: Gradient-checkpointing level (0=none, 1=encoder/decoder,
            3=all blocks).
        filter_num_groups: Number of groups in the grouped spectral conv.
        filter_residual: Whether the spectral filter round-trips its residual
            through the SHT. Off in the baseline; when on, L=0 equivalence with
            the single-track net no longer holds byte-for-byte (the block's
            skip residual is taken from the full normed input, not the filter's
            round-tripped global residual).
        filter_output: Whether to filter the global decoder output through an
            SHT round-trip.
        normalize_big_skip: Whether to normalize the big-skip connection.
        affine_norms: Whether the norm layers use element-wise affine params.
        feed_global_to_local: OPTION 1. Also feed the global inputs to the
            local encoder. Default off.
        parallel_conv1x1: OPTION 2. Add a dense pointwise all->all map whose
            output is summed onto the (global-only) spectral output via Latents
            addition, so ``global = spectral + conv1x1[global]`` and
            ``local = conv1x1[local]``. Default off; when off the local
            filter-stage output is zero, carried by the inner skip.
        per_track_layer_norm: OPTION 3. Compute each conditional layer norm
            separately per track rather than jointly over the concatenation.
            Default off.
    """

    embed_dim: int = 256
    local_embed_dim: int = 0
    filter_type: str = "linear"
    scale_factor: int = 1
    global_layer_norm: bool = False
    num_layers: int = 12
    use_mlp: bool = True
    mlp_ratio: float = 2.0
    activation_function: str = "gelu"
    encoder_layers: int = 1
    pos_embed: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    hard_thresholding_fraction: float = 1.0
    big_skip: bool = True
    checkpointing: int = 0
    filter_num_groups: int = 1
    spectral_ratio: float = 1.0
    filter_residual: bool = False
    filter_output: bool = False
    normalize_big_skip: bool = False
    affine_norms: bool = False
    feed_global_to_local: bool = False
    parallel_conv1x1: bool = False
    per_track_layer_norm: bool = False

    def __post_init__(self):
        if self.filter_type != "linear":
            raise NotImplementedError(
                "TwoTrackSFNONet only supports filter_type='linear', got "
                f"'{self.filter_type}'."
            )
        if self.scale_factor != 1:
            raise NotImplementedError("scale_factor must be 1.")
        if self.local_embed_dim < 0:
            raise ValueError("local_embed_dim must be >= 0.")
        if self.local_embed_dim >= self.embed_dim:
            raise ValueError(
                "local_embed_dim must be < embed_dim so the global track has at "
                f"least one channel; got local_embed_dim={self.local_embed_dim}, "
                f"embed_dim={self.embed_dim}."
            )
        global_channels = self.embed_dim - self.local_embed_dim
        if global_channels % self.filter_num_groups != 0:
            raise ValueError(
                f"global latent width {global_channels} (embed_dim - "
                f"local_embed_dim) must be divisible by filter_num_groups="
                f"{self.filter_num_groups}."
            )
        # The spectral filter is sized to the global width, so spectral_ratio is
        # validated against global_channels (not embed_dim). To keep the actual
        # filter (channels-per-group x num-groups) fixed when the global track is
        # narrowed relative to a single-track baseline, scale spectral_ratio up
        # by the same factor the global width was scaled down.
        validate_spectral_ratio(
            self.spectral_ratio,
            global_channels,
            self.filter_num_groups,
            channels_name="global latent width (embed_dim - local_embed_dim)",
            filter_type=self.filter_type,
        )

    @property
    def global_embed_dim(self) -> int:
        return self.embed_dim - self.local_embed_dim


class TwoTrackFourierNeuralOperatorBlock(nn.Module):
    """A FNO block whose spectral filter acts on only the global slice.

    The pointwise parts (conditional norm, activation, MLP, inner/outer skips)
    act on the full ``[global | local]`` concatenation. The spectral filter
    reads and writes only the global slice; the local slice's filter-stage
    output is zero (or, with option 2, the conv1x1 local map) and is carried
    forward by the inner-skip residual over the full width.

    When ``local_channels == 0`` and both options are off, the submodule tree
    and parameter names are identical to ``sfnonet.FourierNeuralOperatorBlock``
    with ``inner_skip="linear"`` and ``outer_skip="identity"``. The forward
    computation matches it too *only when the filter returns no round-trip
    residual*: the skips here are always taken from the full normed input
    ``x_norm``, while the single-track block takes them from the filter's
    returned residual, which equals ``x_norm`` only when the filter does not
    round-trip (see the module docstring for when it does -- ``filter_residual``
    or ``data_grid='equiangular'``).
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim: int,
        global_channels: int,
        img_shape: tuple[int, int],
        context_config: ContextConfig,
        global_layer_norm: bool = False,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        use_mlp: bool = False,
        checkpointing: int = 0,
        filter_residual: bool = False,
        affine_norms: bool = False,
        filter_num_groups: int = 1,
        spectral_ratio: float = 1.0,
        parallel_conv1x1: bool = False,
        per_track_layer_norm: bool = False,
    ):
        super().__init__()
        self.input_shape_loc = img_shape
        self.output_shape_loc = img_shape
        self.global_channels = global_channels
        self.local_channels = embed_dim - global_channels
        self._per_track_layer_norm = per_track_layer_norm

        def make_norm(n_channels: int) -> ConditionalLayerNorm:
            return ConditionalLayerNorm(
                n_channels,
                img_shape=img_shape,
                global_layer_norm=global_layer_norm,
                context_config=context_config,
                elementwise_affine=affine_norms,
            )

        if per_track_layer_norm:
            self.norm0_global: ConditionalLayerNorm | None = make_norm(global_channels)
            self.norm0_local: ConditionalLayerNorm | None = make_norm(
                self.local_channels
            )
            self.norm1_global: ConditionalLayerNorm | None = make_norm(global_channels)
            self.norm1_local: ConditionalLayerNorm | None = make_norm(
                self.local_channels
            )
            self.norm0: ConditionalLayerNorm | None = None
            self.norm1: ConditionalLayerNorm | None = None
        else:
            self.norm0 = make_norm(embed_dim)
            self.norm1 = make_norm(embed_dim)
            self.norm0_global = None
            self.norm0_local = None
            self.norm1_global = None
            self.norm1_local = None

        # Spectral filter is sized to the global width only.
        self.filter = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            global_channels,
            "linear",
            filter_residual=filter_residual,
            num_groups=filter_num_groups,
            spectral_ratio=spectral_ratio,
        )

        if parallel_conv1x1:
            self.conv1x1: nn.Conv2d | None = nn.Conv2d(
                embed_dim, embed_dim, kernel_size=1, bias=True
            )
        else:
            self.conv1x1 = None

        self.inner_skip = LoRAConv2d(embed_dim, embed_dim, 1, 1)
        self.act_layer = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if use_mlp:
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLP(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing=checkpointing,
            )

        self.outer_skip = nn.Identity()

    def _norm(
        self,
        joint: ConditionalLayerNorm | None,
        norm_global: ConditionalLayerNorm | None,
        norm_local: ConditionalLayerNorm | None,
        x: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """Apply a conditional norm over the local spatial extent.

        Mirrors the single-track block's zero-then-fill spatial slicing so that
        padded regions (distributed spatial parallelism) stay zero. Applies a
        single joint norm over the concatenation, or two per-track norms.
        """
        h, w = self.input_shape_loc
        x_norm = torch.zeros_like(x)
        active = x[..., :h, :w]
        if self._per_track_layer_norm:
            assert norm_global is not None and norm_local is not None
            g = norm_global(active[..., : self.global_channels, :, :], context)
            local_slice = active[..., self.global_channels :, :, :]
            normed = torch.cat([g, norm_local(local_slice, context)], dim=-3)
        else:
            assert joint is not None
            normed = joint(active, context)
        x_norm[..., :h, :w] = normed
        return x_norm

    def forward(
        self, x: torch.Tensor, context: Context, timer: Timer = NullTimer()
    ) -> torch.Tensor:
        x_norm = self._norm(self.norm0, self.norm0_global, self.norm0_local, x, context)

        # Spectral filter on the global slice only; local filter-stage output
        # is zero unless the parallel conv1x1 (option 2) supplies it.
        latents_norm = Latents.new_from_all(x_norm, self.global_channels)
        spectral_global, _ = self.filter(latents_norm.global_channels, timer=timer)
        filter_out = Latents.new_from_global(spectral_global, self.local_channels)
        if self.conv1x1 is not None:
            filter_out = filter_out + Latents.new_from_all(
                self.conv1x1(x_norm), self.global_channels
            )
        x = filter_out.all

        # Inner-skip residual comes from the full normed input, not the
        # filter's (global-only) returned residual.
        x = x + self.inner_skip(x_norm)
        x = self.act_layer(x)

        x_norm1 = self._norm(
            self.norm1, self.norm1_global, self.norm1_local, x, context
        )
        x = x_norm1
        if hasattr(self, "mlp"):
            x = self.mlp(x)
        x = self.drop_path(x)
        x = x + self.outer_skip(x_norm)
        return x


class TwoTrackSphericalFourierNeuralOperatorNet(nn.Module):
    """Two-track SFNO. See module docstring for the design invariant."""

    def __init__(
        self,
        params: TwoTrackSFNONetConfig,
        img_shape: tuple[int, int],
        get_pos_embed: Callable[[], nn.Parameter],
        trans_down: nn.Module,
        itrans_up: nn.Module,
        trans: nn.Module,
        itrans: nn.Module,
        global_in_channels: int,
        local_in_channels: int,
        global_out_channels: int,
        local_out_channels: int,
        context_config: ContextConfig,
    ):
        super().__init__()
        self._params = params
        self.img_shape = img_shape
        self.global_in_channels = global_in_channels
        self.local_in_channels = local_in_channels
        self.global_out_channels = global_out_channels
        self.local_out_channels = local_out_channels
        self.embed_dim = params.embed_dim
        self.global_embed_dim = params.global_embed_dim
        self.local_embed_dim = params.local_embed_dim
        self.num_layers = params.num_layers
        self.big_skip = params.big_skip
        self.checkpointing = params.checkpointing
        self._feed_global_to_local = params.feed_global_to_local
        self._has_local = params.local_embed_dim > 0

        self._spatial_h_slice, self._spatial_w_slice = (
            Distributed.get_instance().get_local_slices(self.img_shape)
        )

        self.trans_down = trans_down
        self.itrans_up = itrans_up
        self.trans = trans
        self.itrans = itrans

        if params.filter_residual:
            self.residual_filter_down: nn.Module = self.trans_down
            self.residual_filter_up: nn.Module = self.itrans_up
        else:
            self.residual_filter_down = nn.Identity()
            self.residual_filter_up = nn.Identity()

        if params.filter_output:
            self.filter_output_down: nn.Module = self.trans_down
            self.filter_output_up: nn.Module = self.itrans_up
        else:
            self.filter_output_down = nn.Identity()
            self.filter_output_up = nn.Identity()

        if params.activation_function not in ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unknown activation function {params.activation_function}"
            )
        act_layer = ACTIVATION_FUNCTIONS[params.activation_function]

        # Global encoder keeps the single-track name ``encoder``.
        self.encoder = _build_encoder(
            in_channels=global_in_channels,
            out_channels=self.global_embed_dim,
            encoder_layers=params.encoder_layers,
            act_layer=act_layer,
        )
        if self._has_local:
            local_in = local_in_channels + (
                global_in_channels if self._feed_global_to_local else 0
            )
            self.local_encoder: nn.Sequential | None = _build_encoder(
                in_channels=local_in,
                out_channels=self.local_embed_dim,
                encoder_layers=params.encoder_layers,
                act_layer=act_layer,
            )
        else:
            self.local_encoder = None

        self.pos_drop = (
            nn.Dropout(p=params.drop_rate) if params.drop_rate > 0.0 else nn.Identity()
        )
        dpr = [
            x.item() for x in torch.linspace(0, params.drop_path_rate, self.num_layers)
        ]

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            first_layer = i == 0
            last_layer = i == self.num_layers - 1
            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans
            self.blocks.append(
                TwoTrackFourierNeuralOperatorBlock(
                    forward_transform,
                    inverse_transform,
                    self.embed_dim,
                    self.global_embed_dim,
                    img_shape=self.img_shape,
                    context_config=context_config,
                    global_layer_norm=params.global_layer_norm,
                    mlp_ratio=params.mlp_ratio,
                    drop_rate=params.drop_rate,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    use_mlp=params.use_mlp,
                    checkpointing=params.checkpointing,
                    filter_residual=params.filter_residual,
                    affine_norms=params.affine_norms,
                    filter_num_groups=params.filter_num_groups,
                    spectral_ratio=params.spectral_ratio,
                    parallel_conv1x1=params.parallel_conv1x1,
                    per_track_layer_norm=params.per_track_layer_norm,
                )
            )

        # Global decoder keeps the single-track name ``decoder``.
        self.decoder = _build_decoder(
            in_channels=self.global_embed_dim + params.big_skip * global_in_channels,
            hidden_channels=self.global_embed_dim,
            out_channels=global_out_channels,
            encoder_layers=params.encoder_layers,
            act_layer=act_layer,
        )
        if self._has_local:
            self.local_decoder: nn.Sequential | None = _build_decoder(
                in_channels=self.local_embed_dim + params.big_skip * local_in_channels,
                hidden_channels=self.local_embed_dim,
                out_channels=local_out_channels,
                encoder_layers=params.encoder_layers,
                act_layer=act_layer,
            )
        else:
            self.local_decoder = None

        if params.pos_embed:
            self.pos_embed: nn.Parameter | None = get_pos_embed()
        else:
            self.pos_embed = None

        if params.normalize_big_skip:
            self.norm_big_skip: nn.Module = ConditionalLayerNorm(
                global_in_channels,
                img_shape=self.img_shape,
                global_layer_norm=params.global_layer_norm,
                context_config=context_config,
                elementwise_affine=params.affine_norms,
            )
            if self._has_local:
                self.local_norm_big_skip: nn.Module | None = ConditionalLayerNorm(
                    local_in_channels,
                    img_shape=self.img_shape,
                    global_layer_norm=params.global_layer_norm,
                    context_config=context_config,
                    elementwise_affine=params.affine_norms,
                )
            else:
                self.local_norm_big_skip = None
        else:
            self.norm_big_skip = NoLayerNorm()
            self.local_norm_big_skip = NoLayerNorm() if self._has_local else None

    @torch.jit.ignore
    def no_weight_decay(self):  # pragma: no cover
        return {"pos_embed"}

    def _forward_features(self, x: torch.Tensor, context: Context) -> torch.Tensor:
        for blk in self.blocks:
            if self.checkpointing >= 3:
                x = checkpoint(blk, x, context)
            else:
                x = blk(x, context)
        return x

    def forward(self, x: torch.Tensor, context: Context) -> torch.Tensor:
        global_in = x[..., : self.global_in_channels, :, :]
        local_in = x[..., self.global_in_channels :, :, :]

        if self.big_skip:
            global_residual = self.residual_filter_up(
                self.residual_filter_down(global_in)
            )
            global_residual = self.norm_big_skip(global_residual, context=context)
            if self._has_local:
                assert self.local_norm_big_skip is not None
                local_residual = self.local_norm_big_skip(local_in, context=context)

        if self.checkpointing >= 1:
            g = checkpoint(self.encoder, global_in)
        else:
            g = self.encoder(global_in)
        if self._has_local:
            assert self.local_encoder is not None
            local_encoder_in = (
                torch.cat([global_in, local_in], dim=-3)
                if self._feed_global_to_local
                else local_in
            )
            latent = torch.cat([g, self.local_encoder(local_encoder_in)], dim=-3)
        else:
            latent = g

        if self.pos_embed is not None:
            latent = (
                latent
                + self.pos_embed[..., self._spatial_h_slice, self._spatial_w_slice]
            )
        latent = self.pos_drop(latent)

        latent = self._forward_features(latent, context)

        global_latent = latent[..., : self.global_embed_dim, :, :]
        local_latent = latent[..., self.global_embed_dim :, :, :]

        if self.big_skip:
            global_dec_in = torch.cat([global_latent, global_residual], dim=-3)
        else:
            global_dec_in = global_latent
        if self.checkpointing >= 1:
            global_out = checkpoint(self.decoder, global_dec_in)
        else:
            global_out = self.decoder(global_dec_in)
        global_out = self.filter_output_up(self.filter_output_down(global_out))

        if not self._has_local:
            return global_out

        assert self.local_decoder is not None
        if self.big_skip:
            local_dec_in = torch.cat([local_latent, local_residual], dim=-3)
        else:
            local_dec_in = local_latent
        local_out = self.local_decoder(local_dec_in)
        return torch.cat([global_out, local_out], dim=-3)


def _build_encoder(
    in_channels: int,
    out_channels: int,
    encoder_layers: int,
    act_layer,
) -> nn.Sequential:
    hidden_dim = out_channels
    current_dim = in_channels
    modules: list[nn.Module] = []
    for _ in range(encoder_layers):
        modules.append(LoRAConv2d(current_dim, hidden_dim, 1, bias=True))
        modules.append(act_layer())
        current_dim = hidden_dim
    modules.append(LoRAConv2d(current_dim, out_channels, 1, bias=False))
    return nn.Sequential(*modules)


def _build_decoder(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    encoder_layers: int,
    act_layer,
) -> nn.Sequential:
    current_dim = in_channels
    modules: list[nn.Module] = []
    for _ in range(encoder_layers):
        modules.append(LoRAConv2d(current_dim, hidden_channels, 1, bias=True))
        modules.append(act_layer())
        current_dim = hidden_channels
    modules.append(LoRAConv2d(current_dim, out_channels, 1, bias=False))
    return nn.Sequential(*modules)


def get_lat_lon_two_track_sfnonet(
    params: TwoTrackSFNONetConfig,
    global_in_channels: int,
    local_in_channels: int,
    global_out_channels: int,
    local_out_channels: int,
    img_shape: tuple[int, int],
    data_grid: str = "equiangular",
    context_config: ContextConfig = ContextConfig(
        embed_dim_scalar=0,
        embed_dim_noise=0,
        embed_dim_labels=0,
        embed_dim_pos=0,
    ),
) -> TwoTrackSphericalFourierNeuralOperatorNet:
    h, w = img_shape
    modes_lat = int(h * params.hard_thresholding_fraction)
    modes_lon = int((w // 2 + 1) * params.hard_thresholding_fraction)

    dist = Distributed.get_instance()
    trans_down = dist.get_sht(
        *img_shape, lmax=modes_lat, mmax=modes_lon, grid=data_grid
    )
    itrans_up = dist.get_isht(
        *img_shape, lmax=modes_lat, mmax=modes_lon, grid=data_grid
    )
    trans = dist.get_sht(
        *img_shape, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
    )
    itrans = dist.get_isht(h, w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss")

    def get_pos_embed() -> nn.Parameter:
        pos_embed = nn.Parameter(torch.zeros(1, params.embed_dim, h, w))
        pos_embed.is_shared_mp = ["matmul"]  # type: ignore[attr-defined]
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    return TwoTrackSphericalFourierNeuralOperatorNet(
        params,
        img_shape=img_shape,
        get_pos_embed=get_pos_embed,
        trans_down=trans_down,
        itrans_up=itrans_up,
        trans=trans,
        itrans=itrans,
        global_in_channels=global_in_channels,
        local_in_channels=local_in_channels,
        global_out_channels=global_out_channels,
        local_out_channels=local_out_channels,
        context_config=context_config,
    )

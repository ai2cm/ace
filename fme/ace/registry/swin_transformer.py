import dataclasses
import math

import torch
from torch import nn

from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.core.dataset_info import DatasetInfo, MissingDatasetInfo
from fme.core.models.boundary_padding import TensorPaddingConfig
from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.models.swin_transformer import SwinTransformerNet


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedder (DiT-style).

    Maps a scalar per batch member to a learned hidden-size vector via
    sinusoidal frequency features followed by a two-layer MLP.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freq_emb = self._sinusoidal_embedding(t, self.frequency_embedding_size)
        return self.mlp(freq_emb)


class _ContextWrappedModule(nn.Module):
    """Wraps a module that takes (x, context: Context) to accept (x, labels=None).

    This adapts the SwinTransformerNet forward signature to the interface
    expected by the Module registry wrapper. Only ``context.labels`` is
    populated; scalar/positional/noise conditioning is left unset.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        context = Context(
            embedding_scalar=None,
            embedding_pos=None,
            labels=labels,
            noise=None,
        )
        return self.module(x, context)


class _TimeConditionedContextWrappedModule(nn.Module):
    """Wraps SwinTransformerNet to accept (x, labels, forward_time).

    Embeds the per-step calendar month and hour-of-day via two separate
    ``TimestepEmbedder`` modules (DiT-style sinusoidal + MLP), sums the
    embeddings, and passes the result as ``context.embedding_scalar`` for
    AdaLN conditioning.  When ``forward_time`` is None a zero embedding is
    used so the model degrades gracefully.

    Args:
        module: The underlying ``SwinTransformerNet``.
        embed_dim_scalar: Must match the ``embed_dim_scalar`` in the net's
            ``ContextConfig``.
        frequency_embedding_size: Sinusoidal feature dimension fed into the
            embedder MLPs.
    """

    def __init__(
        self,
        module: nn.Module,
        embed_dim_scalar: int,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        self.module = module
        self.embed_dim_scalar = embed_dim_scalar
        self.month_embedder = TimestepEmbedder(
            embed_dim_scalar, frequency_embedding_size
        )
        self.hour_embedder = TimestepEmbedder(
            embed_dim_scalar, frequency_embedding_size
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        forward_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if forward_time is not None:
            month = forward_time[:, 0]
            hour = forward_time[:, 1]
            embedding_scalar: torch.Tensor = self.month_embedder(
                month
            ) + self.hour_embedder(hour)
        else:
            embedding_scalar = torch.zeros(
                x.shape[0], self.embed_dim_scalar, device=x.device, dtype=x.dtype
            )
        context = Context(
            embedding_scalar=embedding_scalar,
            embedding_pos=None,
            labels=labels,
            noise=None,
        )
        return self.module(x, context)


@ModuleSelector.register("SwinTransformer")
@dataclasses.dataclass
class SwinTransformerBuilder(ModuleConfig):
    """Configuration for the 2D Swin U-Net architecture.

    A 2D adaptation of ArchesWeather's 3D Swin U-Net to ACE's
    ``(B, C, H, W)`` interface, with U-Net encoder/decoder, shifted-window
    attention, column-wise interaction, and optional AdaLN conditioning.

    Attributes:
        embed_dim: Channel dimension of the first/last U-Net stage.
        depth_multiplier: Scales the per-stage depths ``[2, 6, 6, 2]``.
        num_heads: Attention heads for each of the four stages (length-4 list).
        window_size: ``[ws_h, ws_w]`` attention window.
        mlp_ratio: Hidden-dim multiplier for block MLPs.
        drop_path_rate: Maximum stochastic-depth rate.
        use_skip: Whether to concatenate the layer-1 skip into the decoder.
        mlp_layer: ``"mlp"`` or ``"swiglu"``.
        embed_dim_scalar: Scalar conditioning dimension.  When > 0, AdaLN
            conditioning is enabled but ``context.embedding_scalar`` must be
            populated by the caller.  Use ``TimeConditionedSwinTransformer``
            instead for automatic month/hour conditioning; leave at 0 here.
        embed_dim_labels: Label conditioning dimension. When 0 and the dataset
            has labels and the selector is conditional, it defaults to the
            number of labels, which is the dimension the registry feeds the
            model.
        use_cpb_scaling: When True (default), requires 1D latitude coordinates
            and applies cos-lat scaling to CPB longitude offsets. Set to False
            to use plain log-spaced CPB offsets (Swin V2 style) without a
            latitude requirement.
    """

    embed_dim: int = 96
    depth_multiplier: int = 1
    num_heads: list[int] = dataclasses.field(default_factory=lambda: [3, 6, 6, 3])
    window_size: list[int] = dataclasses.field(default_factory=lambda: [4, 8])
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.2
    use_skip: bool = True
    mlp_layer: str = "mlp"
    embed_dim_scalar: int = 0
    embed_dim_labels: int = 0
    cpb_hidden_dim: int = 64
    padding_conf: TensorPaddingConfig | None = None
    use_cpb_scaling: bool = True

    def __post_init__(self):
        if isinstance(self.padding_conf, dict):
            self.padding_conf = TensorPaddingConfig(**self.padding_conf)

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        return self._build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            dataset_info=dataset_info,
            enable_label_conditioning=len(dataset_info.all_labels) > 0
            or self.embed_dim_labels > 0,
        )

    def build_for_selector(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
        conditional: bool,
    ) -> nn.Module:
        if self.embed_dim_labels > 0 and not conditional:
            raise ValueError(
                "SwinTransformer label conditioning requires "
                "ModuleSelector(conditional=True)"
            )
        return self._build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            dataset_info=dataset_info,
            enable_label_conditioning=conditional,
        )

    def _build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
        enable_label_conditioning: bool,
    ) -> nn.Module:
        n_labels = len(dataset_info.all_labels)
        embed_dim_labels = self.embed_dim_labels
        if not enable_label_conditioning:
            embed_dim_labels = 0
        elif n_labels > 0 and embed_dim_labels == 0:
            # The registry feeds one-hot labels of dimension n_labels.
            embed_dim_labels = n_labels
        context_config = ContextConfig(
            embed_dim_scalar=self.embed_dim_scalar,
            embed_dim_labels=embed_dim_labels,
            embed_dim_noise=0,
            embed_dim_pos=0,
        )
        if self.use_cpb_scaling:
            try:
                lat_coords = dataset_info.horizontal_coordinates.lat_1d
            except MissingDatasetInfo:
                raise ValueError(
                    "SwinTransformer requires 1D latitude coordinates for cos-lat CPB "
                    "scaling, but the dataset provides none. Non-lat-lon grids such as "
                    "HEALPix are not supported. Set use_cpb_scaling=False to disable "
                    "this requirement."
                ) from None
            if lat_coords is None:
                raise ValueError(
                    "SwinTransformer requires 1D latitude coordinates for cos-lat CPB "
                    "scaling, but this coordinate type returns None for lat_1d. "
                    "Set use_cpb_scaling=False to disable this requirement."
                )
        else:
            lat_coords = None
        net = SwinTransformerNet(
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=dataset_info.img_shape,
            embed_dim=self.embed_dim,
            depth_multiplier=self.depth_multiplier,
            num_heads=tuple(self.num_heads),
            window_size=(self.window_size[0], self.window_size[1]),
            mlp_ratio=self.mlp_ratio,
            drop_path_rate=self.drop_path_rate,
            use_skip=self.use_skip,
            context_config=context_config,
            mlp_layer=self.mlp_layer,
            cpb_hidden_dim=self.cpb_hidden_dim,
            lat_coords=lat_coords,
            padding_conf=dataclasses.asdict(self.padding_conf)
            if self.padding_conf is not None
            else None,
        )
        return _ContextWrappedModule(net)


@ModuleSelector.register("NoiseConditionedSwinTransformer")
@dataclasses.dataclass
class NoiseConditionedSwinTransformerBuilder(ModuleConfig):
    """Configuration for a noise-conditioned 2D Swin U-Net.

    Wraps ``SwinTransformerNet`` in ``NoiseConditionedModel`` so that each
    forward pass samples a fresh Gaussian noise field and injects it through
    per-block ``ConditionalLayerNorm`` layers, giving genuinely stochastic
    ensemble members when called multiple times on the same input.

    Attributes:
        embed_dim: Channel dimension of the first/last U-Net stage.
        depth_multiplier: Scales the per-stage depths ``[2, 6, 6, 2]``.
        num_heads: Attention heads for each of the four stages (length-4 list).
        window_size: ``[ws_h, ws_w]`` attention window.
        mlp_ratio: Hidden-dim multiplier for block MLPs.
        drop_path_rate: Maximum stochastic-depth rate.
        use_skip: Whether to concatenate the layer-1 skip into the decoder.
        mlp_layer: ``"mlp"`` or ``"swiglu"``.
        noise_embed_dim: Dimension of the Gaussian noise field injected through
            each block's ``ConditionalLayerNorm``.
        label_embed_dim: Dimension of the learned label embedding space.
            When > 0, a shared ``Linear(n_labels, label_embed_dim)`` layer maps
            one-hot labels before downstream CLN conditioning.
            When 0 (default), one-hot labels are used directly.
            Label conditioning is enabled only when the selector is conditional,
            or when building directly with dataset labels.
        use_cpb_scaling: When True (default), requires 1D latitude coordinates
            and applies cos-lat scaling to CPB longitude offsets. Set to False
            to use plain log-spaced CPB offsets (Swin V2 style) without a
            latitude requirement.
    """

    embed_dim: int = 96
    depth_multiplier: int = 1
    num_heads: list[int] = dataclasses.field(default_factory=lambda: [3, 6, 6, 3])
    window_size: list[int] = dataclasses.field(default_factory=lambda: [4, 8])
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.2
    use_skip: bool = True
    mlp_layer: str = "mlp"
    noise_embed_dim: int = 256
    label_embed_dim: int = 0
    cpb_hidden_dim: int = 64
    padding_conf: TensorPaddingConfig | None = None
    use_cpb_scaling: bool = True

    def __post_init__(self):
        if isinstance(self.padding_conf, dict):
            self.padding_conf = TensorPaddingConfig(**self.padding_conf)

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        return self._build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            dataset_info=dataset_info,
            enable_label_conditioning=len(dataset_info.all_labels) > 0
            or self.label_embed_dim > 0,
        )

    def build_for_selector(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
        conditional: bool,
    ) -> nn.Module:
        if self.label_embed_dim > 0 and not conditional:
            raise ValueError(
                "NoiseConditionedSwinTransformer label conditioning requires "
                "ModuleSelector(conditional=True)"
            )
        return self._build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            dataset_info=dataset_info,
            enable_label_conditioning=conditional,
        )

    def _build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
        enable_label_conditioning: bool,
    ) -> nn.Module:
        n_dataset_labels = len(dataset_info.all_labels)
        n_labels = n_dataset_labels if enable_label_conditioning else 0
        label_embed_dim = self.label_embed_dim if enable_label_conditioning else 0
        if label_embed_dim > 0:
            effective_label_dim = self.label_embed_dim
        else:
            effective_label_dim = n_labels
        context_config = ContextConfig(
            embed_dim_scalar=0,
            embed_dim_labels=effective_label_dim,
            embed_dim_noise=self.noise_embed_dim,
            embed_dim_pos=0,
        )
        if self.use_cpb_scaling:
            try:
                lat_coords = dataset_info.horizontal_coordinates.lat_1d
            except MissingDatasetInfo:
                raise ValueError(
                    "SwinTransformer requires 1D latitude coordinates for cos-lat CPB "
                    "scaling, but the dataset provides none. Non-lat-lon grids such as "
                    "HEALPix are not supported. Set use_cpb_scaling=False to disable "
                    "this requirement."
                ) from None
            if lat_coords is None:
                raise ValueError(
                    "SwinTransformer requires 1D latitude coordinates for cos-lat CPB "
                    "scaling, but this coordinate type returns None for lat_1d. "
                    "Set use_cpb_scaling=False to disable this requirement."
                )
        else:
            lat_coords = None
        net = SwinTransformerNet(
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=dataset_info.img_shape,
            embed_dim=self.embed_dim,
            depth_multiplier=self.depth_multiplier,
            num_heads=tuple(self.num_heads),
            window_size=(self.window_size[0], self.window_size[1]),
            mlp_ratio=self.mlp_ratio,
            drop_path_rate=self.drop_path_rate,
            use_skip=self.use_skip,
            context_config=context_config,
            mlp_layer=self.mlp_layer,
            conditioning="cln",
            cpb_hidden_dim=self.cpb_hidden_dim,
            lat_coords=lat_coords,
            padding_conf=dataclasses.asdict(self.padding_conf)
            if self.padding_conf is not None
            else None,
        )
        return NoiseConditionedModel(
            net,
            img_shape=dataset_info.img_shape,
            embed_dim_noise=self.noise_embed_dim,
            embed_dim_pos=0,
            n_labels=n_labels,
            label_embed_dim=label_embed_dim,
            inverse_sht=None,
        )


@ModuleSelector.register("TimeConditionedSwinTransformer")
@dataclasses.dataclass
class TimeConditionedSwinTransformerBuilder(ModuleConfig):
    """Swin U-Net conditioned on calendar month and hour-of-day via AdaLN.

    Wraps ``SwinTransformerNet`` in ``_TimeConditionedContextWrappedModule``
    which embeds the per-step ``(month, hour)`` pair through two
    ``TimestepEmbedder`` modules (DiT-style sinusoidal features + MLP) and
    injects the summed embedding as ``context.embedding_scalar`` for
    per-block AdaLN modulation.  The timestamp is extracted automatically
    from the training loop and threaded through ``StepArgs.forward_time``.

    Attributes:
        embed_dim: Channel dimension of the first/last U-Net stage.
        depth_multiplier: Scales the per-stage depths ``[2, 6, 6, 2]``.
        num_heads: Attention heads for each of the four stages (length-4 list).
        window_size: ``[ws_h, ws_w]`` attention window.
        mlp_ratio: Hidden-dim multiplier for block MLPs.
        drop_path_rate: Maximum stochastic-depth rate.
        use_skip: Whether to concatenate the layer-1 skip into the decoder.
        mlp_layer: ``"mlp"`` or ``"swiglu"``.
        embed_dim_scalar: Dimension of the time conditioning embedding fed
            to every block's AdaLN.  Must be > 0.
        embed_dim_labels: Label conditioning dimension (0 = disabled).
        frequency_embedding_size: Sinusoidal feature size inside each
            ``TimestepEmbedder`` MLP.
        use_cpb_scaling: When True (default), requires 1D latitude coordinates.
    """

    embed_dim: int = 96
    depth_multiplier: int = 1
    num_heads: list[int] = dataclasses.field(default_factory=lambda: [3, 6, 6, 3])
    window_size: list[int] = dataclasses.field(default_factory=lambda: [4, 8])
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.2
    use_skip: bool = True
    mlp_layer: str = "mlp"
    embed_dim_scalar: int = 256
    embed_dim_labels: int = 0
    cpb_hidden_dim: int = 64
    frequency_embedding_size: int = 256
    padding_conf: TensorPaddingConfig | None = None
    use_cpb_scaling: bool = True

    def __post_init__(self):
        if self.embed_dim_scalar <= 0:
            raise ValueError("embed_dim_scalar must be > 0 for time conditioning")
        if isinstance(self.padding_conf, dict):
            self.padding_conf = TensorPaddingConfig(**self.padding_conf)

    @property
    def has_time_conditioning(self) -> bool:
        return True

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        return self._build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            dataset_info=dataset_info,
            enable_label_conditioning=len(dataset_info.all_labels) > 0
            or self.embed_dim_labels > 0,
        )

    def build_for_selector(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
        conditional: bool,
    ) -> nn.Module:
        if self.embed_dim_labels > 0 and not conditional:
            raise ValueError(
                "TimeConditionedSwinTransformer label conditioning requires "
                "ModuleSelector(conditional=True)"
            )
        return self._build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            dataset_info=dataset_info,
            enable_label_conditioning=conditional,
        )

    def _build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
        enable_label_conditioning: bool,
    ) -> nn.Module:
        n_labels = len(dataset_info.all_labels)
        embed_dim_labels = self.embed_dim_labels
        if not enable_label_conditioning:
            embed_dim_labels = 0
        elif n_labels > 0 and embed_dim_labels == 0:
            embed_dim_labels = n_labels
        context_config = ContextConfig(
            embed_dim_scalar=self.embed_dim_scalar,
            embed_dim_labels=embed_dim_labels,
            embed_dim_noise=0,
            embed_dim_pos=0,
        )
        if self.use_cpb_scaling:
            try:
                lat_coords = dataset_info.horizontal_coordinates.lat_1d
            except MissingDatasetInfo:
                raise ValueError(
                    "SwinTransformer requires 1D latitude coordinates for cos-lat CPB "
                    "scaling, but the dataset provides none. Non-lat-lon grids such as "
                    "HEALPix are not supported. Set use_cpb_scaling=False to disable "
                    "this requirement."
                ) from None
            if lat_coords is None:
                raise ValueError(
                    "SwinTransformer requires 1D latitude coordinates for cos-lat CPB "
                    "scaling, but this coordinate type returns None for lat_1d. "
                    "Set use_cpb_scaling=False to disable this requirement."
                )
        else:
            lat_coords = None
        net = SwinTransformerNet(
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=dataset_info.img_shape,
            embed_dim=self.embed_dim,
            depth_multiplier=self.depth_multiplier,
            num_heads=tuple(self.num_heads),
            window_size=(self.window_size[0], self.window_size[1]),
            mlp_ratio=self.mlp_ratio,
            drop_path_rate=self.drop_path_rate,
            use_skip=self.use_skip,
            context_config=context_config,
            mlp_layer=self.mlp_layer,
            cpb_hidden_dim=self.cpb_hidden_dim,
            lat_coords=lat_coords,
            padding_conf=dataclasses.asdict(self.padding_conf)
            if self.padding_conf is not None
            else None,
        )
        return _TimeConditionedContextWrappedModule(
            net,
            embed_dim_scalar=self.embed_dim_scalar,
            frequency_embedding_size=self.frequency_embedding_size,
        )

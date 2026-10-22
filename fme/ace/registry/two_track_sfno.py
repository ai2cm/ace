import dataclasses
from typing import Literal

import torch

from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.core.dataset_info import DatasetInfo
from fme.core.models.conditional_sfno.layers import ContextConfig
from fme.core.models.conditional_sfno.two_track_sfnonet import (
    TwoTrackSFNONetConfig,
    get_lat_lon_two_track_sfnonet,
)


@ModuleSelector.register("TwoTrackSFNO")
@dataclasses.dataclass
class TwoTrackSFNOBuilder(ModuleConfig):
    """Builder for a two-track (global / local latent) noise-conditioned SFNO.

    The network splits its latent into a global track (which passes through the
    spherical harmonic transform) and a local, pointwise-only track. Unlike the
    single-track builders, the per-track *channel* counts are not known until
    the owning two-track step assigns variables to tracks, so this builder is
    driven by ``build_two_track`` (called directly by the step) rather than the
    standard ``build``. It is still registered in the module registry so the
    config loads from YAML like any other builder.

    Attributes:
        embed_dim: Total latent width (global + local).
        local_embed_dim: Width of the local latent track. 0 recovers the
            single-track network exactly, but only on the ``legendre-gauss``
            grid (``local_embed_dim=0`` with ``data_grid='equiangular'`` is
            rejected in ``__post_init__``; see the note there). Must be 0 if
            and only if the local track carries no input or output channels
            (validated in ``build_two_track``).
        noise_embed_dim: Dimension of the noise conditioning channels. 0
            disables noise conditioning.
        noise_type: "gaussian" or "isotropic" noise.
        context_pos_embed_dim: Dimension of the context positional embedding.
        label_embed_dim: Dimension of a learned label embedding (0 = one-hot).
        data_grid: Grid type for the spherical harmonic transforms.
        spectral_ratio: Fraction of the *global* latent width that
            participates in the spectral filter (the filter is sized to the
            global track, not embed_dim). Double it when the global width is
            halved to keep the actual filter (channels-per-group x groups)
            equal to a single-track baseline's.
        feed_global_to_local: OPTION 1 (default off).
        parallel_conv1x1: OPTION 2 (default off).
        per_track_layer_norm: OPTION 3 (default off).

    The remaining attributes mirror the single-track SFNO config.
    """

    embed_dim: int = 256
    local_embed_dim: int = 256
    filter_type: Literal["linear"] = "linear"
    noise_embed_dim: int = 32
    noise_type: Literal["isotropic", "gaussian"] = "gaussian"
    context_pos_embed_dim: int = 0
    label_embed_dim: int = 0
    global_layer_norm: bool = False
    num_layers: int = 12
    use_mlp: bool = True
    mlp_ratio: float = 2.0
    activation_function: str = "gelu"
    encoder_layers: int = 1
    pos_embed: bool = True
    big_skip: bool = True
    checkpointing: int = 0
    data_grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"
    hard_thresholding_fraction: float = 1.0
    filter_residual: bool = False
    filter_output: bool = False
    normalize_big_skip: bool = False
    affine_norms: bool = False
    filter_num_groups: int = 1
    spectral_ratio: float = 1.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    feed_global_to_local: bool = False
    parallel_conv1x1: bool = False
    per_track_layer_norm: bool = False

    def __post_init__(self):
        if self.context_pos_embed_dim > 0 and self.pos_embed:
            raise ValueError(
                "context_pos_embed_dim and pos_embed should not both be set"
            )
        if self.local_embed_dim == 0 and self.data_grid == "equiangular":
            # local_embed_dim=0 is the single-track-equivalent configuration,
            # used only to warm-start from an old single-track checkpoint. That
            # byte-for-byte equivalence holds on legendre-gauss but NOT on
            # equiangular: there the first/last blocks round-trip their spectral
            # residual, so the two-track block's (full-normed-input) skip differs
            # from the single-track block's (filter-residual) skip and the loaded
            # checkpoint silently produces wrong output. Fail loudly instead.
            raise ValueError(
                "local_embed_dim=0 (single-track-equivalent) is not backwards "
                "compatible with data_grid='equiangular': an old single-track "
                "checkpoint loads without error but does not reproduce its output "
                "on this grid. Warm-start from a single-track checkpoint only with "
                "data_grid='legendre-gauss', or use a nonzero local_embed_dim."
            )
        # Validate the network config (embed_dim / local_embed_dim / groups).
        self._net_config()

    def _net_config(self) -> TwoTrackSFNONetConfig:
        return TwoTrackSFNONetConfig(
            embed_dim=self.embed_dim,
            local_embed_dim=self.local_embed_dim,
            filter_type=self.filter_type,
            global_layer_norm=self.global_layer_norm,
            num_layers=self.num_layers,
            use_mlp=self.use_mlp,
            mlp_ratio=self.mlp_ratio,
            activation_function=self.activation_function,
            encoder_layers=self.encoder_layers,
            pos_embed=self.pos_embed,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            hard_thresholding_fraction=self.hard_thresholding_fraction,
            big_skip=self.big_skip,
            checkpointing=self.checkpointing,
            filter_num_groups=self.filter_num_groups,
            filter_residual=self.filter_residual,
            filter_output=self.filter_output,
            normalize_big_skip=self.normalize_big_skip,
            affine_norms=self.affine_norms,
            spectral_ratio=self.spectral_ratio,
            feed_global_to_local=self.feed_global_to_local,
            parallel_conv1x1=self.parallel_conv1x1,
            per_track_layer_norm=self.per_track_layer_norm,
        )

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> torch.nn.Module:
        raise NotImplementedError(
            "TwoTrackSFNOBuilder cannot be built through the single-tensor "
            "build interface because it needs the per-track channel counts. "
            "It is built by the two-track step via build_two_track."
        )

    def build_two_track(
        self,
        global_in_channels: int,
        local_in_channels: int,
        global_out_channels: int,
        local_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> torch.nn.Module:
        # local_embed_dim must be nonzero iff the local track carries channels.
        has_local_channels = local_in_channels > 0 or local_out_channels > 0
        if has_local_channels and self.local_embed_dim == 0:
            raise ValueError(
                "local_embed_dim must be > 0 when the local track carries any "
                f"input or output channels (got local_in={local_in_channels}, "
                f"local_out={local_out_channels}, local_embed_dim=0)."
            )
        if not has_local_channels and self.local_embed_dim > 0:
            raise ValueError(
                "local_embed_dim must be 0 when the local track carries no input "
                f"or output channels (got local_embed_dim={self.local_embed_dim}); "
                "with no local track, use the single-track step instead."
            )
        n_labels = len(dataset_info.all_labels)
        effective_label_dim = (
            self.label_embed_dim if self.label_embed_dim > 0 else n_labels
        )
        net = get_lat_lon_two_track_sfnonet(
            params=self._net_config(),
            global_in_channels=global_in_channels,
            local_in_channels=local_in_channels,
            global_out_channels=global_out_channels,
            local_out_channels=local_out_channels,
            img_shape=dataset_info.img_shape,
            data_grid=self.data_grid,
            context_config=ContextConfig(
                embed_dim_scalar=0,
                embed_dim_pos=self.context_pos_embed_dim,
                embed_dim_noise=self.noise_embed_dim,
                embed_dim_labels=effective_label_dim,
            ),
        )
        if self.noise_type == "isotropic":
            inverse_sht = net.itrans_up
            lmax = inverse_sht.lmax
            mmax = inverse_sht.mmax
        else:
            inverse_sht = None
            lmax = 0
            mmax = 0
        return NoiseConditionedModel(
            net,
            embed_dim_noise=self.noise_embed_dim,
            embed_dim_pos=self.context_pos_embed_dim,
            n_labels=n_labels,
            label_embed_dim=self.label_embed_dim,
            img_shape=dataset_info.img_shape,
            inverse_sht=inverse_sht,
            lmax=lmax,
            mmax=mmax,
        )

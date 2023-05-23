import dataclasses
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from apex.normalization import FusedLayerNorm

from torch.utils.checkpoint import checkpoint

# helpers
from networks.geometric_v1.layers import trunc_normal_, DropPath, MLP
from networks.geometric_v1.layers import SpectralAttentionS2, SpectralConvS2
from networks.geometric_v1.layers import SpectralAttention2d, SpectralConv2d

import torch_harmonics as harmonics
# to fake the sht module with ffts
from networks.geometric_v1.layers import RealFFT2, InverseRealFFT2

from networks.geometric_v1.contractions import *
# from networks.geometric_v1 import activations
from networks.geometric_v1.activations import *

class SpectralFilterLayer(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type = 'linear',
        sparsity_threshold = 0.0,
        use_complex_kernels = True,
        hidden_size_factor = 2,
        compression = None,
        rank = 128,
        complex_network = True,
        complex_activation = 'real',
        spectral_layers = 1,
        drop_rate = 0.):
        super(SpectralFilterLayer, self).__init__() 

        if filter_type == 'non-linear' and isinstance(forward_transform, harmonics.RealSHT):
            self.filter = SpectralAttentionS2(forward_transform,
                                              inverse_transform,
                                              embed_dim,
                                              sparsity_threshold,
                                              use_complex_network = complex_network,
                                              use_complex_kernels = use_complex_kernels,
                                              hidden_size_factor = hidden_size_factor,
                                              complex_activation = complex_activation,
                                              spectral_layers = spectral_layers,
                                              drop_rate = drop_rate,
                                              bias = False)
        elif filter_type == 'non-linear' and isinstance(forward_transform, RealFFT2):
            self.filter = SpectralAttention2d(forward_transform,
                                              inverse_transform,
                                              embed_dim,
                                              sparsity_threshold,
                                              use_complex_kernels = use_complex_kernels,
                                              hidden_size_factor = hidden_size_factor,
                                              complex_activation = complex_activation,
                                              spectral_layers = spectral_layers,
                                              drop_rate = drop_rate,
                                              bias = False)

        elif filter_type == 'linear' and isinstance(forward_transform, RealSHT):
            self.filter = SpectralConvS2(forward_transform,
                                         inverse_transform,
                                         embed_dim,
                                         sparsity_threshold,
                                         use_complex_kernels = use_complex_kernels,
                                         compression = compression,
                                         rank = rank,
                                         bias = False)
                                              
        elif filter_type == 'linear' and isinstance(forward_transform, RealFFT2):
            self.filter = SpectralConv2d(forward_transform,
                                         inverse_transform,
                                         embed_dim,
                                         sparsity_threshold,
                                         use_complex_kernels = use_complex_kernels,
                                         compression = compression,
                                         rank = rank,
                                         bias = False)
                                              

        else:
            raise(NotImplementedError)

    def forward(self, x):
        return self.filter(x)


class FourierNeuralOperatorBlock(nn.Module):
    def __init__(
            self,
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type = 'linear',
            mlp_ratio = 2.,
            drop_rate = 0.,
            drop_path = 0.,
            act_layer = nn.GELU,
            norm_layer = (nn.LayerNorm, nn.LayerNorm),
            # num_blocks = 8,
            sparsity_threshold = 0.0,
            use_complex_kernels = True,
            compression = None,
            rank = 128,
            inner_skip = 'linear',
            outer_skip = None, # None, nn.linear or nn.Identity
            concat_skip = False,
            mlp_mode = "none",
            complex_network = True,
            complex_activation = 'real',
            spectral_layers = 1,
            checkpointing = False):
        super(FourierNeuralOperatorBlock, self).__init__()
        
        # norm layer
        self.norm0 = norm_layer[0]() #((h,w))

        # convolution layer
        self.filter_layer = SpectralFilterLayer(forward_transform,
                                                inverse_transform,
                                                embed_dim,
                                                filter_type,
                                                sparsity_threshold,
                                                use_complex_kernels = use_complex_kernels,
                                                hidden_size_factor = mlp_ratio,
                                                compression = compression,
                                                rank = rank,
                                                complex_network = complex_network,
                                                complex_activation = complex_activation,
                                                spectral_layers = spectral_layers,
                                                drop_rate = drop_rate)

        if inner_skip == 'linear':
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif inner_skip == 'identity':
            self.inner_skip = nn.Identity()

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2*embed_dim, embed_dim, 1, bias=False)

        if filter_type == 'linear':
            self.act_layer = act_layer()
        
        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # norm layer
        self.norm1 = norm_layer[1]() #((h,w))

        if mlp_mode != "none":
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLP(in_features = embed_dim,
                            hidden_features = mlp_hidden_dim,
                            act_layer = act_layer,
                            drop_rate = drop_rate,
                            checkpointing = checkpointing)

        if outer_skip == 'linear':
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif outer_skip == 'identity':
            self.outer_skip = nn.Identity()

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = nn.Conv2d(2*embed_dim, embed_dim, 1, bias=False)

    def forward(self, x):
        
        residual = x
        
        x = self.norm0(x)
        x = self.filter_layer(x).contiguous()

        if hasattr(self, 'inner_skip'):
            if self.concat_skip:
                x = torch.cat((x, self.inner_skip(residual)), dim=1)
                x = self.inner_skip_conv(x)
            else:
                x = x + self.inner_skip(residual)

        if hasattr(self, 'act_layer'):
            x = self.act_layer(x)

        x = self.norm1(x)

        if hasattr(self, 'mlp'):
            x = self.mlp(x)

        x = self.drop_path(x)

        if hasattr(self, 'outer_skip'):
            if self.concat_skip:
                x = torch.cat((x, self.outer_skip(residual)), dim=1)
                x = self.outer_skip_conv(x)
            else:
                x = x + self.outer_skip(residual)

        return x

    # @torch.jit.ignore
    # def checkpoint_forward(self, x):
    #     return checkpoint(self._forward, x)

    # def forward(self, x):
    #     if self.checkpointing:
    #         return self.checkpoint_forward(x)
    #     else:
    #         return self._forward(x)

@dataclasses.dataclass
class FourierNeuralOperatorBuilder:
    spectral_transform: str = 'sht'
    filter_type: str = 'non-linear'
    scale_factor: int = 16
    embed_dim: int = 256
    num_layers: int = 12
    num_blocks: int = 16
    hard_thresholding_fraction: float = 1.0
    normalization_layer: str = 'instance_norm'
    mlp_mode: str = "none"
    big_skip: bool = True
    compression: str = None
    rank: int = 128
    complex_network: bool = True
    complex_activation: str = 'real'
    spectral_layers: int = 1
    laplace_weighting: bool = False
    checkpointing: bool = False

    def build(self, n_in_channels: int, n_out_channels: int, img_shape_x: int, img_shape_y: int):
        return FourierNeuralOperatorNet(
            params=self,
            in_chans = n_in_channels,
            out_chans = n_out_channels,
            img_size = (img_shape_x, img_shape_y),
        )


class FourierNeuralOperatorNet(nn.Module):
    def __init__(
            self,
            params: FourierNeuralOperatorBuilder,
            spectral_transform = 'sht',
            filter_type = 'non-linear',
            img_size = (721, 1440),
            scale_factor = 16,
            in_chans = 2,
            out_chans = 2,
            embed_dim = 256,
            num_layers = 12,
            mlp_mode = "none",
            mlp_ratio = 2.,
            drop_rate = 0.,
            drop_path_rate = 0.,
            num_blocks = 16,
            sparsity_threshold = 0.0,
            normalization_layer = 'instance_norm',
            hard_thresholding_fraction = 1.0,
            use_complex_kernels = True,
            big_skip = True,
            compression = None,
            rank = 128,
            complex_network = True,
            complex_activation = 'real',
            spectral_layers = 1,
            laplace_weighting = False,
            checkpointing = False): 
        super(FourierNeuralOperatorNet, self).__init__()

        self.params = params
        self.spectral_transform = params.spectral_transform if hasattr(params, "spectral_transform") else spectral_transform
        self.filter_type = params.filter_type if hasattr(params, "filter_type") else filter_type
        self.img_size = img_size
        self.scale_factor = params.scale_factor if hasattr(params, "scale_factor") else scale_factor
        self.in_chans = params.N_in_channels if hasattr(params, "N_in_channels") else in_chans
        self.out_chans = params.N_out_channels if hasattr(params, "N_out_channels") else out_chans
        self.embed_dim = self.num_features = params.embed_dim if hasattr(params, "embed_dim") else embed_dim
        self.num_layers = params.num_layers if hasattr(params, "num_layers") else num_layers
        self.num_blocks = params.num_blocks if hasattr(params, "num_blocks") else num_blocks
        self.hard_thresholding_fraction = params.hard_thresholding_fraction if hasattr(params, "hard_thresholding_fraction") else hard_thresholding_fraction
        self.normalization_layer = params.normalization_layer if hasattr(params, "normalization_layer") else normalization_layer
        self.mlp_mode = params.mlp_mode if hasattr(params, "mlp_mode") else mlp_mode
        self.big_skip = params.big_skip if hasattr(params, "big_skip") else big_skip
        self.compression = params.compression if hasattr(params, "compression") else compression
        self.rank = params.rank if hasattr(params, "rank") else rank
        self.complex_network = params.complex_network if hasattr(params, "complex_network") else complex_network
        self.complex_activation = params.complex_activation if hasattr(params, "complex_activation") else complex_activation
        self.spectral_layers = params.spectral_layers if hasattr(params, "spectral_layers") else spectral_layers
        self.laplace_weighting = params.laplace_weighting if hasattr(params, "laplace_weighting") else laplace_weighting
        self.checkpointing = params.checkpointing if hasattr(params, "checkpointing") else checkpointing

        # compute downsampled image size
        self.h = self.img_size[0] // self.scale_factor
        self.w = self.img_size[1] // self.scale_factor

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer0 = partial(nn.LayerNorm, normalized_shape=(self.img_size[0], self.img_size[1]), eps=1e-6)
            norm_layer1 = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
        elif self.normalization_layer == "instance_norm":
            norm_layer0 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
            norm_layer1 = norm_layer0
        # elif self.normalization_layer == "batch_norm":
        #     norm_layer = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        else:
            raise NotImplementedError(f"Error, normalization {self.normalization_layer} not implemented.") 

        # ENCODER is just an MLP?
        encoder_hidden_dim = self.embed_dim
        encoder_act = nn.GELU

        #encoder0 = nn.Conv2d(self.in_chans, encoder_hidden_dim, 1, bias=True)
        #encoder1 = nn.Conv2d(encoder_hidden_dim, self.embed_dim, 1, bias=False)
        #encoder_act = nn.GELU()
        #self.encoder = nn.Sequential(encoder0, encoder_act, encoder1, norm_layer0())

        self.encoder = MLP(in_features = self.in_chans,
                            hidden_features = encoder_hidden_dim,
                            out_features = self.embed_dim,
                            output_bias = False,
                            act_layer = encoder_act,
                            drop_rate = 0.,
                            checkpointing = checkpointing)

        # self.input_encoding = nn.Conv2d(self.in_chans, self.embed_dim, 1)
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.pos_embed_dim, self.img_size[0], self.img_size[1]))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], self.img_size[1]))

        # prepare the SHT
        modes_lat = int(self.h * self.hard_thresholding_fraction)
        modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)
        
        if self.spectral_transform == 'sht':
            self.trans_down = harmonics.RealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid='equiangular').float()
            self.itrans_up  = harmonics.InverseRealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid='equiangular').float()
            self.trans      = harmonics.RealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid='legendre-gauss').float()
            self.itrans     = harmonics.InverseRealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid='legendre-gauss').float()
            
            # we introduce some ad-hoc rescaling of the weights to aid gradient computation:
            sht_rescaling_factor = 1e5
            self.trans_down.weights = self.trans_down.weights * sht_rescaling_factor
            self.itrans_up.pct = self.itrans_up.pct / sht_rescaling_factor
            self.trans.weights = self.trans.weights * sht_rescaling_factor
            self.itrans.pct = self.itrans.pct / sht_rescaling_factor

        
        elif self.spectral_transform == 'fft':
            self.trans_down = RealFFT2(*self.img_size, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans_up  = InverseRealFFT2(*self.img_size, lmax=modes_lat, mmax=modes_lon).float()
            self.trans      = RealFFT2(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans     = InverseRealFFT2(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
        else:
            raise(ValueError('Unknown spectral transform'))

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            first_layer = (i == 0)
            last_layer = (i == self.num_layers-1)

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = 'linear' if 0 < i < self.num_layers-1 else None
            outer_skip = 'identity' if 0 < i < self.num_layers-1 else None
            mlp_mode = self.mlp_mode if not last_layer else 'none'

            if first_layer:
                norm_layer = (norm_layer0, norm_layer1)
            elif last_layer:
                norm_layer = (norm_layer1, norm_layer0)
            else:
                norm_layer = (norm_layer1, norm_layer1)

            block = FourierNeuralOperatorBlock(forward_transform,
                                               inverse_transform,
                                               self.embed_dim,
                                               filter_type = self.filter_type,
                                               mlp_ratio = mlp_ratio,
                                               drop_rate = drop_rate,
                                               drop_path = dpr[i],
                                               norm_layer = norm_layer,
                                               sparsity_threshold = sparsity_threshold,
                                               use_complex_kernels = use_complex_kernels,
                                               inner_skip = inner_skip,
                                               outer_skip = outer_skip,
                                               mlp_mode = mlp_mode,
                                               compression = self.compression,
                                               rank = self.rank,
                                               complex_network = self.complex_network,
                                               complex_activation = self.complex_activation,
                                               spectral_layers = self.spectral_layers,
                                               checkpointing = self.checkpointing)

            self.blocks.append(block)

        # DECODER is also an MLP
        decoder_hidden_dim = self.embed_dim
        decoder_act = nn.GELU

        #decoder0 = nn.Conv2d(self.embed_dim + self.big_skip*self.in_chans, decoder_hidden_dim, 1, bias=True)
        #decoder1 = nn.Conv2d(decoder_hidden_dim, self.out_chans, 1, bias=False)
        #decoder_act = nn.GELU()
        #self.decoder = nn.Sequential(decoder0, decoder_act, decoder1)

        self.decoder = MLP(in_features = self.embed_dim + self.big_skip*self.in_chans,
                            hidden_features = decoder_hidden_dim,
                            out_features = self.out_chans,
                            output_bias = False,
                            act_layer = decoder_act,
                            drop_rate = 0.,
                            checkpointing = checkpointing) 

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            #nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, FusedLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
            
        return x

    def forward(self, x):

        # save big skip
        if self.big_skip:
            residual = x

        # encoder
        x = self.encoder(x)

        # do positional embedding
        x = x + self.pos_embed

        # forward features
        x = self.forward_features(x)

        # concatenate the big skip
        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        # decoder
        x = self.decoder(x)

        return x

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from apex.normalization import FusedLayerNorm

from torch.utils.checkpoint import checkpoint

from torch_harmonics import *

# helpers
from networks.geometric.layers import trunc_normal_, DropPath

# from utils import comm
from torch_harmonics import *
from networks.geometric.contractions import *
# from networks.geometric import activations
from networks.geometric.activations import *

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 dims,
                 hidden_features = None,
                 out_features = None,
                 act_layer = nn.GELU,
                 drop_rate = 0.,
                 embed_pos = 0,
                 checkpointing = False):
        super(MLP, self).__init__()
        self.checkpointing = checkpointing
        self.embed_pos = embed_pos
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # prepare frequency embedding
        if self.embed_pos > 0:
            ii = torch.linspace(0, 1, dims[0]).reshape(-1, 1).expand(*dims)
            jj = torch.linspace(0, 1, dims[1]).reshape(1, -1).expand(*dims)
            pos = torch.stack([ii, jj], dim=0).reshape(1, 2, *dims)

            print(pos)

            # compute the frequency embeddings to map them onto (-1, 1)
            pos_embeddings = []
            for i in range(self.embed_pos):
                pos_embeddings.append(torch.sin(torch.pi * 2**i * pos))
                pos_embeddings.append(torch.cos(torch.pi * 2**i * pos))

            pos_embeddings = torch.cat(pos_embeddings, dim=1)
            assert pos_embeddings.shape[1] == 2*2*self.embed_pos
            self.register_buffer("pos_embeddings", pos_embeddings)

        self.fc1 = nn.Conv2d(in_features + 4*self.embed_pos, hidden_features, 1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=True)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()
        self.fwd = nn.Sequential(self.fc1, self.act, self.drop, self.fc2, self.drop)



    @torch.jit.ignore
    def checkpoint_forward(self, x):
        return checkpoint(self.fwd, x)
        
    def forward(self, x):

        if self.embed_pos:
            b, c, h, w = x.shape
            x = torch.cat((x, self.pos_embeddings.expand(b, -1, h, w)), dim=1)

        if self.checkpointing:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)


class SpectralConvS2(nn.Module):
    """
    Spectral Convolution as utilized in 
    """
    
    def __init__(self,
                 forward_transform,
                 inverse_transform,
                 hidden_size,
                 sparsity_threshold=0.0,
                 use_complex_kernels=False,
                 compression = None,
                 rank = 128,
                 bias = False):
        super(SpectralConvS2, self).__init__()
        
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.scale = 0.02

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.modes_lat = self.forward_transform.lmax
        self.modes_lon = self.forward_transform.mmax

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        # remember the lower triangular indices
        ii, jj = torch.tril_indices(self.modes_lat, self.modes_lon)
        self.register_buffer("ii", ii)
        self.register_buffer("jj", jj)

        if compression == 'tt':
            self.rank = rank
            # tensortrain coefficients
            g1 = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.rank, 2))
            g2 = nn.Parameter(self.scale * torch.randn(self.rank, self.hidden_size, self.rank, 2))
            g3 = nn.Parameter(self.scale * torch.randn(self.rank, len(ii), 2))
            self.w = nn.ParameterList([g1, g2, g3])

            self.contract_handle = contract_tt #if use_complex_kernels else raise(NotImplementedError)
        else:
            self.w = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.hidden_size, len(ii), 2))
            self.contract_handle = compl_contract_fwd_c if use_complex_kernels else compl_contract_fwd

    
        if bias:
            self.b = nn.Parameter(self.scale * torch.randn(1, self.hidden_size, *self.output_dims))

        
    def forward(self, x):

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        
        x = self.forward_transform(x)
        
        # do spectral conv
        x = torch.view_as_real(x)
        modes = torch.zeros(x.shape, device=x.device)
        modes[:, :, self.ii, self.jj, :] = self.contract_handle(x[:, :, self.ii, self.jj, :], self.w)

        # finalize
        x = F.softshrink(modes, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)

        x = torch.tril(x) # probably can skip the tril
        x = self.inverse_transform(x)
        if hasattr(self, 'b'):
            x = x + self.b
        x = x.type(dtype)
    
        return x

class SpectralAttentionS2(nn.Module):
    """
    geometrical Spectral Attention layer
    """
    
    def __init__(self,
                 forward_transform,
                 inverse_transform,
                 embed_dim,
                 sparsity_threshold = 0.0,
                 hidden_size_factor = 1,
                 use_complex_kernels = False,
                 complex_activation = 'real',
                 bias = False,
                 spectral_layers = 1,
                 drop_rate = 0.,
                 embed_freqs = 0):
        super(SpectralAttentionS2, self).__init__()
        
        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size = int(hidden_size_factor * self.embed_dim)
        self.scale = 0.02
        self.mul_add_handle = compl_mul_add_fwd_c if use_complex_kernels else compl_mul_add_fwd
        self.mul_handle = compl_mul_fwd_c if use_complex_kernels else compl_mul_fwd
        self.spectral_layers = spectral_layers
        self.embed_freqs = embed_freqs

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform.forward
        self.inverse_transform = inverse_transform.forward

        assert inverse_transform.lmax == self.modes_lat
        assert inverse_transform.mmax == self.modes_lon

        # remember the lower triangular indices
        ii, jj = torch.tril_indices(self.modes_lat, self.modes_lon)
        self.register_buffer("ii", ii)
        self.register_buffer("jj", jj)

        # if self.embed_freqs:
        #     freqs = torch.stack([ii, jj], dim=0).reshape(1, 2, -1).cfloat()
        #     freqs = torch.view_as_real(freqs)
        #     self.register_buffer("freqs", freqs)
        #     self.freq_embed = nn.Conv1d(2, self.embed_dim)

        # if self.embed_freqs:
        #     self.freqs = nn.Parameter(self.scale * torch.randn(1, self.embed_dim, len(ii), 2))

        # prepare frequency embedding
        if self.embed_freqs > 0:
            self.freq_max = 720
            freqs = torch.stack([ii/self.freq_max, jj/self.freq_max], dim=0).reshape(1, 2, -1)

            # compute the frequency embeddings to map them onto (-1, 1)
            freq_embeddings = []
            for i in range(self.embed_freqs):
                freq_embeddings.append(torch.sin(2**i * torch.pi * freqs))
                freq_embeddings.append(torch.cos(2**i * torch.pi * freqs))

            freq_embeddings = torch.cat(freq_embeddings, dim=1)
            assert freq_embeddings.shape[1] == 2*2*self.embed_freqs
            freq_embeddings = torch.view_as_real(freq_embeddings.cfloat())
            self.register_buffer("freq_embeddings", freq_embeddings)

        # weights
        w = [self.scale * torch.randn(self.embed_dim + 4*self.embed_freqs, self.hidden_size, 2)]
        # w = [self.scale * torch.randn(self.embed_dim + self.embed_freqs * 2, self.hidden_size, 2)]
        for l in range(1, self.spectral_layers):
            w.append(self.scale * torch.randn(self.hidden_size, self.hidden_size, 2))
        self.w = nn.ParameterList(w)

        if bias:
            self.b = nn.ParameterList([self.scale * torch.randn(self.hidden_size, 1, 2) for _ in range(self.spectral_layers)])
        
        self.wout = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.embed_dim, 2))

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        self.activation = ComplexReLU(mode=complex_activation, bias_shape=(self.hidden_size, 1, 1))
        # self.activation = ComplexSiLU(mode=complex_activation, bias_shape=(self.hidden_size, 1, 1))

    def forward_mlp(self, x):

        x = torch.view_as_real(x)

        xr = x[:, :, self.ii, self.jj, :]

        if self.embed_freqs:
            b = xr.size(0)
            xr = torch.cat((xr, self.freq_embeddings.expand(b, -1, -1, -1)), dim=1)

        for l in range(self.spectral_layers):
            if hasattr(self, 'b'):
                xr = self.mul_add_handle(xr, self.w[l], self.b[l])
            else:
                xr = self.mul_handle(xr, self.w[l])
            xr = torch.view_as_complex(xr)
            xr = self.activation(xr)
            xr = self.drop(xr)
            xr = torch.view_as_real(xr)
    
        x[:, :, self.ii, self.jj, :] = self.mul_handle(xr, self.wout)

        x = torch.view_as_complex(x)

        return x

    def forward(self, x):

        dtype = x.dtype
        x = x.float()

        x = self.forward_transform(x)
        x = self.forward_mlp(x)
        x = self.inverse_transform(x)
        x = x.type(dtype)

        return x

class SpectralFilterLayer(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type = 'convolution',
        sparsity_threshold = 0.0,
        use_complex_kernels = True,
        hidden_size_factor = 1,
        num_blocks = 8,
        compression = None,
        rank = 128,
        complex_activation = 'real',
        spectral_layers = 1,
        embed_freqs = 0,
        drop_rate = 0.):
        super(SpectralFilterLayer, self).__init__() 

        if filter_type == 'convolution':
            self.filter = SpectralConvS2(forward_transform,
                                         inverse_transform,
                                         embed_dim,
                                         sparsity_threshold,
                                         use_complex_kernels = use_complex_kernels,
                                         compression = compression,
                                         rank = rank,
                                         bias = False)

        elif filter_type == 'attention':
            self.filter = SpectralAttentionS2(forward_transform,
                                              inverse_transform,
                                              embed_dim,
                                              sparsity_threshold,
                                              use_complex_kernels = use_complex_kernels,
                                              hidden_size_factor = hidden_size_factor,
                                              complex_activation = complex_activation,
                                              spectral_layers = spectral_layers,
                                              drop_rate = drop_rate,
                                              bias = False,
                                              embed_freqs = embed_freqs)
                                              

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
            filter_type = 'convolution',
            mlp_ratio = 2.,
            drop_rate = 0.,
            drop_path = 0.,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm,
            num_blocks = 8,
            sparsity_threshold = 0.0,
            use_complex_kernels = True,
            compression = None,
            rank = 128,
            inner_skip = 'linear',
            outer_skip = None, # None, nn.linear or nn.Identity
            concat_skip = True,
            use_mlp = False,
            complex_activation = 'real',
            spectral_layers = 1,
            embed_freqs = 0,
            embed_pos = 0,
            checkpointing = False):
        super(FourierNeuralOperatorBlock, self).__init__()
        
        # norm layer
        self.norm1 = norm_layer() #((h,w))

        # convolution layer
        self.filter = SpectralFilterLayer(forward_transform,
                                          inverse_transform,
                                          embed_dim,
                                          filter_type,
                                          sparsity_threshold,
                                          use_complex_kernels = use_complex_kernels,
                                          num_blocks = num_blocks,
                                          compression = compression,
                                          rank = rank,
                                          complex_activation = complex_activation,
                                          spectral_layers = spectral_layers,
                                          embed_freqs = embed_freqs,
                                          drop_rate = drop_rate)

        if inner_skip == 'linear':
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif inner_skip == 'identity':
            self.inner_skip = nn.Identity()

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2*embed_dim, embed_dim, 1, bias=False)

        self.act_layer = act_layer()
        
        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # norm layer
        self.norm2 = norm_layer() #((h,w))
        
        if use_mlp == True:
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLP(in_features = embed_dim,
                           dims = (inverse_transform.nlat, inverse_transform.nlon),
                           hidden_features = mlp_hidden_dim,
                           act_layer = act_layer,
                           drop_rate = drop_rate,
                           embed_pos = embed_pos,
                           checkpointing = checkpointing)

        if outer_skip == 'linear':
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif outer_skip == 'identity':
            self.outer_skip = nn.Identity()

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = nn.Conv2d(2*embed_dim, embed_dim, 1, bias=False)

    def forward(self, x):
        
        residual = x
        
        x = self.norm1(x)
        x = self.filter(x)

        if hasattr(self, 'inner_skip'):
            if self.concat_skip:
                x = torch.cat((x, self.inner_skip(residual)), dim=1)
                x = self.inner_skip_conv(x)
            else:
                x = x + self.inner_skip(residual)

        x = self.act_layer(x)
        x = self.norm2(x)

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


class FourierNeuralOperatorNet(nn.Module):
    def __init__(
            self,
            params,
            nettype = 'fno',
            img_size = (720, 1440),
            scale_factor = 16,
            in_chans = 2,
            out_chans = 2,
            embed_dim = 768,
            num_layers = 12,
            use_mlp = False,
            mlp_ratio = 2.,
            drop_rate = 0.,
            drop_path_rate = 0.,
            num_blocks = 16,
            sparsity_threshold = 0.0,
            normalization_layer = 'instance_norm',
            hard_thresholding_fraction = 1.0,
            use_complex_kernels = True,
            big_skip = True,
            # output_transform = False,
            compression = None,
            rank = 128,
            complex_activation = 'real',
            spectral_layers = 2,
            embed_freqs = 0,
            embed_pos = 0,
            checkpointing = False): 
        super(FourierNeuralOperatorNet, self).__init__()

        self.params = params
        self.nettype = params.nettype if hasattr(params, "nettype") else nettype
        self.img_size = (params.img_shape_x, params.img_shape_y) if hasattr(params, "img_shape_x") and hasattr(params, "img_shape_y") else img_size
        self.scale_factor = params.scale_factor if hasattr(params, "scale_factor") else scale_factor
        self.in_chans = params.N_in_channels if hasattr(params, "N_in_channels") else in_chans
        self.out_chans = params.N_out_channels if hasattr(params, "N_out_channels") else out_chans
        self.embed_dim = self.num_features = params.embed_dim if hasattr(params, "embed_dim") else embed_dim
        self.pos_embed_dim = params.pos_embed_dim if hasattr(params, "pos_embed_dim") else self.embed_dim
        self.num_layers = params.num_layers if hasattr(params, "num_layers") else num_layers
        self.num_blocks = params.num_blocks if hasattr(params, "num_blocks") else num_blocks
        self.hard_thresholding_fraction = params.hard_thresholding_fraction if hasattr(params, "hard_thresholding_fraction") else hard_thresholding_fraction
        self.normalization_layer = params.normalization_layer if hasattr(params, "normalization_layer") else normalization_layer
        self.use_mlp = params.use_mlp if hasattr(params, "use_mlp") else use_mlp
        self.big_skip = params.big_skip if hasattr(params, "big_skip") else big_skip
        # self.output_transform = params.output_transform if hasattr(params, "output_transform") else output_transform
        self.compression = params.compression if hasattr(params, "compression") else compression
        self.rank = params.rank if hasattr(params, "rank") else rank
        self.complex_activation = params.complex_activation if hasattr(params, "complex_activation") else complex_activation
        self.spectral_layers = params.spectral_layers if hasattr(params, "spectral_layers") else spectral_layers
        self.embed_freqs = params.embed_freqs if hasattr(params, "embed_freqs") else embed_freqs
        self.embed_pos = params.embed_pos if hasattr(params, "embed_pos") else embed_pos
        self.checkpointing = params.checkpointing if hasattr(params, "checkpointing") else checkpointing

        # compute downsampled image size
        self.h = self.img_size[0] // self.scale_factor
        self.w = self.img_size[1] // self.scale_factor

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
        elif self.normalization_layer == "instance_norm":
            norm_layer = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        else:
            raise NotImplementedError(f"Error, normalization {self.normalization_layer} not implemented.") 

        encoder_hidden_dim = self.embed_dim

        encoder0 = nn.Conv2d(self.in_chans + self.pos_embed_dim, encoder_hidden_dim, 1, bias=True)
        # encoder0 = nn.Conv2d(self.in_chans + self.pos_embed_dim, encoder_hidden_dim, 1, bias=True)
        encoder1 = nn.Conv2d(encoder_hidden_dim, self.embed_dim, 1, bias=False)
        encoder_act = nn.GELU()
        self.encoder = nn.Sequential(encoder0, encoder_act, encoder1, norm_layer())

        # self.input_encoding = nn.Conv2d(self.in_chans, self.embed_dim, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.pos_embed_dim, self.img_size[0], self.img_size[1]))

        # prepare the SHT
        modes_lat = int(self.h * self.hard_thresholding_fraction)
        modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)
        self.sht_down = RealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid='equiangular').float()
        self.isht_up = InverseRealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid='equiangular').float()
        self.sht = RealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid='legendre-gauss').float()
        self.isht = InverseRealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid='legendre-gauss').float()

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            first_layer = i == 0
            last_layer = i == self.num_layers-1

            forward_transform = self.sht_down if first_layer else self.sht
            inverse_transform = self.isht_up if last_layer else self.isht

            inner_skip = 'linear' if 0 < i < self.num_layers-1 else None
            outer_skip = 'identity' if 0 < i < self.num_layers-1 else None
            use_mlp = self.use_mlp if not last_layer else False

            if self.nettype == 'hfno':
                filter_type = 'convolution'
            elif self.nettype == 'hafno':
                filter_type = 'attention'
            else:
                raise(ValueError('Unknown nettype'))

            block = FourierNeuralOperatorBlock(forward_transform,
                                               inverse_transform,
                                               self.embed_dim,
                                               filter_type = filter_type,
                                               mlp_ratio = mlp_ratio,
                                               drop_rate = drop_rate,
                                               drop_path = dpr[i],
                                               norm_layer = norm_layer,
                                               sparsity_threshold = sparsity_threshold,
                                               use_complex_kernels = use_complex_kernels,
                                               num_blocks = self.num_blocks,
                                               inner_skip = inner_skip,
                                               outer_skip = outer_skip,
                                               use_mlp = use_mlp,
                                               compression = self.compression,
                                               rank = self.rank,
                                               complex_activation = self.complex_activation,
                                               spectral_layers = self.spectral_layers,
                                               embed_freqs = self.embed_freqs,
                                               embed_pos = self.embed_pos,
                                               checkpointing = self.checkpointing)

            self.blocks.append(block)

        decoder_hidden_dim = self.embed_dim

        decoder0 = nn.Conv2d(self.embed_dim + self.big_skip*self.in_chans, decoder_hidden_dim, 1, bias=True)
        decoder_act = nn.GELU()
        decoder1 = nn.Conv2d(decoder_hidden_dim, self.out_chans, 1, bias=False)
        self.decoder = nn.Sequential(decoder0, decoder_act, decoder1)

        # if self.output_transform:
        #     self.output_scale = nn.Parameter(torch.ones(1, self.out_chans, self.img_size[0], self.img_size[1]))
        #     self.output_bias  = nn.Parameter(torch.zeros(1, self.out_chans, self.img_size[0], self.img_size[1]))

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

        if self.big_skip:
            residual = x

        b, c, h, w = x.shape
        x = torch.cat((x, self.pos_embed.expand(b, -1, -1, -1)), dim=1).contiguous()

        x = self.encoder(x)

        x = self.forward_features(x)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        x = self.decoder(x)

        return x

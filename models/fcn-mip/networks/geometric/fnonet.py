from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex.normalization import FusedLayerNorm
import torch.fft

# helpers
from networks.geometric.layers import trunc_normal_, DropPath, MLP, PatchEmbed
from networks.geometric.layers import SpectralConv2d, SpectralConvS2
from networks.geometric.layers import SpectralAttention2d, SpectralAttentionS2

class SpectralFilterLayer(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        embed_dim,
        filter_type = 'convolution',
        sparsity_threshold = 0.0,
        hard_thresholding_fraction = 1.0,
        spectral_transform = 'sht',
        use_complex_kernels = True,
        num_blocks = 8,
        compression = None,
        rank = 128,
        first_layer = False,
        last_layer = False,
        complex_activation = 'real'):
        super(SpectralFilterLayer, self).__init__() 

        if spectral_transform == 'sht':

            output_grid = 'equiangular' if last_layer else 'legendre-gauss'
            input_grid = 'equiangular' if first_layer else 'legendre-gauss'

            if filter_type == 'convolution':
                self.filter = SpectralConvS2(input_dims,
                                             output_dims,
                                             embed_dim,
                                             sparsity_threshold,
                                             hard_thresholding_fraction,
                                             use_complex_kernels = use_complex_kernels,
                                             input_grid = input_grid,
                                             output_grid = output_grid,
                                             compression = compression,
                                             rank = rank,
                                             bias = False)

            elif filter_type == 'attention':
                self.filter = SpectralAttentionS2(input_dims,
                                                  output_dims,
                                                  embed_dim,
                                                  num_blocks,
                                                  sparsity_threshold,
                                                  hard_thresholding_fraction,
                                                  use_complex_kernels = use_complex_kernels,
                                                  complex_activation = complex_activation,
                                                  input_grid = input_grid,
                                                  output_grid = output_grid,
                                                  bias = False)

            else:
                raise(NotImplementedError)

        elif spectral_transform == 'fft':

            if filter_type == 'convolution':
                self.filter = SpectralConv2d(input_dims,
                                             output_dims,
                                             embed_dim,
                                             sparsity_threshold,
                                             hard_thresholding_fraction,
                                             use_complex_kernels=use_complex_kernels,
                                             bias = False)

            elif filter_type == 'attention':
                self.filter = SpectralAttention2d(embed_dim,
                                                  num_blocks,
                                                  sparsity_threshold,
                                                  hard_thresholding_fraction,
                                                  use_complex_kernels=use_complex_kernels)

    def forward(self, x):
        return self.filter(x)


class FourierNeuralOperatorBlock(nn.Module):
    def __init__(
            self,
            input_dims,
            output_dims,
            embed_dim,
            filter_type = 'convolution',
            mlp_ratio = 4.,
            drop_rate = 0.,
            drop_path = 0.,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm,
            num_blocks = 8,
            sparsity_threshold = 0.0,
            hard_thresholding_fraction = 1.0,
            spectral_transform = 'sht',
            use_complex_kernels = True,
            compression = None,
            rank = 128,
            inner_skip = 'linear',
            outer_skip = None, # None, nn.linear or nn.Identity
            concat_skip = True,
            use_mlp = False,
            checkpointing = False,
            first_layer = False,
            last_layer = False,
            complex_activation = 'real'):
        super(FourierNeuralOperatorBlock, self).__init__()

        if inner_skip or outer_skip:
            assert input_dims == output_dims, "skip connections require equal-sized in- and ouputs"
        
        # norm layer
        self.norm1 = norm_layer() #((h,w))

        # convolution layer
        self.filter = SpectralFilterLayer(input_dims,
                                          output_dims,
                                          embed_dim,
                                          filter_type,
                                          sparsity_threshold,
                                          hard_thresholding_fraction,
                                          spectral_transform = spectral_transform,
                                          use_complex_kernels = use_complex_kernels,
                                          num_blocks = num_blocks,
                                          compression = compression,
                                          rank = rank,
                                          first_layer = first_layer,
                                          last_layer = last_layer,
                                          complex_activation = complex_activation)

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
            mlp_ratio = 4.,
            drop_rate = 0.,
            drop_path_rate = 0.,
            num_blocks = 16,
            sparsity_threshold = 0.0,
            normalization_layer = 'instance_norm',
            hard_thresholding_fraction = 1.0,
            use_complex_kernels = True,
            spectral_transform = 'sht',
            append_skip = True,
            compression = None,
            rank = 128,
            complex_activation = 'real'): 
        super(FourierNeuralOperatorNet, self).__init__()

        self.params = params
        self.nettype = params.nettype if hasattr(params, "nettype") else nettype
        self.img_size = (params.img_shape_x, params.img_shape_y) if hasattr(params, "img_shape_x") and hasattr(params, "img_shape_y") else img_size
        self.scale_factor = params.scale_factor if hasattr(params, "scale_factor") else scale_factor
        self.in_chans = params.N_in_channels if hasattr(params, "N_in_channels") else in_chans
        self.out_chans = params.N_out_channels if hasattr(params, "N_out_channels") else out_chans
        self.embed_dim = self.num_features = params.embed_dim if hasattr(params, "embed_dim") else embed_dim
        self.num_layers = params.num_layers if hasattr(params, "num_layers") else num_layers
        self.num_blocks = params.num_blocks if hasattr(params, "num_blocks") else num_blocks
        self.hard_thresholding_fraction = params.hard_thresholding_fraction if hasattr(params, "hard_thresholding_fraction") else hard_thresholding_fraction
        self.spectral_transform = params.spectral_transform if hasattr(params, "spectral_transform") else spectral_transform
        self.normalization_layer = params.normalization_layer if hasattr(params, "normalization_layer") else normalization_layer
        self.use_mlp = params.use_mlp if hasattr(params, "use_mlp") else use_mlp
        self.append_skip = params.append_skip if hasattr(params, "append_skip") else append_skip
        self.compression = params.compression if hasattr(params, "compression") else compression
        self.rank = params.rank if hasattr(params, "rank") else rank
        self.complex_activation = params.complex_activation if hasattr(params, "complex_activation") else complex_activation

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

        self.input_encoding = nn.Conv2d(self.in_chans, self.embed_dim, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], self.img_size[1])) 

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            first_layer = i == 0
            last_layer = i == self.num_layers-1

            input_dims = self.img_size if first_layer else (self.h, self.w)
            output_dims = self.img_size if last_layer else (self.h, self.w)
            inner_skip = 'linear' if 0 < i < self.num_layers-1 else None
            outer_skip = 'identity' if 0 < i < self.num_layers-1 else None
            use_mlp = self.use_mlp if not last_layer else False

            if self.nettype == 'fno':
                filter_type = 'convolution'
            elif self.nettype == 'afno':
                filter_type = 'attention'
            elif self.nettype == 'hybrid':
                if first_layer or last_layer:
                    filter_type = 'attention'
                else:
                    filter_type = 'attention'
            else:
                raise(ValueError('Unknown nettype'))

            block = FourierNeuralOperatorBlock(input_dims,
                                               output_dims,
                                               self.embed_dim,
                                               filter_type = filter_type,
                                               mlp_ratio = mlp_ratio,
                                               drop_rate = drop_rate,
                                               drop_path = dpr[i],
                                               norm_layer = norm_layer,
                                               sparsity_threshold = sparsity_threshold,
                                               hard_thresholding_fraction = self.hard_thresholding_fraction,
                                               spectral_transform = self.spectral_transform,
                                               use_complex_kernels = use_complex_kernels,
                                               num_blocks = self.num_blocks,
                                               inner_skip = inner_skip,
                                               outer_skip = outer_skip,
                                               use_mlp = use_mlp,
                                               first_layer = first_layer,
                                               last_layer = last_layer,
                                               compression = self.compression,
                                               rank = self.rank,
                                               complex_activation = self.complex_activation)

            self.blocks.append(block)

        self.fc_chans = 256
        self.head0 = nn.Linear(self.embed_dim + self.append_skip*self.in_chans, self.fc_chans)
        self.headnn = nn.GELU()
        self.head1 = nn.Linear(self.fc_chans, self.out_chans)

        # self.head = nn.Conv2d(self.embed_dim + self.append_skip*self.in_chans, self.out_chans, 1, bias=False)


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
 
    def forward_head(self, x):
        x = x.permute(0,2,3,1)
        x = self.head0(x)
        x = self.headnn(x)
        x = self.head1(x)
        x = x.permute(0,3,1,2)

        return x

    def forward_features(self, x):
        x = self.input_encoding(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
            
        return x

    def forward(self, x):

        if self.append_skip:
            residual = x

        x = self.forward_features(x)

        if self.append_skip:
            x = torch.cat((x, residual), dim=1)

        x = self.forward_head(x)
        # x = self.head(x)

        return x
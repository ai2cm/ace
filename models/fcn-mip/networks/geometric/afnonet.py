from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex.normalization import FusedLayerNorm
import torch.fft

# helpers
from networks.geometric.layers import trunc_normal_, DropPath, MLP, PatchEmbed
from networks.geometric.layers import SpectralAttention2d, SpectralAttentionS2


class AdaptiveFourierNeuralOperatorBlock(nn.Module):
    def __init__(
            self,
            img_size,
            embed_dim,
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
            inner_skip = 'identity',
            outer_skip = 'identity', # None, nn.linear or nn.Identity
            concat_skip = True, # should maybe check nested skip
            use_mlp = True,
            checkpointing = False, # keep this?
            position = 0 # 1 for inbetween, 0 for first, -1 for last
        ):
        super(AdaptiveFourierNeuralOperatorBlock, self).__init__()
        
        # norm layer
        self.norm1 = norm_layer() #((h,w))

        # filter layer
        if spectral_transform == 'sht':
            output_grid = 'equiangular' if position == -1 else 'legendre-gauss'
            input_grid = 'equiangular' if position == 0 else 'legendre-gauss'
            self.filter = SpectralAttentionS2(img_size,
                                              img_size,
                                              embed_dim,
                                              num_blocks,
                                              sparsity_threshold,
                                              hard_thresholding_fraction,
                                              use_complex_kernels = use_complex_kernels,
                                              input_grid = input_grid,
                                              output_grid = output_grid)
        elif spectral_transform == 'fft':
            self.filter = SpectralAttention2d(embed_dim,
                                              num_blocks,
                                              sparsity_threshold,
                                              hard_thresholding_fraction,
                                              use_complex_kernels=use_complex_kernels)

        if inner_skip == 'linear':
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif inner_skip == 'identity':
            self.inner_skip = nn.Identity()

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2*embed_dim, embed_dim, 1, bias=False)

        
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


class AdaptiveFourierNeuralOperatorNet(nn.Module):
    def __init__(
            self,
            params,
            img_size = (720, 1440),
            patch_size = (16, 16),
            in_chans = 2,
            out_chans = 2,
            embed_dim = 768,
            num_layers = 12,
            use_mlp = True,
            mlp_ratio = 4.,
            drop_rate = 0.,
            drop_path_rate = 0.,
            num_blocks = 16,
            sparsity_threshold = 0.0,
            normalization_layer = 'instance_norm',
            hard_thresholding_fraction = 1.0,
            use_complex_kernels=True,
            spectral_transform='sht'): 
        super(AdaptiveFourierNeuralOperatorNet, self).__init__()

        self.params = params
        self.img_size = (params.img_shape_x, params.img_shape_y) if hasattr(params, "img_shape_x") and hasattr(params, "img_shape_y") else img_size
        self.patch_size = (params.patch_size, params.patch_size) if hasattr(params, "patch_size") else patch_size
        self.in_chans = params.N_in_channels if hasattr(params, "N_in_channels") else in_chans
        self.out_chans = params.N_out_channels if hasattr(params, "N_out_channels") else out_chans
        self.embed_dim = self.num_features = params.embed_dim if hasattr(params, "embed_dim") else embed_dim
        self.num_layers = params.num_layers if hasattr(params, "num_layers") else num_layers
        self.num_blocks = params.num_blocks if hasattr(params, "num_blocks") else num_blocks
        self.hard_thresholding_fraction = params.hard_thresholding_fraction if hasattr(params, "hard_thresholding_fraction") else hard_thresholding_fraction
        self.spectral_transform = params.spectral_transform if hasattr(params, "spectral_transform") else spectral_transform
        self.normalization_layer = params.normalization_layer if hasattr(params, "normalization_layer") else normalization_layer
        self.use_mlp = params.use_mlp if hasattr(params, "use_mlp") else use_mlp

        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches

        # original: x = B, H*W, C
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # new: x = B, C, H*W
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, num_patches))
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
        elif self.normalization_layer == "instance_norm":
            norm_layer = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        else:
            raise NotImplementedError(f"Error, normalization {self.normalization_layer} not implemented.") 

        self.blocks = nn.ModuleList([
            AdaptiveFourierNeuralOperatorBlock((self.h, self.w),
                                               self.embed_dim,
                                               mlp_ratio = mlp_ratio,
                                               drop_rate = drop_rate,
                                               drop_path = dpr[i],
                                               norm_layer = norm_layer,
                                               num_blocks = self.num_blocks,
                                               sparsity_threshold = sparsity_threshold,
                                               hard_thresholding_fraction = self.hard_thresholding_fraction,
                                               use_complex_kernels = use_complex_kernels,
                                               spectral_transform = self.spectral_transform,
                                               inner_skip = 'identity',
                                               outer_skip = 'identity', # becomes the second skip connection if double skip is true
                                               concat_skip = True,
                                               use_mlp = self.use_mlp,
                                               position = -1 if i==self.num_layers-1 else i) 
        for i in range(self.num_layers)])

        # head
        self.head = nn.Conv2d(self.embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], 1, bias=False)

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
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # reshape
        x = x.reshape(B, self.embed_dim, self.h, self.w)

        for blk in self.blocks:
            x = blk(x)
            
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        # new: B, C, H, W
        b = x.shape[0]
        xv = x.view(b, self.patch_size[0], self.patch_size[1], -1, self.h, self.w)
        xvt = torch.permute(xv, (0, 3, 4, 1, 5, 2)).contiguous()
        x = xvt.view(b, -1, (self.h * self.patch_size[0]), (self.w * self.patch_size[1]))
        
        return x
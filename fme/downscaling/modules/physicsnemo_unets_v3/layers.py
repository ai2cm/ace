"""
Performance-optimized version of physicsnemo_unets_v2/layers.py.

Changes from v2:
- Removed _validate_amp function entirely
- Removed amp_mode parameter from all layers
- Removed per-forward dtype conversion blocks
- Conv2d: pre-computes tiled resample filter as buffer in __init__
- PositionalEmbedding: pre-computes frequencies as buffer for cos_sin path
- PositionalEmbedding: swap_sin_cos parameter to return (sin, cos) order
- Attention: cleaner q/k/v reshape pattern
- UNetBlock: removed profile_mode parameter

Original vendorized from physicsnemo:
https://github.com/NVIDIA/physicsnemo/blob/08dc147e194bd181e418735959507d3afc9f3978
"""
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Literal

import numpy as np
import torch
from torch.nn.functional import silu

from .group_norm import get_group_norm

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def _weight_init(shape: tuple, mode: str, fan_in: int, fan_out: int):
    """
    Unified routine for initializing weights and biases.
    This function provides a unified interface for various weight initialization
    strategies like Xavier (Glorot) and Kaiming (He) initializations.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor to initialize. It could represent weights or biases
        of a layer in a neural network.
    mode : str
        The mode/type of initialization to use. Supported values are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
    fan_in : int
        The number of input units in the weight tensor.
    fan_out : int
        The number of output units in the weight tensor.

    Returns:
    -------
    torch.Tensor
        The initialized tensor based on the specified mode.

    Raises:
    ------
    ValueError
        If the provided `mode` is not one of the supported initialization modes.
    """
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ------------------------------------------------------------------------------
# Layer modules
# ------------------------------------------------------------------------------


class Conv2d(torch.nn.Module):
    """
    A custom 2D convolutional layer implementation with support for up-sampling,
    down-sampling, and custom weight and bias initializations.

    Pre-computes the tiled resample filter at initialization time to avoid
    repeated .tile() calls during forward passes.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel : int
        Size of the convolving kernel.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an
        additive bias. By default True.
    up : bool, optional
        Whether to perform up-sampling. By default False.
    down : bool, optional
        Whether to perform down-sampling. By default False.
    resample_filter : List[int], optional
        Filter to be used for resampling. By default [1, 1].
    fused_resample : bool, optional
        If True, performs fused up-sampling and convolution or fused down-sampling
        and convolution. By default False.
    init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases.
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.0.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.0.
    fused_conv_bias: bool, optional
        A boolean flag indicating whether bias will be passed as a parameter of
        conv2d. By default False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        bias: bool = True,
        up: bool = False,
        down: bool = False,
        resample_filter: list[int] = [1, 1],
        fused_resample: bool = False,
        init_mode: str = "kaiming_normal",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
        fused_conv_bias: bool = False,
    ):
        if up and down:
            raise ValueError("Both 'up' and 'down' cannot be true at the same time.")
        if not kernel and fused_conv_bias:
            print(
                "Warning: Kernel is required when fused_conv_bias is enabled. "
                "Setting fused_conv_bias to False."
            )
            fused_conv_bias = False

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        self.fused_conv_bias = fused_conv_bias
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                _weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(_weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

        # Pre-compute tiled resample filter to avoid per-forward .tile() calls
        if up or down:
            if up:
                # Used for both fused and non-fused up paths
                tiled = f.mul(4).tile([in_channels, 1, 1, 1])
            elif down and fused_resample:
                # Fused down path tiles by out_channels
                tiled = f.tile([out_channels, 1, 1, 1])
            else:
                # Non-fused down path tiles by in_channels
                tiled = f.tile([in_channels, 1, 1, 1])
            self.register_buffer("tiled_resample_filter", tiled)
        else:
            self.register_buffer("tiled_resample_filter", None)

    def forward(self, x):
        w = self.weight
        b = self.bias
        f = self.tiled_resample_filter
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (
            (self.resample_filter.shape[-1] - 1) // 2
            if self.resample_filter is not None
            else 0
        )

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f,
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            if self.fused_conv_bias:
                x = torch.nn.functional.conv2d(
                    x, w, padding=max(w_pad - f_pad, 0), bias=b
                )
            else:
                x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            if self.fused_conv_bias:
                x = torch.nn.functional.conv2d(
                    x,
                    f,
                    groups=self.out_channels,
                    stride=2,
                    bias=b,
                )
            else:
                x = torch.nn.functional.conv2d(
                    x,
                    f,
                    groups=self.out_channels,
                    stride=2,
                )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f,
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f,
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                if self.fused_conv_bias:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad, bias=b)
                else:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None and not self.fused_conv_bias:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class Linear(torch.nn.Module):
    """
    A fully connected (dense) layer implementation.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an additive
        bias. By default True.
    init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases.
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mode: str = "kaiming_normal",
        init_weight: int = 1,
        init_bias: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            _weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(_weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.t()
        if self.bias is not None:
            x = x.add_(self.bias)
        return x


class FourierEmbedding(torch.nn.Module):
    """
    Generates Fourier embeddings for timesteps, primarily used in the NCSN++
    architecture.

    Parameters:
    -----------
    num_channels : int
        The number of channels in the embedding. The final embedding size will be
        2 * num_channels because of concatenation of cosine and sine results.
    scale : int, optional
        A scale factor applied to the random frequencies. By default 16.
    """

    def __init__(self, num_channels: int, scale: int = 16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger(2 * np.pi * self.freqs)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class PositionalEmbedding(torch.nn.Module):
    """
    A module for generating positional embeddings based on timesteps.
    This embedding technique is employed in the DDPM++ and ADM architectures.

    Pre-computes frequency tensors as buffers to avoid per-forward torch.arange calls.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    max_positions : int, optional
        Maximum number of positions for the embeddings, by default 10000.
    endpoint : bool, optional
        If True, the embedding considers the endpoint. By default False.
    learnable : bool, optional
        A boolean flag indicating whether learnable positional embedding is enabled.
        Defaults to False.
    freq_embed_dim: int, optional
        The dimension of the frequency embedding. Defaults to None, in which case it
        will be set to num_channels.
    mlp_hidden_dim: int, optional
        The dimension of the hidden layer in the MLP. Defaults to None, in which case
        it will be set to 2 * num_channels.
        Only applicable if learnable is True.
    embed_fn: Literal["cos_sin", "np_sin_cos"], optional
        The function to use for embedding into sin/cos features. Defaults to 'cos_sin'.
        Options:
            - 'cos_sin': Returns in order (cos, sin) by default, or (sin, cos) if
                swap_sin_cos is True.
            - 'np_sin_cos': Uses numpy to compute frequency embeddings and
                returns in order (sin, cos).
    swap_sin_cos : bool, optional
        When True and embed_fn='cos_sin', returns (sin, cos) instead of (cos, sin).
        Defaults to False.
    """

    def __init__(
        self,
        num_channels: int,
        max_positions: int = 10000,
        endpoint: bool = False,
        learnable: bool = False,
        freq_embed_dim: int | None = None,
        mlp_hidden_dim: int | None = None,
        embed_fn: Literal["cos_sin", "np_sin_cos"] = "cos_sin",
        swap_sin_cos: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.learnable = learnable
        self.embed_fn = embed_fn
        self.swap_sin_cos = swap_sin_cos

        if freq_embed_dim is None:
            freq_embed_dim = num_channels
        self.freq_embed_dim = freq_embed_dim

        if learnable:
            if mlp_hidden_dim is None:
                mlp_hidden_dim = 2 * num_channels
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(freq_embed_dim, mlp_hidden_dim, bias=True),
                torch.nn.SiLU(),
                torch.nn.Linear(mlp_hidden_dim, num_channels, bias=True),
            )

        if self.embed_fn == "np_sin_cos":
            half_embed_dim = freq_embed_dim // 2
            pow = np.arange(half_embed_dim, dtype=np.float32) / half_embed_dim
            w = np.exp(-np.log(self.max_positions) * pow)
            self.register_buffer("freqs", torch.from_numpy(w).float())
        elif self.embed_fn == "cos_sin":
            # Pre-compute frequencies for cos_sin path
            half_dim = freq_embed_dim // 2
            freqs = torch.arange(
                start=0, end=half_dim, dtype=torch.float32
            )
            freqs = freqs / (half_dim - (1 if self.endpoint else 0))
            freqs = (1 / self.max_positions) ** freqs
            self.register_buffer("freqs", freqs)

    def _cos_sin_embedding(self, x):
        x = x.ger(self.freqs)
        if self.swap_sin_cos:
            x = torch.cat([x.sin(), x.cos()], dim=1)
        else:
            x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

    def _sin_cos_embedding_np(self, x):
        x = torch.outer(x, self.freqs)
        x = torch.cat([x.sin(), x.cos()], dim=1)
        return x

    def forward(self, x):
        if self.embed_fn == "cos_sin":
            x = self._cos_sin_embedding(x)
        elif self.embed_fn == "np_sin_cos":
            x = self._sin_cos_embedding_np(x)

        if self.learnable:
            x = self.mlp(x)
        return x


class Attention(torch.nn.Module):
    """
    Self-attention block used in U-Net-style architectures.

    Uses a cleaner reshape pattern for q/k/v extraction compared to v2.

    Parameters
    ----------
    out_channels : int
        Number of channels in the input and output feature maps.
    num_heads : int
        Number of attention heads. Must be a positive integer.
    eps : float, optional, default=1e-5
        Epsilon value for numerical stability in GroupNorm.
    init_zero : dict, optional, default={'init_weight': 0}
        Initialization parameters with zero weights for certain layers.
    init_attn : dict, optional, default=None
        Initialization parameters specific to attention mechanism layers.
    init : dict, optional, default={}
        Initialization parameters for convolutional and linear layers.
    use_apex_gn : bool, optional, default=False
        A boolean flag indicating whether we want to use Apex GroupNorm for NHWC layout.
    fused_conv_bias: bool, optional, default=False
        A boolean flag indicating whether bias will be passed as a parameter of conv2d.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of the same shape as input: :math:`(B, C, H, W)`.
    """

    def __init__(
        self,
        *,
        out_channels: int,
        num_heads: int,
        eps: float = 1e-5,
        init_zero: dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
        init: dict[str, Any] = dict(),
        use_apex_gn: bool = False,
        fused_conv_bias: bool = False,
    ) -> None:
        super().__init__()
        # Parameters validation
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"`num_heads` must be a positive integer, but got {num_heads}"
            )
        if out_channels % num_heads != 0:
            raise ValueError(
                f"`out_channels` must be divisible by `num_heads`, but got "
                f"{out_channels} and {num_heads}"
            )
        self.num_heads = num_heads
        self.norm = get_group_norm(
            num_channels=out_channels,
            eps=eps,
            use_apex_gn=use_apex_gn,
        )
        self.qkv = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * 3,
            kernel=1,
            fused_conv_bias=fused_conv_bias,
            **(init_attn if init_attn is not None else init),
        )
        self.proj = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=1,
            fused_conv_bias=fused_conv_bias,
            **init_zero,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x1: torch.Tensor = self.qkv(self.norm(x))

        # Cleaner q/k/v extraction
        q, k, v = x1.reshape(B, 3, self.num_heads, -1, H * W).unbind(1)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(k.shape[-1])
        )
        attn = attn.transpose(-1, -2)

        x: torch.Tensor = self.proj(attn.reshape(*x.shape)).add_(x)
        return x


class UNetBlock(torch.nn.Module):
    """
    Unified U-Net block with optional up/downsampling and self-attention.

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    emb_channels : int
        Number of embedding channels.
    up : bool, optional, default=False
        If True, applies upsampling in the forward pass.
    down : bool, optional, default=False
        If True, applies downsampling in the forward pass.
    attention : bool, optional, default=False
        If True, enables the self-attention mechanism in the block.
    num_heads : int, optional, default=None
        Number of attention heads.
    channels_per_head : int, optional, default=64
        Number of channels per attention head.
    dropout : float, optional, default=0.0
        Dropout probability.
    skip_scale : float, optional, default=1.0
        Scale factor applied to skip connections.
    eps : float, optional, default=1e-5
        Epsilon value used for normalization layers.
    resample_filter : List[int], optional, default=[1, 1]
        Filter for resampling layers.
    resample_proj : bool, optional, default=False
        If True, resampling projection is enabled.
    adaptive_scale : bool, optional, default=True
        If True, uses adaptive scaling in the forward pass.
    init : dict, optional, default={}
        Initialization parameters for convolutional and linear layers.
    init_zero : dict, optional, default={'init_weight': 0}
        Initialization parameters with zero weights for certain layers.
    init_attn : dict, optional, default=None
        Initialization parameters specific to attention mechanism layers.
    use_apex_gn : bool, optional, default=False
        Whether to use Apex GroupNorm for NHWC layout.
    act : str, optional, default=None
        The activation function to use when fusing activation with GroupNorm.
    fused_conv_bias: bool, optional, default=False
        Whether bias will be passed as a parameter of conv2d.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)`.
    emb : torch.Tensor
        Embedding tensor of shape :math:`(B, C_{emb})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H, W)`.
    """

    # NOTE: these attributes have specific usage in old checkpoints, do not
    # reuse them!
    _reserved_attributes: set[str] = set(["norm2", "qkv", "proj"])

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        num_heads: int | None = None,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        resample_filter: list[int] = [1, 1],
        resample_proj: bool = False,
        adaptive_scale: bool = True,
        init: dict[str, Any] = dict(),
        init_zero: dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
        use_apex_gn: bool = False,
        act: str = "silu",
        fused_conv_bias: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else (
                num_heads
                if num_heads is not None
                else out_channels // channels_per_head
            )
        )
        self.attention = attention
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.norm0 = get_group_norm(
            num_channels=in_channels,
            eps=eps,
            use_apex_gn=use_apex_gn,
            act=act,
        )
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            fused_conv_bias=fused_conv_bias,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            **init,
        )
        if self.adaptive_scale:
            self.norm1 = get_group_norm(
                num_channels=out_channels,
                eps=eps,
                use_apex_gn=use_apex_gn,
            )
        else:
            self.norm1 = get_group_norm(
                num_channels=out_channels,
                eps=eps,
                use_apex_gn=use_apex_gn,
                act=act,
            )
        self.conv1 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            fused_conv_bias=fused_conv_bias,
            **init_zero,
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            fused_conv_bias = fused_conv_bias if kernel != 0 else False
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                fused_conv_bias=fused_conv_bias,
                **init,
            )

        if self.attention:
            self.attn = Attention(
                out_channels=out_channels,
                num_heads=self.num_heads,
                eps=eps,
                init_zero=init_zero,
                init_attn=init_attn,
                init=init,
                use_apex_gn=use_apex_gn,
                fused_conv_bias=fused_conv_bias,
            )
        else:
            self.attn = None

    def forward(self, x, emb):
        orig = x
        x = self.conv0(self.norm0(x))
        params = self.affine(emb).unsqueeze(2).unsqueeze(3)

        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = self.norm1(x.add_(params))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.attn:
            x = self.attn(x)
            x = x * self.skip_scale
        return x

    def __setattr__(self, name, value):
        """Prevent setting attributes with reserved names.

        Parameters
        ----------
        name : str
            Attribute name.
        value : Any
            Attribute value.
        """
        if name in getattr(self.__class__, "_reserved_attributes", set()):
            raise AttributeError(f"Attribute '{name}' is reserved and cannot be set.")
        super().__setattr__(name, value)

    @staticmethod
    def _migrate_attention_module(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """``load_state_dict`` pre-hook that handles legacy checkpoints that
        stored attention layers at root.
        """
        _mapping = {
            f"{prefix}norm2.weight": f"{prefix}attn.norm.weight",
            f"{prefix}norm2.bias": f"{prefix}attn.norm.bias",
            f"{prefix}qkv.weight": f"{prefix}attn.qkv.weight",
            f"{prefix}qkv.bias": f"{prefix}attn.qkv.bias",
            f"{prefix}proj.weight": f"{prefix}attn.proj.weight",
            f"{prefix}proj.bias": f"{prefix}attn.proj.bias",
        }

        for old_key, new_key in _mapping.items():
            if old_key in state_dict:
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
                else:
                    raise ValueError(
                        f"Checkpoint contains both legacy and new keys for {old_key}"
                    )

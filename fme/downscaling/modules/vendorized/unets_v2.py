"""
This file contains vendorized code from physicsnemo.

https://github.com/NVIDIA/physicsnemo/blob/08dc147e194bd181e418735959507d3afc9f3978

Note this is a newer version of the layer modules and SongUNet vendorized in unets.py.
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
from torch.utils.checkpoint import checkpoint

from .group_norm import get_group_norm

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def _validate_amp(amp_mode: bool) -> None:
    """Raise if `amp_mode` is False but PyTorch autocast (CPU or CUDA) is active.

    Parameters
    ----------
    amp_mode : bool
        Your intended AMP flag. Set False when you require full precision.
    """
    try:
        cuda_amp = bool(torch.is_autocast_enabled())
    except AttributeError:  # very old PyTorch
        cuda_amp = False
    try:
        cpu_amp = bool(torch.is_autocast_enabled("cpu"))
    except AttributeError:
        cpu_amp = False

    if not amp_mode and (cuda_amp or cpu_amp):
        active = []
        if cuda_amp:
            active.append("cuda")
        if cpu_amp:
            active.append("cpu")
        raise RuntimeError(
            f"amp_mode=False but torch autocast is enabled on: {', '.join(active)}. "
            "Disable autocast for this region or set amp_mode=True if mixed precision "
            "is intended."
        )


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
        The number of input units in the weight tensor. For convolutional layers,
        this typically represents the number of input channels times the kernel height
        times the kernel width.
    fan_out : int
        The number of output units in the weight tensor. For convolutional layers,
        this typically represents the number of output channels times the kernel height
        times the kernel width.

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


def _recursive_property(prop_name: str, prop_type: type, doc: str) -> property:
    """
    Property factory that sets the property on a Module ``self`` and
    recursively on all submodules.
    For ``self``, the property is stored under a semi-private ``_<prop_name>`` attribute
    and for submodules the setter is delegated to the ``setattr`` function.

    Parameters
    ----------
    prop_name : str
        The name of the property.
    prop_type : type
        The type of the property.
    doc : str
        The documentation string for the property.

    Returns:
    -------
    property
        The property object.
    """

    def _setter(self, value: Any):
        if not isinstance(value, prop_type):
            raise TypeError(
                f"{prop_name} must be a {prop_type.__name__} value, "
                f"but got {type(value).__name__}."
            )
        # Set for self
        setattr(self, f"_{prop_name}", value)
        # Set for submodules
        submodules = iter(self.modules())
        next(submodules)  # Skip self
        for m in submodules:
            if hasattr(m, prop_name):
                setattr(m, prop_name, value)

    def _getter(self):
        return getattr(self, f"_{prop_name}")

    return property(_getter, _setter, doc=doc)


# ------------------------------------------------------------------------------
# Layer modules
# ------------------------------------------------------------------------------


class Conv2d(torch.nn.Module):
    """
    A custom 2D convolutional layer implementation with support for up-sampling,
    down-sampling, and custom weight and bias initializations. The layer's weights
    and biases canbe initialized using custom initialization strategies like
    "kaiming_normal", and can be further scaled by factors `init_weight` and
    `init_bias`.

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
        init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.0.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.0.
    fused_conv_bias: bool, optional
        A boolean flag indicating whether bias will be passed as a parameter of
        conv2d. By default False.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled.
        Defaults to False.
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
        amp_mode: bool = False,
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
        self.amp_mode = amp_mode
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

    def forward(self, x):
        weight, bias, resample_filter = self.weight, self.bias, self.resample_filter
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if self.weight is not None and self.weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            if self.bias is not None and self.bias.dtype != x.dtype:
                bias = self.bias.to(x.dtype)
            if (
                self.resample_filter is not None
                and self.resample_filter.dtype != x.dtype
            ):
                resample_filter = self.resample_filter.to(x.dtype)

        w = weight if weight is not None else None
        b = bias if bias is not None else None
        f = resample_filter if resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
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
                    f.tile([self.out_channels, 1, 1, 1]),
                    groups=self.out_channels,
                    stride=2,
                    bias=b,
                )
            else:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.out_channels, 1, 1, 1]),
                    groups=self.out_channels,
                    stride=2,
                )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:  # ask in corrdiff channel whether w will ever be none
                if self.fused_conv_bias:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad, bias=b)
                else:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None and not self.fused_conv_bias:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class Linear(torch.nn.Module):
    """
    A fully connected (dense) layer implementation. The layer's weights and biases can
    be initialized using custom initialization strategies like "kaiming_normal",
    and can be further scaled by factors `init_weight` and `init_bias`.

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
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled.
        Defaults to False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mode: str = "kaiming_normal",
        init_weight: int = 1,
        init_bias: int = 0,
        amp_mode: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.amp_mode = amp_mode
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
        weight, bias = self.weight, self.bias
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if self.weight is not None and self.weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            if self.bias is not None and self.bias.dtype != x.dtype:
                bias = self.bias.to(x.dtype)
        x = x @ weight.t()
        if self.bias is not None:
            x = x.add_(bias)
        return x


class FourierEmbedding(torch.nn.Module):
    """
    Generates Fourier embeddings for timesteps, primarily used in the NCSN++
    architecture.

    This class generates embeddings by first multiplying input tensor `x` and
    internally stored random frequencies, and then concatenating the cosine and sine of
    the resultant.

    Parameters:
    -----------
    num_channels : int
        The number of channels in the embedding. The final embedding size will be
        2 * num_channels because of concatenation of cosine and sine results.
    scale : int, optional
        A scale factor applied to the random frequencies, controlling their range
        and thereby the frequency of oscillations in the embedding space. By default 16.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled.
        Defaults to False.
    """

    def __init__(self, num_channels: int, scale: int = 16, amp_mode: bool = False):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)
        self.amp_mode = amp_mode

    def forward(self, x):
        freqs = self.freqs
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if x.dtype != self.freqs.dtype:
                freqs = self.freqs.to(x.dtype)

        x = x.ger(2 * np.pi * freqs)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class PositionalEmbedding(torch.nn.Module):
    """
    A module for generating positional embeddings based on timesteps.
    This embedding technique is employed in the DDPM++ and ADM architectures.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    max_positions : int, optional
        Maximum number of positions for the embeddings, by default 10000.
    endpoint : bool, optional
        If True, the embedding considers the endpoint. By default False.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled.
        Defaults to False.
    learnable : bool, optional
        A boolean flag indicating whether learnable positional embedding is enabled.
        Defaults to False.
    freq_embed_dim: int, optional
        The dimension of the frequency embedding. Defaults to None, in which case it
        will be set to num_channels.
    mlp_hidden_dim: int, optional
        The dimension of the hidden layer in the MLP. Defaults to None, in which case
        it will be set to 2 * num_channels.
        Only applicable if learnable is True; if learnable is False, this
        parameter is ignored.
    embed_fn: Literal["cos_sin", "np_sin_cos"], optional
        The function to use for embedding into sin/cos features (allows for swapping the
        order of sin/cos). Defaults to 'cos_sin'.
        Options:
            - 'cos_sin': Uses torch to compute frequency embeddings and
                returns in order (cos, sin)
            - 'np_sin_cos': Uses numpy to compute frequency embeddings and
                returns in order (sin, cos)
    """

    def __init__(
        self,
        num_channels: int,
        max_positions: int = 10000,
        endpoint: bool = False,
        amp_mode: bool = False,
        learnable: bool = False,
        freq_embed_dim: int | None = None,
        mlp_hidden_dim: int | None = None,
        embed_fn: Literal["cos_sin", "np_sin_cos"] = "cos_sin",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.amp_mode = amp_mode
        self.learnable = learnable
        self.embed_fn = embed_fn

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

    def _cos_sin_embedding(self, x):
        freqs = torch.arange(
            start=0, end=self.freq_embed_dim // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.freq_embed_dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if freqs.dtype != x.dtype:
                freqs = freqs.to(x.dtype)
        x = x.ger(freqs)
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

    Architectures include DDPM++, NCSN++, and ADM. Applies GroupNorm followed
    by multi-head self-attention and a projection layer.

    Parameters
    ----------
    out_channels : int
        Number of channels :math:`C` in the input and output feature maps.
    num_heads : int
        Number of attention heads. Must be a positive integer.
    eps : float, optional, default=1e-5
        Epsilon value for numerical stability in GroupNorm.
    init_zero : dict, optional, default={'init_weight': 0}
        Initialization parameters with zero weights for certain layers.
    init_attn : dict, optional, default=None
        Initialization parameters specific to attention mechanism layers.
        Defaults to 'init' if not provided.
    init : dict, optional, default={}
        Initialization parameters for convolutional and linear layers.
    use_apex_gn : bool, optional, default=False
        A boolean flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Need to set this as False on cpu.
    amp_mode : bool, optional, default=False
        A boolean flag indicating whether mixed-precision (AMP) training is enabled.
    fused_conv_bias: bool, optional, default=False
        A boolean flag indicating whether bias will be passed as a parameter of conv2d.


    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, H, W)`, where :math:`B` is batch
        size, :math:`C` is `out_channels`, and :math:`H, W` are spatial
        dimensions.

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
        amp_mode: bool = False,
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
            amp_mode=amp_mode,
        )
        self.qkv = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * 3,
            kernel=1,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **(init_attn if init_attn is not None else init),
        )
        self.proj = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=1,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **init_zero,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1: torch.Tensor = self.qkv(self.norm(x))

        # # NOTE: V1.0.1 implementation
        # q, k, v = x1.reshape(
        #     x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
        # ).unbind(2)
        # w = AttentionOp.apply(q, k)
        # attn = torch.einsum("nqk,nck->ncq", w, v)

        q, k, v = (
            (
                x1.reshape(
                    x.shape[0], self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
            )
            .permute(0, 1, 4, 3, 2)
            .unbind(-2)
        )
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(k.shape[-1])
        )
        attn = attn.transpose(-1, -2)

        x: torch.Tensor = self.proj(attn.reshape(*x.shape)).add_(x)
        return x


class UNetBlock(torch.nn.Module):
    """
    Unified U-Net block with optional up/downsampling and self-attention. Represents
    the union of all features employed by the DDPM++, NCSN++, and ADM architectures.

    Parameters:
    -----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int
        Number of output channels :math:`C_{out}`.
    emb_channels : int
        Number of embedding channels :math:`C_{emb}`.
    up : bool, optional, default=False
        If True, applies upsampling in the forward pass.
    down : bool, optional, default=False
        If True, applies downsampling in the forward pass.
    attention : bool, optional, default=False
        If True, enables the self-attention mechanism in the block.
    num_heads : int, optional, default=None
        Number of attention heads. If None, defaults to :math:`C_{out} / 64`.
    channels_per_head : int, optional, default=64
        Number of channels per attention head.
    dropout : float, optional, default=0.0
        Dropout probability.
    skip_scale : float, optional, default=1.0
        Scale factor applied to skip connections.
    eps : float, optional, default=1e-5
        Epsilon value used for normalization layers.
    resample_filter : List[int], optional, default=``[1, 1]``
        Filter for resampling layers.
    resample_proj : bool, optional, default=False
        If True, resampling projection is enabled.
    adaptive_scale : bool, optional, default=True
        If True, uses adaptive scaling in the forward pass.
    init : dict, optional, default=``{}``
        Initialization parameters for convolutional and linear layers.
    init_zero : dict, optional, default=``{'init_weight': 0}``
        Initialization parameters with zero weights for certain layers.
    init_attn : dict, optional, default=``None``
        Initialization parameters specific to attention mechanism layers.
        Defaults to ``init`` if not provided.
    use_apex_gn : bool, optional, default=False
        A boolean flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Need to set this as False on cpu.
    act : str, optional, default=None
        The activation function to use when fusing activation with GroupNorm.
    fused_conv_bias: bool, optional, default=False
        A boolean flag indicating whether bias will be passed as a parameter of conv2d.
    profile_mode: bool, optional, default=False
        A boolean flag indicating whether to enable all nvtx annotations
        during profiling.
    amp_mode : bool, optional, default=False
        A boolean flag indicating whether mixed-precision (AMP) training is
        enabled.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)`, where :math:`B` is batch
        size, :math:`C_{in}` is ``in_channels``, and :math:`H, W` are spatial
        dimensions.
    emb : torch.Tensor
        Embedding tensor of shape :math:`(B, C_{emb})`, where :math:`B` is batch
        size, and :math:`C_{emb}` is ``emb_channels``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H, W)`, where :math:`B` is batch
        size, :math:`C_{out}` is ``out_channels``, and :math:`H, W` are spatial
        dimensions.
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
        profile_mode: bool = False,
        amp_mode: bool = False,
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
        self.profile_mode = profile_mode
        self.amp_mode = amp_mode
        self.norm0 = get_group_norm(
            num_channels=in_channels,
            eps=eps,
            use_apex_gn=use_apex_gn,
            act=act,
            amp_mode=amp_mode,
        )
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            amp_mode=amp_mode,
            **init,
        )
        if self.adaptive_scale:
            self.norm1 = get_group_norm(
                num_channels=out_channels,
                eps=eps,
                use_apex_gn=use_apex_gn,
                amp_mode=amp_mode,
            )
        else:
            self.norm1 = get_group_norm(
                num_channels=out_channels,
                eps=eps,
                use_apex_gn=use_apex_gn,
                act=act,
                amp_mode=amp_mode,
            )
        self.conv1 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
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
                amp_mode=amp_mode,
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
                amp_mode=amp_mode,
                fused_conv_bias=fused_conv_bias,
            )
        else:
            self.attn = None
        # A hook to migrate legacy attention module
        # Commented out after vendorizing because we do not attempt to load
        # legacy checkpoints, this is only used in SongUNetv2
        #self.register_load_state_dict_pre_hook(self._migrate_attention_module)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(self.norm0(x))
        params = self.affine(emb).unsqueeze(2).unsqueeze(3)
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if params.dtype != x.dtype:
                params = params.to(x.dtype)  # type: ignore

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

        The earliest versions of ``UNetBlock`` stored the attention-layer
        parameters directly on the block using attribute names contained in
        ``_reserved_attributes``.  These have since been moved under the
        dedicated ``attn`` sub-module.  This helper migrates the parameter
        names so that older checkpoints can still be loaded.
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
                # NOTE: Only migrate if destination key not already present to
                # avoid accidental overwriting when both are present.
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
                else:
                    raise ValueError(
                        f"Checkpoint contains both legacy and new keys for {old_key}"
                    )


# ------------------------------------------------------------------------------
# Backbone architectures
# ------------------------------------------------------------------------------


class SongUNetv2(torch.nn.Module):
    r"""
    This architecture is a diffusion backbone for 2D image generation.
    It is a reimplementation of the `DDPM++
    <https://proceedings.mlr.press/v139/nichol21a.html>`_ and
    `NCSN++ <https://arxiv.org/abs/2011.13456>`_
    architectures, which are U-Net variants
    with optional self-attention, embeddings, and encoder-decoder components.

    This model supports conditional and unconditional setups, as well as several
    options for various internal architectural choices such as encoder and decoder
    type, embedding type, etc., making it flexible and adaptable to different tasks
    and configurations.

    This architecture supports conditioning on the noise level (called *noise labels*),
    as well as on additional vector-valued labels (called *class labels*) and (optional)
    vector-valued augmentation labels. The conditioning mechanism relies on addition
    of the conditioning embeddings in the U-Net blocks of the encoder. To condition
    on images, the simplest mechanism is to concatenate the image to the input
    before passing it to the SongUNet.

    The model first applies a mapping operation to generate embeddings for all
    the conditioning inputs (the noise level, the class labels, and the
    optional augmentation labels).

    Then, at each level in the U-Net encoder, a sequence of blocks is applied:

    • A first block downsamples the feature map resolution by a factor of 2
      (odd resolutions are floored). This block does not change the number of
      channels.

    • A sequence of ``num_blocks`` U-Net blocks are applied, each with a different
      number of channels. These blocks do not change the feature map
      resolution, but they multiply the number of channels by a factor
      specified in ``channel_mult``.
      If required, the U-Net blocks also apply self-attention at the specified
      resolutions.

    • At the end of the level, the feature map is cached to be used in a skip
      connection in the decoder.

    The decoder is a mirror of the encoder, with the same number of levels and
    the same number of blocks per level. It multiplies the feature map resolution
    by a factor of 2 at each level.

    Parameters
    -----------
    img_resolution : Union[List[int, int], int]
        The resolution of the input/output image. Can be a single int :math:`H` for
        square images or a list :math:`[H, W]` for rectangular images.

        *Note:* This parameter is only used as a convenience to build the
        network. In practice, the model can still be used with images of
        different resolutions. The only exception to this rule is when
        ``additive_pos_embed`` is True, in which case the resolution of the latent
        state :math:`\mathbf{x}` must match ``img_resolution``.
    in_channels : int
        Number of channels :math:`C_{in}` in the input image.
        May include channels from both
        the latent state and additional channels when conditioning on images.
        For an unconditional model, this should be equal to ``out_channels``.
    out_channels : int
        Number of channels :math:`C_{out}` in the output image. Should be equal
        to the number of channels :math:`C_{\mathbf{x}}` in the latent state.
    label_dim : int, optional, default=0
        Dimension of the vector-valued ``class_labels`` conditioning; 0
        indicates no conditioning on class labels.
    augment_dim : int, optional, default=0
        Dimension of the vector-valued `augment_labels` conditioning; 0 means
        no conditioning on augmentation labels.
    model_channels : int, optional, default=128
        Base multiplier for the number of channels accross the entire network.
    channel_mult : List[int], optional, default=[1, 2, 2, 2]
        Multipliers for the number of channels at every level in
        the encoder and decoder. The length of ``channel_mult`` determines the
        number of levels in the U-Net. At level ``i``, the number of channel in
        the feature map is ``channel_mult[i] * model_channels``.
    channel_mult_emb : int, optional, default=4
        Multiplier for the number of channels in the embedding vector. The
        embedding vector has ``model_channels * channel_mult_emb`` channels.
    num_blocks : int, optional, default=4
        Number of U-Net blocks at each level.
    attn_resolutions : List[int], optional, default=[16]
        Resolutions of the levels at which self-attention layers are applied.
        Note that the feature map resolution must match exactly the value
        provided in `attn_resolutions` for the self-attention layers to be
        applied.
    dropout : float, optional, default=0.10
        Dropout probability applied to intermediate activations within the
        U-Net blocks.
    label_dropout : float, optional, default=0.0
        Dropout probability applied to the `class_labels`. Typically used for
        classifier-free guidance.
    embedding_type : Literal["fourier", "positional", "zero"], optional,
        default="positional"
        Diffusion timestep embedding type: 'positional' for DDPM++, 'fourier'
        for NCSN++, 'zero' for none.
    channel_mult_noise : int, optional, default=1
        Multiplier for the number of channels in the noise level embedding.
        The noise level embedding vector has
        ``model_channels * channel_mult_noise`` channels.
    encoder_type : Literal["standard", "skip", "residual"], optional, default="standard"
        Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++, 'skip'
        for skip connections.
    decoder_type : Literal["standard", "skip"], optional, default="standard"
        Decoder architecture: 'standard' or 'skip' for skip connections.
    resample_filter : List[int], optional, default=[1, 1]
        Resampling filter coefficients applied in the U-Net blocks
        convolutions: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    checkpoint_level : int, optional, default=0
        Number of levels that should use gradient checkpointing. Only levels at
        which the feature map resolution is large enough will be checkpointed
        (0 disables checkpointing, higher values means more layers are checkpointed).
        Higher values trade memory for computation.
    additive_pos_embed : bool, optional, default=False
        If ``True``, adds a learnable positional embedding after the first
        convolution layer. Used in StormCast model.

        *Note:* Those positional embeddings encode spatial position information
        of the image pixels, unlike the ``embedding_type`` parameter which encodes
        temporal information about the diffusion process. In that sense it is a
        simpler version of the positional embedding used in
        :class:`~physicsnemo.models.diffusion_unets.SongUNetPosEmbd`.
    use_apex_gn : bool, optional, default=False
        A flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Apex needs to be installed for this to work. Need to set this as False on cpu.
    act : str, optional, default=None
        The activation function to use when fusing activation with GroupNorm.
        Required when ``use_apex_gn`` is ``True``.
    profile_mode : bool, optional, default=False
        A flag indicating whether to enable all nvtx annotations during
        profiling.
    amp_mode : bool, optional, default=True
        A flag indicating whether mixed-precision (AMP) training is enabled.


    Forward
    -------
    x : torch.Tensor
        The input image of shape :math:`(B, C_{in}, H_{in}, W_{in})`. In
        general ``x`` is the channel-wise concatenation of the latent state
        :math:`\mathbf{x}` and additional images used for conditioning. For an
        unconditional model, ``x`` is simply the latent state
        :math:`\mathbf{x}`.

        *Note:* :math:`H_{in}` and :math:`W_{in}` do not need to match
        :math:`H` and :math:`W` defined in ``img_resolution``, except when
        ``additive_pos_embed`` is ``True``. In that case, the resolution of
        ``x`` must match ``img_resolution``.
    noise_labels : torch.Tensor
        The noise labels of shape :math:`(B,)`. Used for conditioning on
        the diffusion noise level.
    class_labels : torch.Tensor
        The class labels of shape :math:`(B, \text{label_dim})`. Used for
        conditioning on any vector-valued quantity. Can pass ``None`` when
        ``label_dim`` is 0.
    augment_labels : torch.Tensor, optional, default=None
        The augmentation labels of shape :math:`(B, \text{augment_dim})`. Used
        for conditioning on any additional vector-valued quantity. Can pass
        ``None`` when ``augment_dim`` is 0.

    Outputs
    -------
    torch.Tensor
        The denoised latent state of shape :math:`(B, C_{out}, H_{in}, W_{in})`.


    .. important::
        • The terms *noise levels* (or *noise labels*) are used to refer to
          the diffusion time-step, as these are conceptually equivalent.
        • The terms *labels* and *classes* originate from the original paper
          and EDM repository, where this architecture was used for
          class-conditional image generation. While these terms suggest
          class-based conditioning, the architecture can actually be
          conditioned on any vector-valued conditioning.
        • The term *positional embedding* used in the `embedding_type`
          parameter also comes from the original paper and EDM repository.
          Here, *positional* refers to the diffusion time-step, similar to
          how position is used in transformer architectures. Despite the
          name, these embeddings encode temporal information about the
          diffusion process rather than spatial position information.
        • Limitations on input image resolution: for a model that has
          :math:`N` levels, the latent state :math:`\mathbf{x}` must have
          resolution that is a multiple of :math:`2^{N-1}` in each dimension.
          This is due to a limitation in the decoder that does not support
          shape mismatch in the residual connections from the encoder to the
          decoder. For images that do not match this requirement, it is
          recommended to interpolate your data on a grid of the required
          resolution beforehand.

    Example:
    --------
    >>> model = SongUNet(img_resolution=16, in_channels=2, out_channels=2)
    >>> noise_labels = torch.randn([1])
    >>> class_labels = torch.randint(0, 1, (1, 1))
    >>> input_image = torch.ones([1, 2, 16, 16])
    >>> output_image = model(input_image, noise_labels, class_labels)
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])
    """

    # Arguments of the __init__ method that can be overridden with the
    # ``Module.from_checkpoint`` method.
    _overridable_args: set[str] = {"use_apex_gn", "act"}

    def __init__(
        self,
        img_resolution: list[int] | int,
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: list[int] = [1, 2, 2, 2],
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: list[int] = [16],
        dropout: float = 0.10,
        label_dropout: float = 0.0,
        embedding_type: Literal["fourier", "positional", "zero"] = "positional",
        channel_mult_noise: int = 1,
        encoder_type: Literal["standard", "skip", "residual"] = "standard",
        decoder_type: Literal["standard", "skip"] = "standard",
        resample_filter: list[int] = [1, 1],
        checkpoint_level: int = 0,
        additive_pos_embed: bool = False,
        use_apex_gn: bool = True,
        act: str = "silu",
        profile_mode: bool = False,
        amp_mode: bool = True,
    ):
        valid_embedding_types = ["fourier", "positional", "zero"]
        if embedding_type not in valid_embedding_types:
            raise ValueError(
                f"Invalid embedding_type: {embedding_type}. "
                f"Must be one of {valid_embedding_types}."
            )

        valid_encoder_types = ["standard", "skip", "residual"]
        if encoder_type not in valid_encoder_types:
            raise ValueError(
                f"Invalid encoder_type: {encoder_type}. "
                f"Must be one of {valid_encoder_types}."
            )

        valid_decoder_types = ["standard", "skip"]
        if decoder_type not in valid_decoder_types:
            raise ValueError(
                f"Invalid decoder_type: {decoder_type}. "
                f"Must be one of {valid_decoder_types}."
            )

        super().__init__()
        self.label_dropout = label_dropout
        self.embedding_type = embedding_type
        emb_channels = model_channels * channel_mult_emb
        self.emb_channels = emb_channels
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=0.7071067811865476,  # 1 / sqrt(2)
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
            use_apex_gn=use_apex_gn,
            act=act,
            fused_conv_bias=True,
            profile_mode=profile_mode,
            amp_mode=amp_mode,
        )
        self.use_apex_gn = use_apex_gn

        # for compatibility with older versions that took only 1 dimension
        self.img_resolution = img_resolution
        if isinstance(img_resolution, int):
            self.img_shape_y = self.img_shape_x = img_resolution
        else:
            self.img_shape_y = img_resolution[0]
            self.img_shape_x = img_resolution[1]

        self._num_levels = len(channel_mult)
        self._input_shape_mult = 2 ** (self._num_levels - 1)

        # set the threshold for checkpointing based on image resolution
        self.checkpoint_threshold = (
            math.floor(math.sqrt(self.img_shape_x * self.img_shape_y))
            >> checkpoint_level
        ) + 1

        # Optional additive learned positition embed after the first conv
        self.additive_pos_embed = additive_pos_embed
        if self.additive_pos_embed:
            self.spatial_emb = torch.nn.Parameter(
                torch.randn(1, model_channels, self.img_shape_y, self.img_shape_x)
            )
            torch.nn.init.trunc_normal_(self.spatial_emb, std=0.02)

        # Mapping.
        if self.embedding_type != "zero":
            self.map_noise = (
                PositionalEmbedding(num_channels=noise_channels, endpoint=True)
                if embedding_type == "positional"
                else FourierEmbedding(num_channels=noise_channels)
            )
            self.map_label = (
                Linear(
                    in_features=label_dim,
                    out_features=noise_channels,
                    **init,
                )
                if label_dim
                else None
            )
            self.map_augment = (
                Linear(
                    in_features=augment_dim,
                    out_features=noise_channels,
                    bias=False,
                    **init,
                )
                if augment_dim
                else None
            )
            self.map_layer0 = Linear(
                in_features=noise_channels,
                out_features=emb_channels,
                **init,
            )
            self.map_layer1 = Linear(
                in_features=emb_channels,
                out_features=emb_channels,
                **init,
            )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = self.img_shape_y >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin,
                    out_channels=cout,
                    kernel=3,
                    fused_conv_bias=True,
                    **init,
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=1,
                        fused_conv_bias=True,
                        **init,
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        fused_conv_bias=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = self.img_shape_y >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = get_group_norm(
                    num_channels=cout,
                    eps=1e-6,
                    use_apex_gn=use_apex_gn,
                )
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    in_channels=cout,
                    out_channels=out_channels,
                    kernel=3,
                    fused_conv_bias=True,
                    **init_zero,
                )

        # Set properties recursively on submodules
        self.profile_mode = profile_mode
        self.amp_mode = amp_mode
        if self.use_apex_gn:
            self.to(memory_format=torch.channels_last)

    # Properties that are recursively set on submodules
    profile_mode = _recursive_property(
        "profile_mode", bool, "Should be set to ``True`` to enable profiling."
    )
    amp_mode = _recursive_property(
        "amp_mode",
        bool,
        "Should be set to ``True`` to enable automatic mixed precision.",
    )

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Validate input shapes
        batch_size = x.shape[0]

        if x.ndim != 4:
            raise ValueError(
                f"Expected 'x' to be a 4D tensor, "
                f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
            )

        # Check spatial dimensions are powers of 2 or multiples of 2^{N-1}
        for d in x.shape[-2:]:
            # Check if d is a power of 2
            is_power_of_2 = (d & (d - 1)) == 0 and d > 0
            # If not power of 2, must be multiple of self._input_shape_mult
            if not (
                (is_power_of_2 and d < self._input_shape_mult)
                or (d % self._input_shape_mult == 0)
            ):
                raise ValueError(
                    f"Input spatial dimensions ({x.shape[-2:]}) must be "
                    f"either powers of 2 or multiples of 2**(N-1) where "
                    f"N (={self._num_levels}) is the number of levels "
                    f"in the U-Net."
                )

        # TODO: noise_labels of shape (1,) means that all inputs share the
        # same noise level. This should be removed in the future, though.
        if noise_labels.ndim != 1 or noise_labels.shape[0] not in (batch_size, 1):
            raise ValueError(
                f"Expected 'noise_labels' shape ({batch_size},) or (1,), "
                f"got {tuple(noise_labels.shape)}"
            )

        if class_labels is not None and (
            class_labels.ndim != 2 or class_labels.shape[0] != batch_size
        ):
            raise ValueError(
                f"Expected 'class_labels' shape ({batch_size}, C), "
                f"got {tuple(class_labels.shape)}"
            )

        if augment_labels is not None and (
            augment_labels.ndim != 2 or augment_labels.shape[0] != batch_size
        ):
            raise ValueError(
                f"Expected 'augment_labels' shape ({batch_size}, C), "
                f"got {tuple(augment_labels.shape)}"
            )

        if (
            self.use_apex_gn
            and (not x.is_contiguous(memory_format=torch.channels_last))
            and x.dim() == 4
        ):
            x = x.to(memory_format=torch.channels_last)
        if self.embedding_type != "zero":
            # Mapping.
            emb = self.map_noise(noise_labels)
            emb = (
                emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
            )  # swap sin/cos
            if self.map_label is not None:
                tmp = class_labels
                if self.training and self.label_dropout:
                    tmp = tmp * (
                        torch.rand([x.shape[0], 1], device=x.device)
                        >= self.label_dropout
                    ).to(tmp.dtype)
                emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
            if self.map_augment is not None and augment_labels is not None:
                emb = emb + self.map_augment(augment_labels)
            emb = silu(self.map_layer0(emb))
            emb = silu(self.map_layer1(emb))
        else:
            emb = torch.zeros(
                (noise_labels.shape[0], self.emb_channels),
                device=x.device,
                dtype=x.dtype,
            )

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            elif "_conv" in name:
                x = block(x)
                if self.additive_pos_embed:
                    x = x + self.spatial_emb.to(dtype=x.dtype)
                skips.append(x)
            else:
                # For UNetBlocks check if we should use gradient checkpointing
                if isinstance(block, UNetBlock):
                    if (
                        math.floor(math.sqrt(x.shape[-2] * x.shape[-1]))
                        > self.checkpoint_threshold
                    ):
                        # self.checkpoint = checkpoint?
                        # else: self.checkpoint  = lambda(block,x,emb:block(x,emb))
                        x = checkpoint(block, x, emb, use_reentrant=False)
                    else:
                        # AssertionError: Only support NHWC layout.
                        x = block(x, emb)
                else:
                    x = block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                # check for checkpointing on decoder blocks and up sampling blocks
                if (
                    math.floor(math.sqrt(x.shape[-2] * x.shape[-1]))
                    > self.checkpoint_threshold
                    and "_block" in name
                ) or (
                    math.floor(math.sqrt(x.shape[-2] * x.shape[-1]))
                    > (self.checkpoint_threshold / 2)
                    and "_up" in name
                ):
                    x = checkpoint(block, x, emb, use_reentrant=False)
                else:
                    x = block(x, emb)
        return aux

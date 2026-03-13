"""
Performance-optimized version of physicsnemo_unets_v2/unets.py.

Changes from v2:
- Removed amp_mode and profile_mode parameters and _recursive_property mechanism
- Uses swap_sin_cos=True on PositionalEmbedding to avoid reshape+flip in forward
- Uses pre-computed checkpoint_threshold consistently (no re-computation of sqrt)
- All layer constructors no longer receive amp_mode or profile_mode

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
from typing import Literal

import numpy as np
import torch
from torch.nn.functional import silu
from torch.utils.checkpoint import checkpoint

from fme.core.benchmark.timer import NullTimer, Timer
from fme.downscaling.modules.utils import check_level_compatibility, validate_shape

from .group_norm import get_group_norm
from .layers import Conv2d, FourierEmbedding, Linear, PositionalEmbedding, UNetBlock


# ------------------------------------------------------------------------------
# Backbone architectures
# ------------------------------------------------------------------------------


class SongUNetv3(torch.nn.Module):
    r"""
    Performance-optimized version of SongUNetv2.

    This architecture is a diffusion backbone for 2D image generation.
    It is a reimplementation of the `DDPM++
    <https://proceedings.mlr.press/v139/nichol21a.html>`_ and
    `NCSN++ <https://arxiv.org/abs/2011.13456>`_
    architectures, which are U-Net variants
    with optional self-attention, embeddings, and encoder-decoder components.

    Parameters
    -----------
    img_resolution : Union[List[int, int], int]
        The resolution of the input/output image.
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    label_dim : int, optional, default=0
        Dimension of the vector-valued ``class_labels`` conditioning.
    augment_dim : int, optional, default=0
        Dimension of the vector-valued `augment_labels` conditioning.
    model_channels : int, optional, default=128
        Base multiplier for the number of channels.
    channel_mult : List[int], optional, default=[1, 2, 2, 2]
        Multipliers for the number of channels at every level.
    channel_mult_emb : int, optional, default=4
        Multiplier for the number of channels in the embedding vector.
    num_blocks : int, optional, default=4
        Number of U-Net blocks at each level.
    attn_resolutions : List[int], optional, default=[16]
        Resolutions at which self-attention layers are applied.
    dropout : float, optional, default=0.10
        Dropout probability.
    label_dropout : float, optional, default=0.0
        Dropout probability applied to the `class_labels`.
    embedding_type : Literal["fourier", "positional", "zero"], optional,
        default="positional"
        Diffusion timestep embedding type.
    channel_mult_noise : int, optional, default=1
        Multiplier for noise level embedding channels.
    encoder_type : Literal["standard", "skip", "residual"], optional, default="standard"
        Encoder architecture type.
    decoder_type : Literal["standard", "skip"], optional, default="standard"
        Decoder architecture type.
    resample_filter : List[int], optional, default=[1, 1]
        Resampling filter coefficients.
    checkpoint_level : int, optional, default=0
        Number of levels using gradient checkpointing.
    additive_pos_embed : bool, optional, default=False
        If True, adds a learnable positional embedding after the first conv.
    use_apex_gn : bool, optional, default=False
        Whether to use Apex GroupNorm for NHWC layout.
    act : str, optional, default=None
        Activation function for fusing with GroupNorm.

    Forward
    -------
    x : torch.Tensor
        Input image of shape :math:`(B, C_{in}, H_{in}, W_{in})`.
    noise_labels : torch.Tensor
        Noise labels of shape :math:`(B,)`.
    class_labels : torch.Tensor
        Class labels of shape :math:`(B, label_dim)`.
    augment_labels : torch.Tensor, optional
        Augmentation labels of shape :math:`(B, augment_dim)`.

    Outputs
    -------
    torch.Tensor
        Denoised latent state of shape :math:`(B, C_{out}, H_{in}, W_{in})`.
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
        check_img_resolution = (
            min(img_resolution) if isinstance(img_resolution, list) else img_resolution
        )
        check_level_compatibility(check_img_resolution, channel_mult, attn_resolutions)

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
        self.channel_mult = channel_mult

        # set the threshold for checkpointing based on image resolution
        self.checkpoint_threshold = (
            math.floor(math.sqrt(self.img_shape_x * self.img_shape_y))
            >> checkpoint_level
        ) + 1

        # Optional additive learned position embed after the first conv
        self.additive_pos_embed = additive_pos_embed
        if self.additive_pos_embed:
            self.spatial_emb = torch.nn.Parameter(
                torch.randn(1, model_channels, self.img_shape_y, self.img_shape_x)
            )
            torch.nn.init.trunc_normal_(self.spatial_emb, std=0.02)

        # Mapping.
        if self.embedding_type != "zero":
            self.map_noise = (
                PositionalEmbedding(
                    num_channels=noise_channels,
                    endpoint=True,
                    swap_sin_cos=True,
                )
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

        if self.use_apex_gn:
            self.to(memory_format=torch.channels_last)

    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        augment_labels=None,
        timer: Timer = NullTimer(),
    ):
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
        with timer.child("mapping"):
            if self.embedding_type != "zero":
                # Mapping - no sin/cos swap needed, PositionalEmbedding
                # already returns (sin, cos) via swap_sin_cos=True
                emb = self.map_noise(noise_labels)
                if self.map_label is not None:
                    tmp = class_labels
                    if self.training and self.label_dropout:
                        tmp = tmp * (
                            torch.rand([x.shape[0], 1], device=x.device)
                            >= self.label_dropout
                        ).to(tmp.dtype)
                    emb = emb + self.map_label(
                        tmp * np.sqrt(self.map_label.in_features)
                    )
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
        with timer.child("encoder") as enc_timer:
            skips = []
            aux = x

            for name, block in self.enc.items():
                with enc_timer.child(name.split("_", 1)[0]) as level_timer:
                    if "aux_down" in name:
                        with level_timer.child("down"):
                            aux = block(aux)
                    elif "aux_skip" in name:
                        with level_timer.child("skip"):
                            x = skips[-1] = x + block(aux)
                    elif "aux_residual" in name:
                        with level_timer.child("residual"):
                            x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
                    elif "_conv" in name:
                        with level_timer.child("conv"):
                            x = block(x)
                            if self.additive_pos_embed:
                                x = x + self.spatial_emb.to(dtype=x.dtype)
                            skips.append(x)
                    else:
                        with level_timer.child("block"):
                            if isinstance(block, UNetBlock):
                                if (
                                    math.floor(
                                        math.sqrt(x.shape[-2] * x.shape[-1])
                                    )
                                    > self.checkpoint_threshold
                                ):
                                    x = checkpoint(
                                        block, x, emb, use_reentrant=False
                                    )
                                else:
                                    x = block(x, emb)
                            else:
                                x = block(x)
                            skips.append(x)

        # Decoder.
        with timer.child("decoder") as dec_timer:
            aux = None
            tmp = None
            for name, block in self.dec.items():
                with dec_timer.child(name.split("_", 1)[0]) as level_timer:
                    if "aux_up" in name:
                        with level_timer.child("up"):
                            aux = block(aux)
                    elif "aux_norm" in name:
                        with level_timer.child("norm"):
                            tmp = block(x)
                    elif "aux_conv" in name:
                        with level_timer.child("conv"):
                            tmp = block(silu(tmp))
                            aux = tmp if aux is None else tmp + aux
                    else:
                        timer_name = "up" if "_up" in name else "block"
                        with level_timer.child(timer_name):
                            if x.shape[1] != block.in_channels:
                                x = torch.cat([x, skips.pop()], dim=1)

                            if (
                                math.floor(
                                    math.sqrt(x.shape[-2] * x.shape[-1])
                                )
                                > self.checkpoint_threshold
                                and "_block" in name
                            ) or (
                                math.floor(
                                    math.sqrt(x.shape[-2] * x.shape[-1])
                                )
                                > (self.checkpoint_threshold / 2)
                                and "_up" in name
                            ):
                                x = checkpoint(
                                    block, x, emb, use_reentrant=False
                                )
                            else:
                                x = block(x, emb)
        return aux

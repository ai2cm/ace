from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRAConv2d(nn.Conv2d):
    """
    Drop-in Conv2d with optional LoRA.

    - API matches torch.nn.Conv2d, with extra args:
        lora_rank: int = 0        (0 disables LoRA)
        lora_alpha: float = None  (defaults to lora_rank)
        lora_dropout: float = 0.0

    - Can load a checkpoint saved from nn.Conv2d even when lora_rank > 0
      (i.e., state_dict only has "weight"/"bias").
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        *,
        lora_rank: int = 0,
        lora_alpha: float | None = None,
        lora_dropout: float = 0.0,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.lora_down: nn.Conv2d | None = None
        self.lora_up: nn.Conv2d | None = None
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

        if lora_rank < 0:
            raise ValueError(f"lora_rank must be >= 0, got {lora_rank}")
        if lora_dropout < 0.0:
            raise ValueError(f"lora_dropout must be >= 0, got {lora_dropout}")

        self.lora_rank = int(lora_rank)
        self.lora_alpha = (
            float(lora_alpha) if lora_alpha is not None else float(lora_rank)
        )
        self.lora_dropout_p = float(lora_dropout)

        self._lora_merged = False

        if self.lora_rank > 0:
            # Group-compatible LoRA via two convs:
            #   down: 1x1 grouped conv: in_channels -> (groups * r), groups=groups
            #   up:   kxk grouped conv: (groups * r) -> out_channels, groups=groups
            # This produces a delta with the same grouped structure as the base conv.
            mid_channels = self.groups * self.lora_rank

            self.lora_down = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=self.groups,
                bias=False,
                **factory_kwargs,
            )
            self.lora_up = nn.Conv2d(
                in_channels=mid_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=False,
                padding_mode=self.padding_mode,
                **factory_kwargs,
            )

            self.lora_dropout = (
                nn.Dropout(p=self.lora_dropout_p)
                if self.lora_dropout_p > 0
                else nn.Identity()
            )

            # Scaling as in LoRA: alpha / r
            self.lora_scaling = self.lora_alpha / float(self.lora_rank)
        else:
            self.lora_dropout = nn.Identity()
            self.lora_scaling = 0.0
        self.reset_lora_parameters()  # base parameters already reset in super init

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        # Init: down ~ Kaiming, up = 0 so the module starts
        # identical to base Conv2d.
        if self.lora_down is not None:
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        if self.lora_up is not None:
            nn.init.zeros_(self.lora_up.weight)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        if self.lora_rank > 0:
            return (
                f"{base}, lora_rank={self.lora_rank}, lora_alpha={self.lora_alpha}, "
                f"lora_dropout={self.lora_dropout_p}, lora_merged={self._lora_merged}"
            )
        return f"{base}, lora_rank=0"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self.lora_rank == 0 or self._lora_merged:
            return y
        assert self.lora_down is not None and self.lora_up is not None
        return (
            y + self.lora_up(self.lora_down(self.lora_dropout(x))) * self.lora_scaling
        )

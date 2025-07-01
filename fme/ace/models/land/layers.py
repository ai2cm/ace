import torch
import torch.nn as nn


class LearnedPositionalEmbedding2D(nn.Module):
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, channels, height, width))

    def forward(self, x):
        # x: [B, C, H, W]
        return x + self.pos_embed


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: torch.nn.Module = nn.ReLU,
    ):
        super().__init__()

        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=input_dim, out_channels=output_dim, kernel_size=(1, 1)
            )
        )
        layers.append(activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

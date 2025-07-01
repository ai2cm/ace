from typing import Literal

import torch
import torch.nn as nn

from fme.ace.models.land.layers import MLP, LearnedPositionalEmbedding2D


class LandNet(nn.Module):
    """LandNet architecture.

    This is a simple feedforward network that can be used for land
    model emulating.

    Example:
    -------
    >>> import torch
    >>> model = LandNet(
    ...     img_shape=(128, 128),
    ...     input_channels=4,
    ...     hidden_dims=[64,64],
    ...     output_channels=3,
    ...     network_type="MLP",
    ... )
    >>> x = torch.randn(1, 4, 128, 128)  # Example input
    >>> out = model(x)
    >>> print(out.shape)
    torch.Size([1, 3, 128, 128])
    """

    def __init__(
        self,
        img_shape: tuple,
        input_channels: int,
        hidden_dims: list[int],
        output_channels: int,
        network_type: Literal["MLP"] = "MLP",
        activation: torch.nn.Module = nn.ReLU,
        use_positional_embedding: bool = False,
    ):
        super().__init__()
        self.use_positional_embedding = use_positional_embedding

        # positional embedding layer 2
        if self.use_positional_embedding:
            self.pos_embed = LearnedPositionalEmbedding2D(
                hidden_dims[0], img_shape[0], img_shape[1]
            )

        self.num_layers = len(hidden_dims)

        self.layers = []
        prev_dim = input_channels

        if network_type == "MLP":
            layer = MLP
        else:
            raise ValueError(f"Unsupported network type: {network_type}")

        for hidden_dim in hidden_dims:
            self.layers.append(
                layer(
                    prev_dim,
                    hidden_dim,
                    activation=activation,
                )
            )

            prev_dim = hidden_dim

        self.layers.append(
            layer(
                prev_dim,
                output_channels,
                activation=nn.Identity,  # No activation for the last layer
            )
        )

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        for layer_count, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_positional_embedding and layer_count == 0:
                x = self.pos_embed(x)
        return x

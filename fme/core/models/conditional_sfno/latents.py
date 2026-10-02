"""A small wrapper over a two-track ``[global | local]`` latent tensor.

The two-track SFNO carries a single concatenated latent whose leading
``global`` channels participate in the spherical (spectral) operator and
whose trailing ``local`` channels only ever see pointwise operations. Most
sublayers (norms, MLP, skip connections) act on the full concatenation and
keep taking a plain ``Tensor``; ``Latents`` exists so the two places that do
care about the split -- the spectral filter call site and the parallel
``conv1x1`` -- can address the global and local parts without threading a raw
channel-count integer through the block. The split index is a construction
detail supplied to the constructors; it is never exposed on the accessor
surface, which returns plain tensors.
"""

import torch


class Latents:
    """A ``[global | local]`` latent, hiding the concat and channel counts."""

    def __init__(self, global_channels: torch.Tensor, local_channels: torch.Tensor):
        if global_channels.shape[:-3] != local_channels.shape[:-3]:
            raise ValueError(
                "global and local tracks must share leading (batch) dimensions, "
                f"got {tuple(global_channels.shape)} and {tuple(local_channels.shape)}"
            )
        if global_channels.shape[-2:] != local_channels.shape[-2:]:
            raise ValueError(
                "global and local tracks must share spatial dimensions, "
                f"got {tuple(global_channels.shape)} and {tuple(local_channels.shape)}"
            )
        self._global = global_channels
        self._local = local_channels

    @classmethod
    def new_from_all(cls, tensor: torch.Tensor, global_channels: int) -> "Latents":
        """Build from a full ``[global | local]`` tensor split at ``global_channels``.

        Args:
            tensor: concatenated latent of shape ``(..., global + local, H, W)``.
            global_channels: number of leading channels in the global track.
        """
        return cls(
            tensor[..., :global_channels, :, :],
            tensor[..., global_channels:, :, :],
        )

    @classmethod
    def new_from_global(
        cls, global_tensor: torch.Tensor, local_channels: int
    ) -> "Latents":
        """Build from a global-only tensor, padding the local track with zeros.

        Args:
            global_tensor: the global track, shape ``(..., global, H, W)``.
            local_channels: width of the (zero) local track to attach.
        """
        local_shape = (
            *global_tensor.shape[:-3],
            local_channels,
            *global_tensor.shape[-2:],
        )
        local = global_tensor.new_zeros(local_shape)
        return cls(global_tensor, local)

    @property
    def global_channels(self) -> torch.Tensor:
        return self._global

    @property
    def local_channels(self) -> torch.Tensor:
        return self._local

    @property
    def all(self) -> torch.Tensor:
        return torch.cat([self._global, self._local], dim=-3)

    def __add__(self, other: "Latents") -> "Latents":
        return Latents(
            self._global + other._global,
            self._local + other._local,
        )

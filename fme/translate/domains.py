"""Domains: the named spaces that translate components map between.

A *domain* is a named space where data (or latent state) lives: an ordered set
of channels on one grid. Domains are the pairing key between components and
data: transforms declare an input and an output domain, backbones declare the
domain they step in, and data streams (added with the data-loading PR) declare
the domain they provide. Building a pool binds each domain to a
:class:`DatasetInfo` by name.

Channels are declared as a union of plain variable names and *latent-channel
blocks*: a block names a group of free channels once and says how many there
are (e.g. ``LatentChannels(name="z", channels=512)``) rather than enumerating
them. Blocks expand to indexed names (``z_0`` ... ``z_511``) so that
everything downstream (checkpoint name-matching, diagnostics) sees an ordinary
ordered list of channel names.

A domain with no dataset behind it (a latent space) declares ``grid_like``,
naming the domain whose grid it shares — per the multi-resolution design, the
latent lives on the backbone's (coarsest) grid.
"""

import dataclasses

__all__ = ["DomainConfig", "LatentChannels"]


@dataclasses.dataclass
class LatentChannels:
    """A named block of latent channels.

    Parameters:
        name: The block name; expanded channel names are ``{name}_{i}``.
        channels: The number of channels in the block.
    """

    name: str
    channels: int

    def __post_init__(self):
        if not self.name:
            raise ValueError("LatentChannels requires a non-empty name.")
        if self.channels < 1:
            raise ValueError(
                f"LatentChannels {self.name!r} requires channels >= 1, "
                f"got {self.channels}."
            )

    @property
    def names(self) -> list[str]:
        return [f"{self.name}_{i}" for i in range(self.channels)]


@dataclasses.dataclass
class DomainConfig:
    """A named space where data or latent state lives.

    Parameters:
        channels: Ordered channel declaration: a union of variable names and
            latent-channel blocks. Order defines channel order for every
            component built against this domain.
        grid_like: For a domain with no dataset behind it (a latent space),
            the name of the domain whose grid it shares. Domains without
            ``grid_like`` must be bound to a ``DatasetInfo`` (i.e. paired
            with data) at pool build time.
    """

    channels: list[str | LatentChannels]
    grid_like: str | None = None

    def __post_init__(self):
        if not self.channels:
            raise ValueError("A domain must declare at least one channel.")
        names = self.names
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"Domain channel names must be unique after latent-block "
                f"expansion; found duplicates: {sorted(duplicates)}."
            )

    @property
    def names(self) -> list[str]:
        """The ordered channel names, with latent blocks expanded."""
        result: list[str] = []
        for entry in self.channels:
            if isinstance(entry, LatentChannels):
                result.extend(entry.names)
            else:
                result.append(entry)
        return result

    @property
    def n_channels(self) -> int:
        return len(self.names)

"""Uneven-shard-safe replacement for torch_harmonics' internal ``_gather``.

Upstream ``_gather`` calls ``all_gather`` with unequal shard sizes, which
dynamo cannot trace and gloo rejects. This pads shards to equal size, gathers,
and trims; equal shards behave as upstream.

Based on torch-harmonics 0.8.0 ``_gather`` (BSD-3-Clause, Copyright (c) 2022
The torch-harmonics Authors), with pad-and-trim added for uneven shards; the
0.8.0 version does not pad. Recheck on pin bumps: as of 0.9.3 upstream pads
only for backends that need equal shapes, so NCCL under dynamo still breaks.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_harmonics.distributed.primitives as thd_primitives

_original_gather = None


def _padded_gather(
    input_: torch.Tensor,
    dim_: int,
    shapes_: list[int] | None,
    group=None,
) -> torch.Tensor:
    """``_gather`` that pads uneven shards; ``shapes_`` is per-rank sizes or None."""
    comm_size = dist.get_world_size(group=group)
    if shapes_ is not None and len(shapes_) != comm_size:
        raise ValueError(
            f"Expected {comm_size} shard sizes, got {len(shapes_)}: {shapes_}"
        )
    if dim_ >= input_.dim():
        raise ValueError(
            f"Gather dim {dim_} out of range for a {input_.dim()}-d tensor"
        )
    if comm_size == 1:
        return input_
    input_ = input_.contiguous()
    if shapes_ is None or len(set(shapes_)) == 1:
        pieces = [torch.empty_like(input_) for _ in range(comm_size)]
        dist.all_gather(pieces, input_, group=group)
        return torch.cat(pieces, dim=dim_)
    dim = dim_ % input_.dim()
    # F.pad takes (before, after) pairs starting from the last dimension
    pad = [0, 0] * (input_.dim() - 1 - dim) + [0, max(shapes_) - input_.shape[dim]]
    padded = F.pad(input_, pad).contiguous()
    pieces = [torch.empty_like(padded) for _ in range(comm_size)]
    dist.all_gather(pieces, padded, group=group)
    return torch.cat(
        [piece.narrow(dim, 0, size) for piece, size in zip(pieces, shapes_)],
        dim=dim,
    )


def patch_uneven_gather() -> None:
    """Replace torch-harmonics' ``_gather`` with :func:`_padded_gather`; idempotent."""
    global _original_gather
    if _original_gather is not None:
        return  # already patched
    _original_gather = thd_primitives._gather
    thd_primitives._gather = _padded_gather

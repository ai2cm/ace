"""Uneven-shard-safe replacement for torch_harmonics' internal ``_gather``.

``torch_harmonics.distributed.primitives._gather`` gathers shards whose sizes
along ``dim_`` may differ across ranks (e.g. 45 latitudes split over 2 ranks
into 23 and 22 rows) by calling ``torch.distributed.all_gather`` with
unequal tensor sizes. It does so in the backward of
``scatter_to_polar_region``, used by the distributed DISCO convolution, and in
``gather_from_polar_region``. Unequal sizes fail in two ways:

* under ``torch.compile``, dynamo rewrites ``all_gather`` to a functional
  collective that assumes equal shapes, so tracing fails with
  ``expand: attempting to expand a dimension of length 23 -> 22``;
* the gloo backend rejects unequal sizes even when running eagerly.

The replacement pads each shard to the largest size along ``dim_``, gathers
equal shapes, then trims each piece back to its size before concatenating,
the same padding torch-harmonics itself applies for gloo in later releases.
When all shards have equal sizes it behaves exactly like the original.

Copied and adapted from torch-harmonics 0.8.0 (the version pinned in
pyproject.toml). Recheck this patch whenever that pin is bumped: as of
torch-harmonics 0.9.3, upstream pads only for backends that need equal shapes,
so NCCL under dynamo still takes the unequal path.
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
    """Gather ``input_`` along ``dim_`` from all ranks of ``group``.

    Semantics mirror torch-harmonics 0.8.0's ``_gather``: ``shapes_`` holds
    the size along ``dim_`` of each rank's shard, or is None when all shards
    have the same shape.
    """
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
    """Replace torch-harmonics' ``_gather`` with :func:`_padded_gather`.

    Safe to call multiple times; only the first call installs the patch.
    """
    global _original_gather
    if _original_gather is not None:
        return  # already patched
    _original_gather = thd_primitives._gather
    thd_primitives._gather = _padded_gather

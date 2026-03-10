"""Gloo-compatible fallback for torch.distributed.all_to_all.

The gloo backend does not support ``all_to_all``, but
``torch_harmonics.distributed`` relies on it for distributed spectral
transforms.  This module provides a drop-in replacement implemented with
point-to-point operations (``send`` / ``recv``), and a helper that
monkey-patches ``torch.distributed.all_to_all`` so the fallback is used
transparently whenever gloo is the active backend for a given process
group.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

_original_all_to_all = None


def _gloo_all_to_all(
    output_tensor_list: list[torch.Tensor],
    input_tensor_list: list[torch.Tensor],
    group=None,
    async_op: bool = False,
):
    """Implement ``all_to_all`` semantics using ``send`` / ``recv``.

    Semantics mirror ``torch.distributed.all_to_all``: rank *r* sends
    ``input_tensor_list[i]`` to rank *i* and receives into
    ``output_tensor_list[i]`` from rank *i*.

    Uses synchronous point-to-point ops with a rank-ordering scheme to
    avoid deadlocks.  ``send`` / ``recv`` accept global ranks and
    translate them internally, unlike ``isend`` / ``irecv`` which can
    cause hangs when group-local ranks differ from global ranks.

    Note: ``async_op=True`` is not supported; this implementation is
    always synchronous and always returns ``None``.
    """
    if async_op:
        raise NotImplementedError("_gloo_all_to_all does not support async_op=True")
    rank = dist.get_rank(group=group)
    size = dist.get_world_size(group=group)

    # Map group-local ranks → global ranks (send/recv expect global).
    group_ranks = (
        dist.get_process_group_ranks(group) if group is not None else list(range(size))
    )

    # Self-copy (no communication needed).
    output_tensor_list[rank].copy_(input_tensor_list[rank])

    # Exchange with every other rank using synchronous send/recv.
    # Lower-ranked process sends first to avoid deadlock.
    for i in range(size):
        if i == rank:
            continue
        global_peer = group_ranks[i]
        if rank < i:
            dist.send(input_tensor_list[i].contiguous(), dst=global_peer, group=group)
            dist.recv(output_tensor_list[i], src=global_peer, group=group)
        else:
            dist.recv(output_tensor_list[i], src=global_peer, group=group)
            dist.send(input_tensor_list[i].contiguous(), dst=global_peer, group=group)

    return None


def patch_gloo_alltoall() -> None:
    """Monkey-patch ``torch.distributed.all_to_all`` with a gloo-safe wrapper.

    The wrapper checks the backend of the process group at call time:
    * **gloo** → delegates to :func:`_gloo_all_to_all` (send/recv).
    * **anything else** → delegates to the original ``all_to_all``.

    Safe to call multiple times; only the first call installs the patch.
    """
    global _original_all_to_all
    if _original_all_to_all is not None:
        return  # already patched

    _original_all_to_all = dist.all_to_all

    def _patched_all_to_all(
        output_tensor_list,
        input_tensor_list,
        group=None,
        async_op=False,
    ):
        backend = dist.get_backend(group)
        if backend == "gloo" and async_op:
            raise NotImplementedError(
                "all_to_all with async_op=True is not supported on the gloo backend"
            )
        if backend == "gloo":
            return _gloo_all_to_all(
                output_tensor_list,
                input_tensor_list,
                group=group,
                async_op=async_op,
            )
        return _original_all_to_all(
            output_tensor_list,
            input_tensor_list,
            group=group,
            async_op=async_op,
        )

    dist.all_to_all = _patched_all_to_all

"""Traceable, gloo-compatible replacement for torch.distributed.all_to_all.

torch_harmonics' spatial transposes use list-form ``all_to_all``, which gloo
does not implement and dynamo cannot trace (a graph break per transpose). The
functional ``all_to_all_single`` has neither problem.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

_original_all_to_all = None


def _real_view(tensor: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(tensor) if tensor.is_complex() else tensor


def _all_to_all_via_single(
    output_tensor_list: list[torch.Tensor],
    input_tensor_list: list[torch.Tensor],
    group: dist.ProcessGroup | None = None,
) -> None:
    """``all_to_all`` semantics via one flattened ``all_to_all_single``."""
    if group is None:
        group = dist.group.WORLD
    # gloo does not exchange complex tensors, so exchange their real views
    input_tensor_list = [_real_view(t) for t in input_tensor_list]
    output_tensor_list = [_real_view(t) for t in output_tensor_list]
    input_split_sizes = [t.numel() for t in input_tensor_list]
    output_split_sizes = [t.numel() for t in output_tensor_list]
    received = funcol.all_to_all_single(
        torch.cat([t.reshape(-1) for t in input_tensor_list]),
        output_split_sizes,
        input_split_sizes,
        group,
    )
    for output, chunk in zip(
        output_tensor_list, torch.split(received, output_split_sizes)
    ):
        output.copy_(chunk.view(output.shape))


def patch_all_to_all() -> None:
    """Patch ``dist.all_to_all`` for synchronous calls; idempotent.

    ``async_op=True`` calls still go to the original (unsupported on gloo).
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
        if async_op:
            return _original_all_to_all(
                output_tensor_list,
                input_tensor_list,
                group=group,
                async_op=async_op,
            )
        _all_to_all_via_single(output_tensor_list, input_tensor_list, group=group)
        return None

    dist.all_to_all = _patched_all_to_all

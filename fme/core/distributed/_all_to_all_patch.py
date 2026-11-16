"""Traceable, gloo-compatible replacement for torch.distributed.all_to_all.

``torch_harmonics.distributed`` transposes data between spatial ranks with
the list form of ``torch.distributed.all_to_all``, which has two problems:

* the gloo backend does not implement it;
* dynamo cannot trace it on any backend, so every transpose breaks the graph
  of a compiled module.

This module monkey-patches ``torch.distributed.all_to_all`` with an
implementation on top of the functional ``all_to_all_single`` collective,
which gloo implements and dynamo traces into the graph.
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
    """Implement ``all_to_all`` semantics with ``all_to_all_single``.

    Rank *r* sends ``input_tensor_list[i]`` to rank *i* and receives into
    ``output_tensor_list[i]`` from rank *i*. The inputs are flattened into
    one buffer, exchanged with per-rank split sizes, and the received chunks
    copied into the outputs, so only data is moved and values are unchanged.
    """
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
    """Monkey-patch ``torch.distributed.all_to_all`` with
    :func:`_all_to_all_via_single` for synchronous calls on every backend.

    Asynchronous calls (``async_op=True``) delegate to the original
    ``all_to_all``, which gloo does not support.

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

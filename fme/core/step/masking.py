"""Input-mask helpers shared across step implementations.

Two operations every step that supports ``allow_missing_variables``
needs:

- ``apply_input_mask``: replace normalized-space values for
  masked-out samples with 0 (the normalized-space climatological
  mean). Explicit, in step logic — not the ``fill_nans_on_normalize``
  silent fallback in the normalizer.
- ``build_channel_mask_dict``: build per-variable spatial bool
  tensors (1.0 present, 0.0 masked) that can be packed and
  concatenated to the network input so the model receives the
  missing-flag signal alongside the (zeroed) value.

These were previously private to ``fme.core.step.single_module``;
extracted here so the CMIP6 step can use the same contract.
"""

import torch

from fme.core.typing_ import TensorDict, TensorMapping


def apply_input_mask(input_norm: TensorDict, data_mask: TensorMapping) -> TensorDict:
    """Zero out masked input variables in normalized space.

    For each variable in ``data_mask`` with False entries, sets those
    batch members' values to 0 in the normalized input. This is
    equivalent to replacing with the climatological mean in physical
    space (mean=0 after standardization).

    Args:
        input_norm: Normalized input tensors keyed by variable name.
        data_mask: Per-variable boolean masks of shape ``[batch]``
            (True = present, False = masked).

    Returns:
        A new TensorDict with the same keys; masked entries set to 0.
    """
    result = dict(input_norm)
    for name, mask in data_mask.items():
        if name in result:
            # mask shape: [batch]; data shape: [batch, ...spatial...]
            broadcast_mask = mask.view(mask.shape[0], *([1] * (result[name].ndim - 1)))
            result[name] = torch.where(broadcast_mask, result[name], 0.0)
    return result


def build_channel_mask_dict(
    in_names: list[str],
    data_mask: TensorMapping | None,
    packed_input: torch.Tensor,
) -> TensorDict:
    """Build a dict of per-variable spatial mask tensors.

    Returns a ``TensorDict`` keyed by variable name, with each value a
    ``(batch, *spatial)`` float tensor (1.0 = present, 0.0 = masked).
    The caller is responsible for packing this dict into the correct
    channel order. Variables not in ``data_mask`` get an all-ones
    mask — they're treated as universally present (e.g.,
    higher-coverage variables; or all variables when the loader
    doesn't provide a mask at all, such as inference).

    Args:
        in_names: Input variable names, in channel order.
        data_mask: Per-variable boolean masks of shape ``[batch]``,
            or None (all variables present).
        packed_input: The packed input tensor — used for shape and
            device inference. Expected shape ``(batch, *_, lat, lon)``.

    Returns:
        TensorDict keyed by input name; each value of shape
        ``(batch, lat, lon)``.
    """
    batch = packed_input.shape[0]
    spatial = packed_input.shape[-2:]
    device = packed_input.device
    result: TensorDict = {}
    for name in in_names:
        if data_mask is not None and name in data_mask:
            mask_1d = data_mask[name].to(device=device, dtype=torch.float)
            result[name] = mask_1d.view(batch, 1, 1).expand(batch, *spatial)
        else:
            result[name] = torch.ones(batch, *spatial, device=device)
    return result

"""Checkpoint weight averaging ("model soup").

Averages the parameters of several checkpoints of the same model, following
Wortsman et al. (2022), "Model soups: averaging weights of multiple fine-tuned
models improves accuracy without increasing inference time".

Usage as a script::

    python -m fme.core.checkpoint_soup OUTPUT.tar IN1.tar IN2.tar [--weights W1 W2]
"""

import argparse
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from fme.core.ema import EMA_CHECKPOINT_KEY

SOUP_CHECKPOINT_KEY = "soup"

# Checkpoint keys holding training-resumption state, which does not describe
# the averaged weights and so is not copied into the soup.
_DROPPED_CHECKPOINT_KEYS = ("optimization",)
# Key of the EMA weights within the EMA tracker state (``EMATracker._PARAMS_KEY``).
_EMA_PARAMS_KEY = "ema_params"


def _normalized_weights(n: int, weights: Sequence[float] | None) -> list[float]:
    if n == 0:
        raise ValueError("At least one input is required.")
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"Got {len(weights)} weights for {n} inputs.")
    if any(w < 0 for w in weights):
        raise ValueError(f"Weights must be non-negative, got {list(weights)}.")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError(f"Weights must have a positive sum, got {list(weights)}.")
    return [float(w) / total for w in weights]


def average_state_dicts(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Average several state dicts of the same model.

    The average is computed in float32 on CPU regardless of the input dtypes
    and devices, and each result is cast back to the dtype of the
    corresponding tensor in the first state dict. Results are on CPU.

    Non-floating tensors (e.g. integer or boolean buffers such as step
    counters) are not averaged: they are copied from the first state dict.

    Args:
        state_dicts: The state dicts to average. All must have the same keys,
            and tensors under the same key must have the same shape.
        weights: Optional non-negative weight per state dict. They are
            normalized to sum to one. If None, a uniform average is taken.

    Returns:
        The averaged state dict.

    Raises:
        ValueError: If the state dicts have different keys or shapes, or if the
            weights are invalid.
    """
    normalized = _normalized_weights(len(state_dicts), weights)
    first = state_dicts[0]
    keys = set(first.keys())
    for i, state_dict in enumerate(state_dicts[1:], start=1):
        if set(state_dict.keys()) != keys:
            missing = sorted(keys - set(state_dict.keys()))
            extra = sorted(set(state_dict.keys()) - keys)
            raise ValueError(
                f"State dict {i} has different keys than state dict 0: "
                f"missing {missing}, extra {extra}."
            )
    result: dict[str, torch.Tensor] = {}
    for key, reference in first.items():
        for i, state_dict in enumerate(state_dicts[1:], start=1):
            if state_dict[key].shape != reference.shape:
                raise ValueError(
                    f"Tensor {key!r} has shape {tuple(state_dict[key].shape)} in "
                    f"state dict {i} but {tuple(reference.shape)} in state dict 0."
                )
        if not reference.is_floating_point():
            result[key] = reference.detach().to("cpu", copy=True)
            continue
        total = torch.zeros(reference.shape, dtype=torch.float32)
        for weight, state_dict in zip(normalized, state_dicts):
            total += weight * state_dict[key].detach().to("cpu", torch.float32)
        result[key] = total.to(reference.dtype)
    return result


def _flatten_tensors(
    state: Any, prefix: tuple[str, ...], tensors: dict[tuple[str, ...], torch.Tensor]
) -> Any:
    """Move tensor leaves of nested mappings into ``tensors``.

    Returns the structure with each tensor leaf replaced by ``None``, so that
    the non-tensor structure can be compared across checkpoints.
    """
    if isinstance(state, torch.Tensor):
        tensors[prefix] = state
        return None
    if isinstance(state, Mapping):
        return {
            key: _flatten_tensors(value, prefix + (key,), tensors)
            for key, value in state.items()
        }
    return state


def _unflatten_tensors(
    skeleton: Any, prefix: tuple[str, ...], tensors: Mapping[tuple[str, ...], Any]
) -> Any:
    if prefix in tensors:
        return tensors[prefix]
    if isinstance(skeleton, Mapping):
        return {
            key: _unflatten_tensors(value, prefix + (key,), tensors)
            for key, value in skeleton.items()
        }
    return skeleton


def _average_nested_states(
    states: Sequence[Any], weights: Sequence[float] | None
) -> Any:
    """Average the tensor leaves of nested mappings with identical structure."""
    flat: list[dict[tuple[str, ...], torch.Tensor]] = []
    skeletons = []
    for state in states:
        tensors: dict[tuple[str, ...], torch.Tensor] = {}
        skeletons.append(_flatten_tensors(state, (), tensors))
        flat.append(tensors)
    for i, skeleton in enumerate(skeletons[1:], start=1):
        if skeleton != skeletons[0]:
            raise ValueError(
                f"Input {i} has a different non-tensor module state than input 0."
            )
    # join nested keys with a separator that cannot appear in parameter names
    joined = [{"\0".join(k): v for k, v in tensors.items()} for tensors in flat]
    averaged = average_state_dicts(joined, weights)
    return _unflatten_tensors(
        skeletons[0], (), {tuple(k.split("\0")): v for k, v in averaged.items()}
    )


def _states_equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and a.dtype == b.dtype
            and torch.equal(a.cpu(), b.cpu())
        )
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_states_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_states_equal(x, y) for x, y in zip(a, b))
    return a == b


def _weights_key(stepper_state: Mapping[str, Any]) -> str:
    # "module" is the layout of legacy SingleModuleStepper checkpoints
    for key in ("step", "module"):
        if key in stepper_state:
            return key
    raise ValueError(
        "Stepper state has no module weights (expected a 'step' or 'module' key), "
        f"keys are {sorted(stepper_state.keys())}."
    )


def soup_checkpoint_dicts(
    checkpoints: Sequence[Mapping[str, Any]],
    sources: Sequence[str],
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Average the stepper weights of several loaded training checkpoints.

    In-memory counterpart of :func:`soup_checkpoints`, which documents the
    layout of the result.

    Args:
        checkpoints: The loaded checkpoints to average.
        sources: A description of each checkpoint (e.g. its path), recorded
            in the result and used in error messages.
        weights: Optional non-negative weight per checkpoint, normalized to
            sum to one. If None, a uniform average is taken.

    Returns:
        The averaged checkpoint.

    Raises:
        ValueError: If the checkpoints have different stepper configs or
            module states that cannot be averaged.
    """
    if len(sources) != len(checkpoints):
        raise ValueError(
            f"Got {len(sources)} sources for {len(checkpoints)} checkpoints."
        )
    normalized = _normalized_weights(len(checkpoints), weights)
    stepper_states = [checkpoint["stepper"] for checkpoint in checkpoints]
    for source, stepper_state in zip(sources[1:], stepper_states[1:]):
        if not _states_equal(stepper_state["config"], stepper_states[0]["config"]):
            raise ValueError(
                f"Checkpoint {source} has a different stepper config "
                f"than {sources[0]}."
            )
    weights_key = _weights_key(stepper_states[0])
    for source, stepper_state in zip(sources[1:], stepper_states[1:]):
        if _weights_key(stepper_state) != weights_key:
            raise ValueError(
                f"Checkpoint {source} has a different stepper state layout "
                f"than {sources[0]}."
            )
    averaged = _average_nested_states(
        [stepper_state[weights_key] for stepper_state in stepper_states], normalized
    )

    output = {
        key: value
        for key, value in checkpoints[0].items()
        if key not in _DROPPED_CHECKPOINT_KEYS
    }
    output["stepper"] = {**stepper_states[0], weights_key: averaged}
    if EMA_CHECKPOINT_KEY in output:
        output[EMA_CHECKPOINT_KEY] = {
            key: value
            for key, value in output[EMA_CHECKPOINT_KEY].items()
            if key != _EMA_PARAMS_KEY
        }
    output[SOUP_CHECKPOINT_KEY] = {
        "sources": [str(source) for source in sources],
        "weights": normalized,
    }
    return output


def soup_checkpoints(
    paths: Sequence[str],
    output_path: str,
    weights: Sequence[float] | None = None,
) -> None:
    """Average the stepper weights of several training checkpoints.

    The output checkpoint can be used wherever a stepper checkpoint is
    expected, e.g. for inference or evaluation, or as
    ``ParameterInitializationConfig.weights_path``.

    Every tensor under the stepper's module state (``checkpoint["stepper"]
    ["step"]``, or ``["module"]`` for legacy checkpoints) is averaged as in
    :func:`average_state_dicts`; non-floating tensors are taken from the first
    checkpoint. All other entries (stepper config, dataset info, training
    history, epoch, ...) are copied from the first checkpoint. Optimizer state
    and stored EMA parameters are dropped, so that loading the soup with
    "use EMA if available" uses the averaged weights. The sources and
    normalized weights are recorded under the top-level ``"soup"`` key.

    Epoch EMA checkpoints (``ema_ckpt_XXXX.tar``) are saved with the EMA
    weights applied, so their stepper weights are the EMA weights. For
    checkpoints saved with optimization state (``ckpt.tar``), the stepper
    weights are the raw (non-EMA) weights.

    Args:
        paths: Paths of the checkpoints to average.
        output_path: Path to write the averaged checkpoint to.
        weights: Optional non-negative weight per checkpoint, normalized to
            sum to one. If None, a uniform average is taken.

    Raises:
        ValueError: If the checkpoints have different stepper configs or
            module states that cannot be averaged.
    """
    _normalized_weights(len(paths), weights)  # validate before loading
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False) for path in paths
    ]
    output = soup_checkpoint_dicts(checkpoints, sources=paths, weights=weights)
    torch.save(output, output_path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Average the stepper weights of several checkpoints."
    )
    parser.add_argument("output", help="Path to write the averaged checkpoint to.")
    parser.add_argument("inputs", nargs="+", help="Checkpoints to average.")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Weight per input checkpoint (normalized to sum to one). "
        "Defaults to a uniform average.",
    )
    args = parser.parse_args(argv)
    soup_checkpoints(args.inputs, args.output, weights=args.weights)
    logging.info(f"Wrote average of {len(args.inputs)} checkpoints to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

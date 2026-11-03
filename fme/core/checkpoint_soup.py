"""Checkpoint weight averaging ("model soup", Wortsman et al. 2022).

Usage::

    python -m fme.core.checkpoint_soup OUTPUT.tar IN1.tar IN2.tar [--weights W1 W2]
"""

import argparse
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from fme.core.distributed import Distributed
from fme.core.ema import EMA_CHECKPOINT_KEY

SOUP_CHECKPOINT_KEY = "soup"
_EMA_PARAMS_KEY = "ema_params"  # EMATracker._PARAMS_KEY


def _normalized_weights(n: int, weights: Sequence[float] | None) -> list[float]:
    if n == 0:
        raise ValueError("At least one input is required.")
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"Got {len(weights)} weights for {n} inputs.")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(
            f"Weights must be non-negative with a positive sum, got {list(weights)}."
        )
    total = float(sum(weights))
    return [float(w) / total for w in weights]


def _average(values: Sequence[Any], weights: Sequence[float], name: str) -> Any:
    """Weighted average of the tensor leaves of nested mappings.

    Floating tensors are accumulated on CPU in at least float32 and cast back to
    the input dtype. Non-floating tensors are copied from the first input. Any
    other leaf must be equal across inputs.
    """
    first = values[0]
    if isinstance(first, torch.Tensor):
        for i, value in enumerate(values[1:], start=1):
            if not isinstance(value, torch.Tensor) or value.shape != first.shape:
                shape = getattr(value, "shape", type(value).__name__)
                raise ValueError(
                    f"{name!r} has shape {tuple(first.shape)} in input 0 "
                    f"but {shape} in input {i}."
                )
        if not first.is_floating_point():
            return first.detach().to("cpu", copy=True)
        accumulate_dtype = torch.promote_types(first.dtype, torch.float32)
        total = torch.zeros(first.shape, dtype=accumulate_dtype)
        for weight, value in zip(weights, values):
            total += weight * value.detach().to("cpu", accumulate_dtype)
        return total.to(first.dtype)
    if isinstance(first, Mapping):
        for i, value in enumerate(values[1:], start=1):
            if not isinstance(value, Mapping) or value.keys() != first.keys():
                keys = set(value.keys()) if isinstance(value, Mapping) else set()
                raise ValueError(
                    f"{name!r} has different keys in input {i} than in input 0: "
                    f"missing {sorted(first.keys() - keys)}, "
                    f"extra {sorted(keys - first.keys())}."
                )
        return {
            key: _average([value[key] for value in values], weights, f"{name}.{key}")
            for key in first
        }
    for i, value in enumerate(values[1:], start=1):
        if value != first:
            raise ValueError(f"{name!r} differs between input {i} and input 0.")
    return first


def average_state_dicts(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Average state dicts of the same model.

    Floating tensors are averaged in at least float32 and returned on CPU in
    their input dtype; non-floating tensors (e.g. step counters) are copied
    from the first state dict.

    Args:
        state_dicts: State dicts with identical keys and shapes.
        weights: Non-negative weight per state dict, normalized to sum to one.
            Defaults to a uniform average.
    """
    normalized = _normalized_weights(len(state_dicts), weights)
    return _average(state_dicts, normalized, name="state_dict")


def _weights_key(stepper_state: Mapping[str, Any]) -> str:
    # "module" is the layout of legacy SingleModuleStepper checkpoints
    for key in ("step", "module"):
        if key in stepper_state:
            return key
    raise ValueError(
        "Stepper state has no 'step' or 'module' key, "
        f"keys are {sorted(stepper_state.keys())}."
    )


def soup_checkpoints(
    paths: Sequence[str],
    output_path: str,
    weights: Sequence[float] | None = None,
) -> None:
    """Average the stepper weights of training checkpoints of the same model.

    The output loads like any stepper checkpoint, e.g. for inference or as
    ``ParameterInitializationConfig.weights_path``.

    Every tensor under ``checkpoint["stepper"]["step"]`` (``["module"]`` for
    legacy checkpoints) is averaged as in :func:`average_state_dicts`. Everything
    else is copied from the first checkpoint, except the optimizer state and
    stored EMA weights, which are dropped so that loading the soup with
    ``use_ema_if_available`` keeps the averaged weights. The sources and
    normalized weights are recorded under ``checkpoint["soup"]``.

    Epoch EMA checkpoints (``ema_ckpt_XXXX.tar``) store the EMA weights as
    stepper weights; ``ckpt.tar`` stores the raw weights, so souping it
    averages the raw (non-EMA) weights.

    Args:
        paths: Checkpoints to average. Their stepper configs must be equal.
        output_path: Where to write the averaged checkpoint.
        weights: Non-negative weight per checkpoint, normalized to sum to one.
            Defaults to a uniform average.
    """
    normalized = _normalized_weights(len(paths), weights)
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False) for path in paths
    ]
    stepper_states = [checkpoint["stepper"] for checkpoint in checkpoints]
    weights_key = _weights_key(stepper_states[0])
    for path, checkpoint, stepper_state in zip(paths, checkpoints, stepper_states):
        if stepper_state["config"] != stepper_states[0]["config"]:
            raise ValueError(
                f"Checkpoint {path} has a different stepper config than {paths[0]}."
            )
        if weights_key not in stepper_state:
            raise ValueError(
                f"Checkpoint {path} has no stepper {weights_key!r} state, "
                f"unlike {paths[0]}."
            )
        if _EMA_PARAMS_KEY in checkpoint.get(EMA_CHECKPOINT_KEY, {}):
            logging.warning(
                f"Checkpoint {path} stores its EMA weights separately (e.g. "
                "ckpt.tar), so its raw, non-EMA weights are averaged."
            )
    averaged = _average(
        [stepper_state[weights_key] for stepper_state in stepper_states],
        normalized,
        name=weights_key,
    )

    output = dict(checkpoints[0])
    output.pop("optimization", None)
    output["stepper"] = {**stepper_states[0], weights_key: averaged}
    if EMA_CHECKPOINT_KEY in output:
        output[EMA_CHECKPOINT_KEY] = {
            key: value
            for key, value in output[EMA_CHECKPOINT_KEY].items()
            if key != _EMA_PARAMS_KEY
        }
    output[SOUP_CHECKPOINT_KEY] = {
        "sources": [str(path) for path in paths],
        "weights": normalized,
    }
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
        help="Weight per input (normalized to sum to one). Defaults to uniform.",
    )
    args = parser.parse_args(argv)
    soup_checkpoints(args.inputs, args.output, weights=args.weights)
    logging.info(f"Wrote average of {len(args.inputs)} checkpoints to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with Distributed.context():
        main()

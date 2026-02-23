#!/usr/bin/env python
"""
Diagnostic script to inspect channels_last memory format behaviour in a
downscaling diffusion model checkpoint.

Reports:
  1. Whether the model weights are stored in channels_last format.
  2. How many memory-format conversions occur during a forward pass when the
     input tensors are already in channels_last layout.

Usage:
    python check_channels_last.py <checkpoint_path> [--device cpu]
"""

import argparse
import os
from dataclasses import dataclass

import torch

os.environ.setdefault("FME_DISTRIBUTED_BACKEND", "none")

from fme.core.device import get_device
from fme.downscaling.models import DiffusionModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check channels_last memory format in a downscaling model"
    )
    parser.add_argument("checkpoint_path", help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (e.g. 'cpu'). Defaults to auto-detected device.",
    )
    return parser.parse_args()


def check_weights_channels_last(model: torch.nn.Module) -> dict[str, bool]:
    """Return a dict mapping parameter names to whether they are channels_last."""
    results = {}
    for name, param in model.named_parameters():
        if param.ndim == 4:
            results[name] = param.is_contiguous(memory_format=torch.channels_last)
    return results


@dataclass
class ConversionEvent:
    module_name: str
    module_type: str
    direction: str  # "to_channels_last" or "from_channels_last"
    tensor_role: str  # e.g. "input_0", "output_0"


def _is_channels_last(t: torch.Tensor) -> bool:
    return t.ndim == 4 and t.is_contiguous(memory_format=torch.channels_last)


def _is_contiguous_only(t: torch.Tensor) -> bool:
    """True when the tensor is standard contiguous but NOT channels_last."""
    return (
        t.ndim == 4
        and t.is_contiguous()
        and not t.is_contiguous(memory_format=torch.channels_last)
    )


def count_format_conversions(
    model: torch.nn.Module,
    latent: torch.Tensor,
    conditioning: torch.Tensor,
    noise_level: torch.Tensor,
) -> list[ConversionEvent]:
    """
    Run a forward pass and record every point where a 4-D tensor flips
    between channels_last and standard contiguous layout.
    """
    events: list[ConversionEvent] = []
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, outputs):
            in_tensors = []
            if isinstance(inputs, torch.Tensor):
                in_tensors.append(("input_0", inputs))
            elif isinstance(inputs, (tuple, list)):
                for i, t in enumerate(inputs):
                    if isinstance(t, torch.Tensor):
                        in_tensors.append((f"input_{i}", t))

            out_tensors = []
            if isinstance(outputs, torch.Tensor):
                out_tensors.append(("output_0", outputs))
            elif isinstance(outputs, (tuple, list)):
                for i, t in enumerate(outputs):
                    if isinstance(t, torch.Tensor):
                        out_tensors.append((f"output_{i}", t))

            for role, t in in_tensors + out_tensors:
                pass  # collected below per input→output pair

            for in_role, in_t in in_tensors:
                for out_role, out_t in out_tensors:
                    if _is_channels_last(in_t) and _is_contiguous_only(out_t):
                        events.append(
                            ConversionEvent(
                                module_name=name,
                                module_type=type(module).__name__,
                                direction="from_channels_last",
                                tensor_role=f"{in_role}->{out_role}",
                            )
                        )
                    elif _is_contiguous_only(in_t) and _is_channels_last(out_t):
                        events.append(
                            ConversionEvent(
                                module_name=name,
                                module_type=type(module).__name__,
                                direction="to_channels_last",
                                tensor_role=f"{in_role}->{out_role}",
                            )
                        )

        return hook

    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(latent, conditioning, noise_level)

    for h in handles:
        h.remove()

    return events


def load_model(checkpoint_path: str) -> DiffusionModel:
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    checkpoint["model"].setdefault("static_inputs", None)
    return DiffusionModel.from_state(checkpoint["model"])


def main():
    args = parse_args()

    if args.device == "cpu":
        os.environ["FME_FORCE_CPU"] = "1"

    print(f"Loading checkpoint: {args.checkpoint_path}")
    model = load_model(args.checkpoint_path)
    device = get_device()
    print(f"Device: {device}\n")

    # --- 1. Check weight memory formats ---
    print("=" * 70)
    print("Weight memory format check (4-D parameters only)")
    print("=" * 70)
    weight_info = check_weights_channels_last(model.module)
    n_cl = sum(v for v in weight_info.values())
    n_total = len(weight_info)
    print(f"  {n_cl}/{n_total} 4-D parameters are channels_last\n")
    for name, is_cl in weight_info.items():
        tag = "channels_last" if is_cl else "contiguous"
        print(f"  [{tag:14s}] {name}")

    # --- 2. Count format conversions during forward pass ---
    print("\n" + "=" * 70)
    print("Forward-pass memory format conversion check")
    print("=" * 70)

    n_batch = 1
    n_out = len(model.out_packer.names)
    n_in = len(model.in_packer.names)
    if model.config.use_fine_topography:
        n_in += 1
    h, w = model.fine_shape

    latent = torch.randn(n_batch, n_out, h, w, device=device)
    conditioning = torch.randn(n_batch, n_in, h, w, device=device)
    noise_level = torch.ones(n_batch, 1, 1, 1, device=device)

    latent = latent.to(memory_format=torch.channels_last)
    conditioning = conditioning.to(memory_format=torch.channels_last)

    print(f"  Input shapes: latent={tuple(latent.shape)}, "
          f"conditioning={tuple(conditioning.shape)}, "
          f"noise_level={tuple(noise_level.shape)}")
    print(f"  Input latent is channels_last: {_is_channels_last(latent)}")
    print(f"  Input conditioning is channels_last: {_is_channels_last(conditioning)}\n")

    events = count_format_conversions(model.module, latent, conditioning, noise_level)

    to_cl = [e for e in events if e.direction == "to_channels_last"]
    from_cl = [e for e in events if e.direction == "from_channels_last"]

    print(f"  Total conversion events: {len(events)}")
    print(f"    contiguous -> channels_last: {len(to_cl)}")
    print(f"    channels_last -> contiguous: {len(from_cl)}")

    if events:
        print("\n  Detailed conversion events:")
        print(f"  {'Module':<55s} {'Type':<25s} {'Direction':<22s} {'Tensors'}")
        print("  " + "-" * 130)
        for e in events:
            print(
                f"  {e.module_name:<55s} {e.module_type:<25s} "
                f"{e.direction:<22s} {e.tensor_role}"
            )
    else:
        print("\n  No memory format conversions detected — the model stays in "
              "channels_last throughout the forward pass.")

    print()


if __name__ == "__main__":
    main()

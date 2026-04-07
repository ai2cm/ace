"""
Script to resave downscaling checkpoints to the current StaticInputs format.

Old checkpoints saved before the StaticInputs coordinate serialization change
lack a top-level "coords" key in the static_inputs state. This script migrates
them to the new format so they can be loaded by restore_checkpoint().

Usage:
    python scripts/checkpoint_publication/resave_downscaling_ckpt.py \
        --checkpoint-dir /path/to/old/checkpoints \
        --static-inputs '{"fine_topography": "/path/to/topo.nc"}' \
        --output-dir /path/to/new/checkpoints

The script processes latest.ckpt, ema_ckpt.tar, best.ckpt, and
best_histogram_tail.ckpt if they exist.
"""

import argparse
import json
import os

import torch

from fme.core.distributed import Distributed
from fme.downscaling.data.static import _has_coords_in_state, load_static_inputs
from fme.downscaling.models import DiffusionModel


def _migrate_model_state(
    model_state: dict, static_inputs_config: dict[str, str]
) -> dict:
    """
    Migrate model state to new StaticInputs format if needed.

    Returns updated model state (may be the same object if no migration needed).
    """
    raw_static = model_state.get("static_inputs", {})
    if not raw_static and not static_inputs_config:
        return model_state  # no static inputs, nothing to do

    if _has_coords_in_state(raw_static):
        return model_state  # already in a loadable format

    # Old format: no coords at all. Reload from config files.
    print(
        "Migrating model state to new StaticInputs format with coordinates. "
        f"Using static inputs config: {static_inputs_config}"
    )
    static_inputs = load_static_inputs(static_inputs_config)
    new_model_state = dict(model_state)
    new_model_state["static_inputs"] = static_inputs.get_state()
    return new_model_state


def _migrate_checkpoint(checkpoint: dict, static_inputs_config: dict[str, str]) -> dict:
    """Migrate a checkpoint dict in-place (returns updated dict)."""
    if "model" in checkpoint:
        checkpoint = dict(checkpoint)
        checkpoint["model"] = _migrate_model_state(
            checkpoint["model"], static_inputs_config
        )
    return checkpoint


def _verify_checkpoint(output_path: str) -> None:
    """Verify the saved checkpoint can be used to rebuild the model via from_state."""
    checkpoint = torch.load(
        output_path, map_location=torch.device("cpu"), weights_only=False
    )
    if "model" not in checkpoint:
        return
    DiffusionModel.from_state(checkpoint["model"])
    print(f"  Verified: model rebuilt successfully from {output_path}")


def _process_file(
    input_path: str,
    output_path: str,
    static_inputs_config: dict[str, str],
) -> None:
    print(f"Loading {input_path}")
    checkpoint = torch.load(
        input_path, map_location=torch.device("cpu"), weights_only=False
    )
    checkpoint = _migrate_checkpoint(checkpoint, static_inputs_config)
    print(f"Saving {output_path}")
    torch.save(checkpoint, output_path)
    _verify_checkpoint(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Resave downscaling checkpoints to current StaticInputs format."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing the old checkpoints.",
    )
    parser.add_argument(
        "--static-inputs",
        required=True,
        help=(
            "JSON mapping of field names to file paths for static inputs, "
            'e.g. \'{"fine_topography": "/path/to/topo.nc"}\''
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the migrated checkpoints.",
    )
    args = parser.parse_args()

    static_inputs_config: dict[str, str] = json.loads(args.static_inputs)
    os.makedirs(args.output_dir, exist_ok=True)

    filenames = [
        "latest.ckpt",
        "ema_ckpt.tar",
        "best.ckpt",
        "best_histogram_tail.ckpt",
    ]
    processed = 0
    for filename in filenames:
        input_path = os.path.join(args.checkpoint_dir, filename)
        if not os.path.isfile(input_path):
            continue
        output_path = os.path.join(args.output_dir, filename)
        _process_file(input_path, output_path, static_inputs_config)
        processed += 1

    if processed == 0:
        raise FileNotFoundError(
            f"No checkpoint files found in {args.checkpoint_dir}. "
            f"Expected one or more of: {filenames}"
        )
    print(f"Done. Migrated {processed} checkpoint(s) to {args.output_dir}")


if __name__ == "__main__":
    with Distributed.context():
        main()

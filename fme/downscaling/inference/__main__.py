"""
Generate and save downscaled data using a trained FME model.

This module provides a flexible API for generating high-resolution downscaled
outputs from trained diffusion models.

Usage:
    # Single-GPU
    python -m fme.downscaling.inference config.yaml

    # Multi-GPU using DDP
    torchrun --nproc_per_node=8 -m fme.downscaling.inference config.yaml

Output Structure:
    Each target generates a zarr file at: {output_dir}/{target_name}.zarr

    Zarr dimensions: (time, ensemble, latitude, longitude)
"""

import argparse

from .inference import main


def parse_args():
    parser = argparse.ArgumentParser(description="Downscaling generation script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)

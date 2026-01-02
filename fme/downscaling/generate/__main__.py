"""
Generate and save downscaled data using a trained FME model.

This module provides a flexible API for generating high-resolution downscaled
outputs from trained diffusion models. It supports:

Usage:
    python -m fme.downscaling.generate config.yaml

    # Multi-GPU
    torchrun --nproc_per_node=8 -m fme.downscaling.generate config.yaml

Output Structure:
    Each target generates a zarr file at: {output_dir}/{target_name}.zarr

    Zarr dimensions: (time, ensemble, latitude, longitude)

Example:
        /results/downscaling_run/
        ├── hurricane_landfall.zarr/
        │   ├── PRATEsfc/
        │   └── WIND10m/
        ├── miami_timeseries.zarr/
        └── conus_summer_2021.zarr/
"""

import argparse

from .generate import main


def parse_args():
    parser = argparse.ArgumentParser(description="Downscaling generation script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)

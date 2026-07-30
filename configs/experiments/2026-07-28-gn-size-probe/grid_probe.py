"""Dump each arm's fine-resolution lat/lon vectors. No generation.

The fine latitude grid is not a uniform 1/32 deg: spacing is ~0.031162 deg
(32.09 cells/deg), so a domain's fine cells are aligned to that domain rather
than to a grid shared across domains. Two arms cropped to the same nominal
lat/lon box therefore land on different cells -- 509 vs 514 rows for the PNW
16 deg and 42 deg arms -- which is what gn_frozen_eval.py's cross-arm alignment
check caught.

Recovering the coordinate vectors is cheap and needs no model, so this exists
to align already-generated fields after the fact rather than re-running two
hours of generation.

Writes ``grid-coords.npz`` with ``<arm>|lat`` and ``<arm>|lon``.

Usage:
    python grid_probe.py <config.yaml>
"""

import argparse
import logging
import os

import dacite
import numpy as np
import yaml

from fme.downscaling.evaluator import EvaluatorConfig


def main(config_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    config: EvaluatorConfig = dacite.from_dict(
        data_class=EvaluatorConfig, data=raw, config=dacite.Config(strict=True)
    )
    os.makedirs(config.experiment_dir, exist_ok=True)

    requirements = config.model.data_requirements
    arrays: dict[str, np.ndarray] = {}
    for event in config.events or []:
        data = event.get_paired_gridded_data(
            base_data_config=config.data, requirements=requirements
        )
        batch = next(iter(data.get_generator()))
        fine = batch[0].fine.latlon_coordinates
        coarse = batch[0].coarse.latlon_coordinates
        lat = fine.lat.cpu().numpy()
        lon = fine.lon.cpu().numpy()
        arrays[f"{event.name}|lat"] = lat
        arrays[f"{event.name}|lon"] = lon
        arrays[f"{event.name}|coarse_lat"] = coarse.lat.cpu().numpy()
        arrays[f"{event.name}|coarse_lon"] = coarse.lon.cpu().numpy()
        logging.info(
            f"{event.name}: fine lat {lat.shape} {lat[0]:.6f}..{lat[-1]:.6f} "
            f"(d={np.diff(lat).mean():.8f}); "
            f"fine lon {lon.shape} {lon[0]:.6f}..{lon[-1]:.6f} "
            f"(d={np.diff(lon).mean():.8f})"
        )

    out = os.path.join(config.experiment_dir, "grid-coords.npz")
    np.savez_compressed(out, **arrays)
    logging.info(f"wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().config_path)

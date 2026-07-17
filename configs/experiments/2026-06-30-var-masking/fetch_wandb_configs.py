"""Fetch the config.yaml from every beaker dataset in wandb_to_beaker_map.json.

For each entry in the map, stream the top-level ``config.yaml`` out of the
beaker result dataset and write it to ``wandb_configs/<run_name>.yaml``.

Usage:
    python fetch_wandb_configs.py [--map PATH] [--out-dir DIR] [--force]
                                  [--version {v1,v2}]
"""

import argparse
import json
import pathlib
import subprocess

from generate_masking_configs import BASE_CONFIG_FILENAMES, stem_has_version

HERE = pathlib.Path(__file__).parent
DEFAULT_MAP = HERE / "wandb_to_beaker_map.json"
DEFAULT_OUT = HERE / "wandb_configs"


def _stream_config(dataset_id: str) -> str | None:
    """Return the contents of config.yaml in a beaker dataset, else None."""
    proc = subprocess.run(
        ["beaker", "dataset", "stream-file", dataset_id, "config.yaml"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        type=pathlib.Path,
        default=DEFAULT_MAP,
        help=f"wandb->beaker map to read (default: {DEFAULT_MAP}).",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=DEFAULT_OUT,
        help=f"Directory to write configs into (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch configs that already exist locally.",
    )
    parser.add_argument(
        "--version",
        "-v",
        choices=sorted(BASE_CONFIG_FILENAMES),
        default=None,
        help="Restrict to runs of this baseline version (default: all).",
    )
    args = parser.parse_args()

    run_to_dataset: dict[str, str] = json.loads(args.map.read_text())
    if args.version is not None:
        run_to_dataset = {
            run_name: dataset_id
            for run_name, dataset_id in run_to_dataset.items()
            if stem_has_version(run_name, args.version)
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for run_name, dataset_id in sorted(run_to_dataset.items()):
        out_path = args.out_dir / f"{run_name}.yaml"
        if out_path.exists() and not args.force:
            print(f"  skip {run_name}: already fetched")
            continue
        config = _stream_config(dataset_id)
        if config is None:
            print(f"  skip {run_name}: no config.yaml in {dataset_id}")
            continue
        out_path.write_text(config)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()

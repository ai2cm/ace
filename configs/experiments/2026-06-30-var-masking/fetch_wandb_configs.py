"""Fetch the config.yaml from every beaker dataset in wandb_to_beaker_map.json.

For each entry in the map, stream the top-level ``config.yaml`` out of the
beaker result dataset and write it to ``wandb_configs/<run_name>.yaml``.

Usage:
    python fetch_wandb_configs.py [--map PATH] [--out-dir DIR] [--force]
"""

import argparse
import json
import pathlib
import subprocess

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
    args = parser.parse_args()

    run_to_dataset: dict[str, str] = json.loads(args.map.read_text())
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

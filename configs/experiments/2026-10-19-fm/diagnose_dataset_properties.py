"""Diagnose why `DatasetProperties.update` warns for a training config's datasets.

`DatasetProperties.update` (fme/core/dataset/properties.py) compares, in order,
the timestep, variable metadata, vertical coordinate, horizontal coordinates
and spatial mask provider of each concatenated dataset, and raises (or with
`strict: false`, warns) on the first mismatch only. This script builds every
train and validation dataset entry exactly as training does (same requested
names and n_timesteps) and reports every check that differs, so mismatches
hidden behind the first one become visible.

It only reads data (times, coordinates, masks and attributes); nothing is
written.

Usage (from the repo root, with the training data paths mounted):

    python configs/experiments/2026-10-19-fm/diagnose_dataset_properties.py \
        configs/experiments/2026-10-19-fm/base_configs/<train-config>.yaml

    # or submit as a CPU beaker job with weka mounted:
    python configs/experiments/2026-10-19-fm/diagnose_dataset_properties.py \
        --submit configs/experiments/2026-10-19-fm/base_configs/<train-config>.yaml
"""

import argparse
import dataclasses
import logging
import pathlib
import subprocess
import warnings
from typing import Any

import dacite
import torch

from fme.ace.train.train_config import TrainConfig
from fme.core.cli import prepare_config
from fme.core.coordinates import (
    HorizontalCoordinates,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
    VerticalCoordinate,
)
from fme.core.dataset.properties import DatasetProperties
from fme.core.distributed import Distributed

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT_REL = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

CHECKS = ("timestep", "metadata", "vertical", "horizontal", "mask")


@dataclasses.dataclass
class Entry:
    name: str
    family: str
    properties: DatasetProperties


def _entry_name(split: str, index: int, entry) -> str:
    subset = getattr(entry, "subset", None)
    start = getattr(subset, "start_time", None)
    stop = getattr(subset, "stop_time", None)
    window = f" [{start}..{stop}]" if start or stop else ""
    return f"{split}[{index}] {entry.data_path}/{entry.file_pattern}{window}"


def _family(entry) -> str:
    labels = getattr(entry, "labels", None)
    if labels:
        return ",".join(sorted(labels))
    return str(entry.data_path)


def build_entries(config: TrainConfig) -> list[Entry]:
    requirements = config._get_train_window_data_requirements()
    names = list(requirements.names)
    logging.info(f"Requested names ({len(names)}): {names}")
    logging.info(
        f"allow_missing_variables={requirements.allow_missing_variables}, "
        f"n_timesteps={requirements.n_timesteps_schedule}"
    )
    loaders = [("train", config.train_loader)] + [
        (f"val{i}", v.loader) for i, v in enumerate(config.validation_list)
    ]
    result = []
    for split, loader in loaders:
        n_timesteps = requirements.n_timesteps_schedule.add(loader.time_buffer)
        dataset_config = loader.dataset
        concat = getattr(dataset_config, "concat", None)
        sub_configs = list(concat) if concat is not None else [dataset_config]
        for i, sub in enumerate(sub_configs):
            name = _entry_name(split, i, sub)
            logging.info(f"Building {name}")
            _, properties = sub.build(
                names,
                n_timesteps,
                allow_missing_variables=requirements.allow_missing_variables,
            )
            result.append(Entry(name, _family(sub), properties))
    return result


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> str:
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    return f"{(a.double() - b.double()).abs().max().item():.6g}"


def describe_vertical(v: VerticalCoordinate) -> str:
    if isinstance(v, HybridSigmaPressureCoordinate):
        return (
            f"HybridSigmaPressure(ak={[round(x, 4) for x in v.ak.tolist()]}, "
            f"bk={[round(x, 6) for x in v.bk.tolist()]})"
        )
    return repr(v)


def diff_vertical(a: VerticalCoordinate, b: VerticalCoordinate) -> str | None:
    if a == b:
        return None
    if type(a) is not type(b):
        return f"type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, HybridSigmaPressureCoordinate) and isinstance(
        b, HybridSigmaPressureCoordinate
    ):
        return (
            f"ak max|diff|={_max_abs_diff(a.ak, b.ak)}, "
            f"bk max|diff|={_max_abs_diff(a.bk, b.bk)}"
        )
    return f"{a!r} vs {b!r}"


def describe_horizontal(h: HorizontalCoordinates) -> str:
    if isinstance(h, LatLonCoordinates):
        return (
            f"LatLon dims={tuple(h.dims)} lat[{len(h.lat)}]="
            f"{h.lat[0].item():.4f}..{h.lat[-1].item():.4f} lon[{len(h.lon)}]="
            f"{h.lon[0].item():.4f}..{h.lon[-1].item():.4f}"
        )
    return f"{type(h).__name__} dims={tuple(h.dims)}"


def diff_horizontal(a: HorizontalCoordinates, b: HorizontalCoordinates) -> str | None:
    if a == b and tuple(a.dims) == tuple(b.dims):
        return None
    if type(a) is not type(b):
        return f"type {type(a).__name__} vs {type(b).__name__}"
    parts = []
    if tuple(a.dims) != tuple(b.dims):
        parts.append(f"dims {tuple(a.dims)} vs {tuple(b.dims)}")
    if isinstance(a, LatLonCoordinates) and isinstance(b, LatLonCoordinates):
        parts.append(f"lat max|diff|={_max_abs_diff(a.lat, b.lat)}")
        parts.append(f"lon max|diff|={_max_abs_diff(a.lon, b.lon)}")
    return ", ".join(parts) or "differs"


def diff_mask(a, b) -> str | None:
    if a == b:
        return None
    keys_a, keys_b = set(a.masks), set(b.masks)
    parts = []
    if keys_a != keys_b:
        parts.append(
            f"only-ref={sorted(keys_a - keys_b)}, only-other={sorted(keys_b - keys_a)}"
        )
    for key in sorted(keys_a & keys_b):
        if not torch.equal(a.masks[key], b.masks[key]):
            parts.append(f"{key} max|diff|={_max_abs_diff(a.masks[key], b.masks[key])}")
    return "; ".join(parts) or "differs"


def diff_metadata(a: dict, b: dict) -> list[str]:
    lines = []
    for name in sorted(set(a) | set(b)):
        va, vb = a.get(name), b.get(name)
        if va == vb:
            continue
        if va is None or vb is None:
            lines.append(f"{name}: ref={va} other={vb}")
            continue
        fields = []
        if va.units != vb.units:
            fields.append(f"units {va.units!r} -> {vb.units!r}")
        if va.long_name != vb.long_name:
            fields.append(f"long_name {va.long_name!r} -> {vb.long_name!r}")
        lines.append(f"{name}: " + "; ".join(fields))
    return lines


def compare(ref: DatasetProperties, other: DatasetProperties) -> dict[str, Any]:
    return {
        "timestep": None
        if ref.timestep == other.timestep
        else f"{ref.timestep} vs {other.timestep}",
        "metadata": diff_metadata(ref.variable_metadata, other.variable_metadata)
        or None,
        "vertical": diff_vertical(ref.vertical_coordinate, other.vertical_coordinate),
        "horizontal": diff_horizontal(
            ref.horizontal_coordinates, other.horizontal_coordinates
        ),
        "mask": diff_mask(ref.spatial_mask_provider, other.spatial_mask_provider),
    }


def first_failing_check(ref: DatasetProperties, other: DatasetProperties) -> str:
    """The message `DatasetProperties.update` would raise, if any."""
    copy = ref.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        copy.update(other, strict=False)
    return str(caught[0].message) if caught else "-"


def report(entries: list[Entry]):
    ref = entries[0]
    print("\n" + "=" * 100)
    print(f"REFERENCE entry 0: {ref.name}")
    print(f"  timestep: {ref.properties.timestep}")
    print(f"  vertical: {describe_vertical(ref.properties.vertical_coordinate)}")
    print(f"  horizontal: {describe_horizontal(ref.properties.horizontal_coordinates)}")
    print(f"  masks: {sorted(ref.properties.spatial_mask_provider.masks)}")

    print("\n" + "=" * 100)
    print("PER-ENTRY SUMMARY vs entry 0 (X = differs)")
    header = f"{'idx':>3} {'family':<8} " + " ".join(f"{c:<10}" for c in CHECKS)
    print(header + " timestep / first update() error / name")
    results = []
    for i, entry in enumerate(entries):
        result = compare(ref.properties, entry.properties)
        results.append(result)
        flags = " ".join(f"{('X' if result[c] else '.'):<10}" for c in CHECKS)
        print(
            f"{i:>3} {entry.family:<8} {flags} {entry.properties.timestep} | "
            f"{first_failing_check(ref.properties, entry.properties)} | {entry.name}"
        )

    print("\n" + "=" * 100)
    print("DETAILS per distinct diff vs entry 0")
    seen: dict[str, list[int]] = {}
    for i, result in enumerate(results):
        key = repr(result)
        seen.setdefault(key, []).append(i)
    for key, indices in seen.items():
        result = results[indices[0]]
        print(f"\n--- entries {indices} (family {entries[indices[0]].family}) ---")
        if not any(result.values()):
            print("  identical to entry 0")
            continue
        for check in CHECKS:
            value = result[check]
            if not value:
                continue
            if check == "metadata":
                print(f"  metadata: {len(value)} variables differ")
                for line in value:
                    print(f"    {line}")
            else:
                print(f"  {check}: {value}")
        props = entries[indices[0]].properties
        if result["vertical"]:
            print(f"  vertical (this): {describe_vertical(props.vertical_coordinate)}")
        if result["horizontal"]:
            print(
                "  horizontal (this): "
                f"{describe_horizontal(props.horizontal_coordinates)}"
            )
        if result["mask"]:
            print(f"  masks (this): {sorted(props.spatial_mask_provider.masks)}")

    print("\n" + "=" * 100)
    print("WITHIN-FAMILY consistency (each entry vs first entry of its family)")
    families: dict[str, list[int]] = {}
    for i, entry in enumerate(entries):
        families.setdefault(entry.family, []).append(i)
    for family, indices in families.items():
        fam_ref = entries[indices[0]].properties
        differing = []
        for i in indices[1:]:
            result = compare(fam_ref, entries[i].properties)
            bad = [c for c in CHECKS if result[c]]
            if bad:
                differing.append((i, bad, result))
        print(
            f"family {family}: entries {indices}: "
            + ("all identical" if not differing else f"{len(differing)} differ")
        )
        for i, bad, result in differing:
            print(f"  entry {i} differs in {bad}")
            for check in bad:
                value = result[check]
                if check == "metadata":
                    for line in value:
                        print(f"    {line}")
                else:
                    print(f"    {check}: {value}")

    print("\n" + "=" * 100)
    print("FULL METADATA per family (first entry of each family)")
    names = sorted(set().union(*(set(e.properties.variable_metadata) for e in entries)))
    for name in names:
        values = {
            family: entries[indices[0]].properties.variable_metadata.get(name)
            for family, indices in families.items()
        }
        distinct = set(values.values())
        marker = "DIFF" if len(distinct) > 1 else "same"
        print(f"[{marker}] {name}")
        if len(distinct) > 1:
            for family, value in values.items():
                print(f"    {family:<8} {value}")
        else:
            print(f"    all: {next(iter(distinct))}")


def run(config_path: str):
    logging.basicConfig(level=logging.INFO)
    config_data = prepare_config(config_path)
    config = dacite.from_dict(
        data_class=TrainConfig, data=config_data, config=dacite.Config(strict=True)
    )
    entries = build_entries(config)
    report(entries)


def submit(config_path: str, name: str, clusters: list[str], dry_run: bool):
    config_rel = pathlib.Path(config_path).resolve().relative_to(REPO_ROOT)
    image = (REPO_ROOT / "latest_deps_only_image.txt").read_text().strip()
    cmd = [
        "gantry",
        "run",
        "--name",
        name,
        "--task-name",
        name,
        "--description",
        "Read-only diagnosis of DatasetProperties mismatches between datasets",
        "--beaker-image",
        image,
        "--workspace",
        "ai2/ace",
        "--priority",
        "high",
        "--preemptible",
        "--budget",
        "ai2/atec-climate",
        "--gpus",
        "0",
        "--shared-memory",
        "16GiB",
        "--weka",
        "climate-default:/climate-default",
        "--env",
        "FME_FORCE_CPU=1",
        "--system-python",
        "--install",
        "pip install --no-deps .",
        "--allow-dirty",
    ]
    for cluster in clusters:
        cmd += ["--cluster", cluster]
    cmd += ["--", "python", str(SCRIPT_REL), str(config_rel)]
    print("Submitting:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config_yaml", help="Training config to diagnose.")
    parser.add_argument(
        "--submit", action="store_true", help="Submit as a CPU beaker job."
    )
    parser.add_argument("--name", default="diagnose-dataset-properties")
    parser.add_argument("--cluster", nargs="+", default=["ai2/phobos"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit:
        submit(args.config_yaml, args.name, args.cluster, args.dry_run)
    else:
        with Distributed.context():
            run(args.config_yaml)


if __name__ == "__main__":
    main()

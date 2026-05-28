#!/usr/bin/env python
"""Compare parameter counts of SFNO and Swin Transformer configs.

Usage (from repo root, in the fme conda env):
    python configs/experiments/2026-05-28-swin-transformer/compare_model_sizes.py
"""

from pathlib import Path

import yaml

import fme.ace.registry  # noqa: F401 — registers all builders
from fme.core.dataset_info import DatasetInfo
from fme.core.registry.module import ModuleSelector


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


def count_params(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    step_cfg = cfg["stepper"]["step"]["config"]
    in_names: list[str] = step_cfg["in_names"]
    out_names: list[str] = step_cfg["out_names"]
    builder_cfg = step_cfg["builder"]

    dataset_info = DatasetInfo(img_shape=(45, 90), all_labels=set())
    selector = ModuleSelector(
        type=builder_cfg["type"],
        config=builder_cfg.get("config", {}),
    )
    module = selector.build(
        n_in_channels=len(in_names),
        n_out_channels=len(out_names),
        dataset_info=dataset_info,
    ).torch_module

    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "builder_type": builder_cfg["type"],
        "n_in": len(in_names),
        "n_out": len(out_names),
        "total": total,
        "trainable": trainable,
    }


def main() -> None:
    config_dir = Path(__file__).resolve().parent
    configs = {
        "SFNO": config_dir / "ace-train-config-4deg-AIMIP-sfno.yaml",
        "Swin": config_dir / "ace-train-config-4deg-AIMIP-nc-swin.yaml",
    }

    results = {}
    for label, path in configs.items():
        print(f"Building {label} ({path.name}) ...")
        info = count_params(path)
        results[label] = info
        print(f"  builder:   {info['builder_type']}")
        print(f"  channels:  {info['n_in']} in / {info['n_out']} out")
        print(f"  total:     {_fmt(info['total'])} ({info['total']:,})")
        print(f"  trainable: {_fmt(info['trainable'])} ({info['trainable']:,})")
        print()

    col = 14
    print(
        f"{'Model':<8} {'Builder':<24} {'In':>{col//2}} {'Out':>{col//2}}"
        f" {'Total':>{col}} {'Trainable':>{col}}"
    )
    print("-" * (8 + 24 + col // 2 * 2 + col * 2 + 5))
    for label, info in results.items():
        print(
            f"{label:<8} {info['builder_type']:<24}"
            f" {info['n_in']:>{col//2}} {info['n_out']:>{col//2}}"
            f" {_fmt(info['total']):>{col}} {_fmt(info['trainable']):>{col}}"
        )

    labels = list(results)
    if len(labels) == 2:
        a, b = labels
        ratio = results[a]["total"] / results[b]["total"]
        print(f"\n{a} / {b} parameter ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()

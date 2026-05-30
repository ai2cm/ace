#!/usr/bin/env python
"""Compare parameter counts of SFNO and Swin Transformer configs.

Usage (from repo root, in the fme conda env):
    python configs/experiments/2026-05-28-swin-transformer/compare_model_sizes.py
"""

from pathlib import Path

import pandas as pd
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


def _count_crossformer_params(step_cfg: dict) -> dict:
    import dacite

    from fme.ace.step.camulator import (
        CrossFormerSelector,
        NoiseConditionedCrossFormerSelector,
    )

    atmo_prog = step_cfg["atmosphere_prognostic_names"]
    atmo_levels = step_cfg["atmosphere_levels"]
    surf_prog = step_cfg["surface_prognostic_names"]
    forcing = step_cfg["forcing_names"]
    atmo_diag = step_cfg.get("atmosphere_diagnostic_names") or []
    surf_diag = step_cfg.get("surface_diagnostic_names") or []
    level_start = step_cfg.get("atmosphere_level_start", 0)

    atmo_in_names = [
        f"{name}_{i}"
        for name in atmo_prog
        for i in range(level_start, level_start + atmo_levels)
    ]
    atmo_out_names = atmo_in_names + [
        f"{name}_{i}"
        for name in atmo_diag
        for i in range(level_start, level_start + atmo_levels)
    ]
    in_names = forcing + surf_prog + atmo_in_names
    out_names = atmo_out_names + surf_prog + surf_diag

    builder_cfg = step_cfg["builder"]
    selector_cls = (
        NoiseConditionedCrossFormerSelector
        if builder_cfg["type"] == "NoiseConditionedCrossFormer"
        else CrossFormerSelector
    )
    selector = dacite.from_dict(
        data_class=selector_cls,
        data=builder_cfg,
        config=dacite.Config(strict=False),
    )
    module = selector.build(
        n_atmo_channels=len(atmo_prog),
        n_atmo_groups=atmo_levels,
        n_surf_channels=len(surf_prog),
        n_aux_channels=len(forcing),
        n_atmo_diagnostic_channels=len(atmo_diag),
        n_surf_diagnostic_channels=len(surf_diag),
        img_shape=(45, 90),
    )

    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "builder_type": builder_cfg["type"],
        "n_in": len(in_names),
        "n_out": len(out_names),
        "total": total,
        "trainable": trainable,
    }


def count_params(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    step = cfg["stepper"]["step"]
    step_cfg = step["config"]

    if step.get("type") == "CrossFormer":
        return _count_crossformer_params(step_cfg)

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
        "Swin": config_dir / "ace-train-config-4deg-AIMIP-swin.yaml",
        "NC-Swin": config_dir / "ace-train-config-4deg-AIMIP-nc-swin.yaml",
        "Camulator": config_dir / "ace-train-config-4deg-AIMIP-crossformer.yaml",
        "NC-Camulator": config_dir / "ace-train-config-4deg-AIMIP-nc-crossformer.yaml",
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

    rows = [
        {
            "Model": label,
            "Builder": info["builder_type"],
            "In": info["n_in"],
            "Out": info["n_out"],
            "Total": _fmt(info["total"]),
            "Trainable": _fmt(info["trainable"]),
        }
        for label, info in results.items()
    ]
    df = pd.DataFrame(rows).set_index("Model")
    print(df.to_string())

    labels = list(results)
    if len(labels) == 2:
        a, b = labels
        ratio = results[a]["total"] / results[b]["total"]
        print(f"\n{a} / {b} parameter ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()

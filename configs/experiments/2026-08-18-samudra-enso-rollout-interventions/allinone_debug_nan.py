#!/usr/bin/env python3
"""Localize the all-in-one arm's step-one NaN loss.

Builds the trainer exactly as fme.ace.train does, pulls the first training
batch, and runs it through train_on_batch with a NullOptimization (no weight
update, no NaN guard). Prints NaN counts per variable for the raw batch, the
per-channel losses, and the generated data, so the first NaN-bearing tensor
names itself.
"""

import dataclasses
import sys

import dacite
import torch

from fme.ace.train.train_config import TrainConfig
from fme.core.cli import prepare_config, prepare_directory
from fme.core.optimization import NullOptimization


def nan_report(label, mapping):
    bad = []
    for name, t in sorted(mapping.items()):
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            n = int(torch.isnan(t).sum())
            if n:
                bad.append(f"{name}={n}/{t.numel()}")
    print(f"[{label}] tensors with NaN:", "; ".join(bad) if bad else "NONE")


def main(yaml_path, override=None):
    config_data = prepare_config(yaml_path, override=override)
    config = dacite.from_dict(
        data_class=TrainConfig, data=config_data, config=dacite.Config(strict=True)
    )
    config.set_random_seed()
    config = dataclasses.replace(
        config,
        resume_results=prepare_directory(
            config.experiment_dir, config_data, config.resume_results
        ),
    )
    trainer = config.build_trainer()
    stepper = trainer.stepper

    batch = next(iter(trainer.train_data.loader))
    nan_report("raw batch", batch.data)

    stepped = stepper.train_on_batch(batch, NullOptimization())
    print("metrics:", {k: float(v) for k, v in stepped.metrics.items()})
    pcl = stepped.per_channel_losses
    if pcl is not None:

        def as_float(v):
            if isinstance(v, int | float):
                return float(v)
            if hasattr(v, "loss"):
                return float(v.loss.float().mean())
            return float(v.float().mean())

        vals = {k: as_float(v) for k, v in pcl.items()}
        top = sorted(vals.items(), key=lambda kv: -abs(kv[1]))[:12]
        print("top-12 per-channel losses:", [(k, f"{v:.3g}") for k, v in top])
    nan_report("gen_data", stepped.gen_data)
    nan_report("target_data", stepped.target_data)


if __name__ == "__main__":
    main(sys.argv[1], override=sys.argv[2:] or None)

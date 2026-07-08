#!/usr/bin/env python
"""
Bundle a mixture-of-experts denoising predictor into a single self-contained
checkpoint.

Reads a YAML describing a ``DenoisingMoEConfig`` (the list of expert
checkpoints, sigma ranges, and sampler parameters), builds the
``DenoisingMoEPredictor``, and saves it to one ``.pt`` file. That file can
then be loaded later via
``DenoisingMoECheckpointConfig(mixture_of_experts_path=...)`` with no need to
retain the original per-expert checkpoint paths or any rename mappings.

Usage:
    python bundle_denoising_moe_checkpoint.py <moe_config.yaml> <output.pt>

Example config (moe_config.yaml):
    num_diffusion_generation_steps: 18
    churn: 0.0
    denoising_expert_configs:
      - sigma_min: 0.002
        sigma_max: 1.0
        checkpoint_config:
          checkpoint_path: /path/to/low_noise_expert.ckpt
          rename: {x: renamed_x}
      - sigma_min: 1.0
        sigma_max: 80.0
        checkpoint_config:
          checkpoint_path: /path/to/high_noise_expert.ckpt
          rename: {x: renamed_x}
"""

import argparse
from typing import cast

import dacite
import yaml

from fme.core.device import get_device
from fme.downscaling.models import CheckpointModelConfig
from fme.downscaling.modules.physicsnemo_unets_v2.group_norm import apex_available
from fme.downscaling.predictors import DenoisingMoEConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to a YAML file describing a DenoisingMoEConfig.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to write the bundled MoE checkpoint (.pt).",
    )
    return parser.parse_args()


def _force_disable_apex_gn(cfg: CheckpointModelConfig) -> bool | None:
    """Set ``use_apex_gn=False`` on the loaded expert checkpoint's UNet config
    so the module can be built without apex. Returns the previous value so the
    caller can restore it in the saved bundle.

    Reaches into the loaded checkpoint dict because ``CheckpointModelConfig``
    intentionally refuses ``model_updates`` for the ``module`` field.
    """
    module_config = cfg._checkpoint["model"]["config"]["module"]["config"]
    prev = module_config.get("use_apex_gn")
    module_config["use_apex_gn"] = False
    return prev


def _force_disable_amp_bf16(cfg: CheckpointModelConfig) -> bool | None:
    """Set ``use_amp_bf16=False`` on the loaded expert checkpoint so the module
    can be built on devices that reject bf16 autocast (currently MPS). Returns
    the previous value so the caller can restore it in the saved bundle.
    """
    model_config = cfg._checkpoint["model"]["config"]
    prev = model_config.get("use_amp_bf16")
    model_config["use_amp_bf16"] = False
    return prev


def main(config_path: str, output_path: str) -> None:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    moe_config: DenoisingMoEConfig = dacite.from_dict(
        data_class=DenoisingMoEConfig,
        data=raw,
        config=dacite.Config(strict=True),
    )

    # apex's optimized GroupNorm isn't installable on macOS / CPU-only boxes;
    # bf16 autocast isn't supported on MPS. Build with both off where needed,
    # then restore the original settings in the saved bundle so a GPU consumer
    # can still use them when loading.
    on_mps = get_device().type == "mps"
    apex_originals: list[bool | None] = []
    bf16_originals: list[bool | None] = []
    if not apex_available():
        print("apex not available locally; building experts with use_apex_gn=False.")
        for rc in moe_config.denoising_expert_configs:
            apex_originals.append(_force_disable_apex_gn(rc.checkpoint_config))
    if on_mps:
        print("MPS does not support bf16 autocast; building with use_amp_bf16=False.")
        for rc in moe_config.denoising_expert_configs:
            bf16_originals.append(_force_disable_amp_bf16(rc.checkpoint_config))

    predictor = moe_config.build()

    if apex_originals:
        for expert, prev in zip(predictor._experts, apex_originals, strict=True):
            if prev is not None:
                cast(dict, expert.config.module.config)["use_apex_gn"] = prev
        print("Restored original use_apex_gn settings in the saved bundle.")
    if bf16_originals:
        for expert, prev in zip(predictor._experts, bf16_originals, strict=True):
            if prev is not None:
                expert.config.use_amp_bf16 = prev
        print("Restored original use_amp_bf16 settings in the saved bundle.")

    predictor.save(output_path)
    print(f"Wrote MoE bundle to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path, args.output_path)

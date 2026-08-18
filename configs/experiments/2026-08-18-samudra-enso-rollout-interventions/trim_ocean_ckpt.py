#!/usr/bin/env python3
"""Trim the nino-readout head out of the pretrained ocean checkpoint.

The wave-1 arms drop the 12 nino34_lead channels, which on the feature branch
were produced by a pooled-MLP head (`n_vector_outputs: 12`) that main's
Samudra does not have. The conv trunk is 86-wide in both worlds, so no tensor
needs slicing: the checkpoint only carries extra head parameters that
`overwrite_weights` would reject (its subset check runs before exclusions).
This script deletes those parameters and verifies the surviving keys against a
Samudra module built by *current* code with the wave-1 channel counts.

Usage:
    beaker dataset fetch 01KXKZ85HTDSGGXWD2DPW2QRFW \
        --prefix training_checkpoints/best_inference_ckpt.tar -o raw/
    python trim_ocean_ckpt.py \
        raw/training_checkpoints/best_inference_ckpt.tar ocean_ckpt_trimmed.tar
    beaker dataset create ocean_ckpt_trimmed.tar \
        --name samudra-nino-pretrain-trimmed-no-readout-head \
        --desc "066c88b best_inference_ckpt minus the nino34_lead MLP head"

Then mount the new dataset at /ocean_ckpt.tar in launch-wave1.sh.
"""

import sys

import torch
import yaml

from fme.ace.registry.m2lines import SamudraBuilder
from fme.core.dataset_info import DatasetInfo

BASELINE_FT = "baseline-ft-run-config.yaml"
N_IN = 95  # nino-lineage ocean in_names
N_OUT = 86  # out_names after dropping the 12 nino channels


def main(src: str, dst: str) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    # Locate the module state dict inside the stepper checkpoint.
    stepper = ckpt["stepper"]
    module_state = stepper["module"]

    # Reference: what current code expects for the trimmed channel counts.
    with open(BASELINE_FT) as f:
        cfg = yaml.safe_load(f)
    bcfg = cfg["stepper"]["ocean"]["stepper"]["step"]["config"]["builder"]["config"]
    bcfg.pop("n_vector_outputs", None)
    ref = SamudraBuilder(**bcfg).build(N_IN, N_OUT, DatasetInfo())
    ref_keys = {f"module.{k}" for k in ref.state_dict()}

    ckpt_keys = set(module_state.keys())
    extra = sorted(ckpt_keys - ref_keys)  # readout head -> delete
    missing = sorted(ref_keys - ckpt_keys)  # must be empty

    print(f"checkpoint params: {len(ckpt_keys)} | reference params: {len(ref_keys)}")
    print(f"extra in checkpoint (deleting): {len(extra)}")
    for k in extra:
        print("  -", k)
    if missing:
        raise SystemExit(
            f"reference expects {len(missing)} params the checkpoint lacks, "
            f"e.g. {missing[:4]} - architectures do not match, refusing to trim"
        )
    shape_mismatch = [
        k
        for k in ckpt_keys & ref_keys
        if tuple(module_state[k].shape)
        != tuple(ref.state_dict()[k[len("module.") :]].shape)
    ]
    if shape_mismatch:
        raise SystemExit(f"shape mismatches on shared keys: {shape_mismatch[:6]}")

    for k in extra:
        del module_state[k]
    torch.save(ckpt, dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

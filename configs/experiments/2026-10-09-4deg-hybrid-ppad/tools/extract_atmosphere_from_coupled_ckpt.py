#!/usr/bin/env python
"""Lift the atmosphere stepper out of a coupled training checkpoint into an
atmosphere-only checkpoint the single-module evaluator (fme.ace.evaluator) can load.

Mirror of configs/experiments/2026-09-18-residual-anchor-4deg/tools/
extract_ocean_from_coupled_ckpt.py
(branch troya/2026-08-18-enso-rollout-interventions): the coupled checkpoint stores
each component as a full Stepper state under ``stepper.ocean_state`` /
``stepper.atmosphere_state``; the uncoupled evaluator expects
``checkpoint["stepper"]`` to be a Stepper state. Same atmosphere weights the
coupled evaluator would use from this file, so "ACE after coupled fine-tuning"
can be scored by the ACE-only evaluators (eval-ace/, prescribed SST and sea ice).

    PYTHONPATH=<ace checkout> python extract_atmosphere_from_coupled_ckpt.py \
        coupled_ckpt.tar atmos_ckpt.tar
"""

import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
atmos = ck["stepper"]["atmosphere_state"]
out = {
    "stepper": atmos,
    "epoch": ck.get("epoch"),
    "num_batches_seen": ck.get("num_batches_seen"),
    "source": (
        "atmosphere_state of a coupled checkpoint "
        "(extract_atmosphere_from_coupled_ckpt)"
    ),
}
torch.save(out, dst)
# prove it loads as an atmosphere-only stepper
from fme.ace.stepper.single_module import Stepper  # noqa: E402

stepper = Stepper.from_state(
    torch.load(dst, map_location="cpu", weights_only=False)["stepper"]
)
print(
    f"wrote {dst}: epoch {out['epoch']}, atmosphere stepper loads; "
    f"{len(stepper.prognostic_names)} prognostic names, "
    f"first {sorted(stepper.prognostic_names)[:4]}"
)

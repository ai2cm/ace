#!/usr/bin/env python
"""Lift the ocean stepper out of a coupled training checkpoint into an ocean-only
checkpoint the single-module evaluator can load.

The coupled checkpoint stores each component as a full Stepper state under
``stepper.ocean_state`` / ``stepper.atmosphere_state``; the ocean-only evaluator
expects ``checkpoint["stepper"]`` to be a Stepper state. Same weights the coupled
evaluator would use from this file.

    python extract_ocean_from_coupled_ckpt.py coupled_ckpt.tar ocean_ckpt.tar
"""

import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
ocean = ck["stepper"]["ocean_state"]
out = {
    "stepper": ocean,
    "epoch": ck.get("epoch"),
    "num_batches_seen": ck.get("num_batches_seen"),
    "source": "ocean_state of a coupled checkpoint (extract_ocean_from_coupled_ckpt)",
}
torch.save(out, dst)
# prove it loads as an ocean-only stepper
from fme.ace.stepper.single_module import Stepper  # noqa: E402

stepper = Stepper.from_state(
    torch.load(dst, map_location="cpu", weights_only=False)["stepper"]
)
print(
    f"wrote {dst}: epoch {out['epoch']}, ocean stepper loads; "
    f"{len(stepper.prognostic_names)} prognostic names, "
    f"first {sorted(stepper.prognostic_names)[:4]}"
)

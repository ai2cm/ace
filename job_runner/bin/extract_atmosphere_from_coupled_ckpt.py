#!/usr/bin/env python
r"""Lift the atmosphere stepper out of a coupled training checkpoint into an
atmosphere-only checkpoint the single-module evaluator (fme.ace.evaluator) can load.

Mirror of configs/experiments/2026-09-18-residual-anchor-4deg/tools/
extract_ocean_from_coupled_ckpt.py
(branch troya/2026-08-18-enso-rollout-interventions): the coupled checkpoint stores
each component as a full Stepper state under ``stepper.ocean_state`` /
``stepper.atmosphere_state``; the uncoupled evaluator expects
``checkpoint["stepper"]`` to be a Stepper state. Same atmosphere weights the
coupled evaluator would use from this file, so "ACE after coupled fine-tuning"
can be scored by the ACE-only evaluators (eval-ace/, prescribed SST and sea ice).

Which weights: the trainer saves ``best_ckpt.tar`` / ``best_inference_ckpt.tar``
inside the EMA context (validate_using_ema), so those already hold the EMA
weights the inline evaluation scored. The latest ``ckpt.tar`` (the only
per-epoch file) holds the RAW weights, with the EMA parameters under the
checkpoint's ``ema`` key; pass ``--ema`` to apply them to the atmosphere
(the atmosphere is module 0 of the coupled stepper's ModuleList, so its EMA
names are ``"0." + module parameter name`` with the dots removed).

    PYTHONPATH=<ace checkout> python extract_atmosphere_from_coupled_ckpt.py \\
        coupled_ckpt.tar atmos_ckpt.tar [--ema]
"""

import argparse

import torch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument(
    "--ema",
    action="store_true",
    help="replace the atmosphere's trainable weights with the EMA weights stored "
    "in the checkpoint (needed for ckpt.tar; best_*_ckpt.tar already are EMA)",
)
args = ap.parse_args()
ck = torch.load(args.src, map_location="cpu", weights_only=False)
atmos = ck["stepper"]["atmosphere_state"]
n_ema = 0
if args.ema:
    ema = ck["ema"]
    params_key = [
        k
        for k in ema
        if k
        not in (
            "decay",
            "num_updates",
            "faster_decay_at_start",
            "module_name_to_ema_name",
        )
    ]
    assert len(params_key) == 1, f"no EMA params in checkpoint: {list(ema)}"
    ema_params = ema[params_key[0]]
    name_map = ema["module_name_to_ema_name"]
    module_state = atmos["step"]["module"]
    for k in list(module_state):
        ema_name = name_map.get(f"0.{k}")
        if ema_name is not None:
            assert module_state[k].shape == ema_params[ema_name].shape, k
            module_state[k] = ema_params[ema_name].clone()
            n_ema += 1
    assert n_ema > 0, "no atmosphere parameter matched the EMA name map"
out = {
    "stepper": atmos,
    "epoch": ck.get("epoch"),
    "num_batches_seen": ck.get("num_batches_seen"),
    "current_epoch_num_batches_seen": ck.get("current_epoch_num_batches_seen"),
    "source": (
        "atmosphere_state of a coupled checkpoint "
        f"(extract_atmosphere_from_coupled_ckpt, ema={args.ema}, {n_ema} EMA tensors)"
    ),
}
torch.save(out, args.dst)
# prove it loads as an atmosphere-only stepper
from fme.ace.stepper.single_module import Stepper  # noqa: E402

stepper = Stepper.from_state(
    torch.load(args.dst, map_location="cpu", weights_only=False)["stepper"]
)
print(
    f"wrote {args.dst}: epoch {out['epoch']}, batches {out['num_batches_seen']} "
    f"(+{out['current_epoch_num_batches_seen']} this epoch), ema={args.ema} "
    f"({n_ema} tensors); atmosphere stepper loads; "
    f"{len(stepper.prognostic_names)} prognostic names"
)

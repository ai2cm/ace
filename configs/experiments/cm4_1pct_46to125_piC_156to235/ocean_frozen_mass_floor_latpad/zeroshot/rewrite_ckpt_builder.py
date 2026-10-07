"""Copy a Samudra checkpoint with its stored builder padding options changed.

The evaluator builds the network from the config stored in the checkpoint and
has no override for builder options (StepperOverrideConfig covers ocean,
multi_call, derived_forcings, prescribed names and the corrector only). Since
``lat_pad`` and ``pad_to_pool_multiple`` add no parameters, a zero-shot arm is
the same weights under a rewritten ``stepper.config.step.config.builder``.

Usage (from the repo root, with this branch's fme importable):

    python configs/.../zeroshot/rewrite_ckpt_builder.py \
        best_inference_ckpt.tar out/training_checkpoints/best_inference_ckpt.tar \
        --lat-pad pole --pad-to-pool-multiple true

then upload ``out/`` as a beaker dataset and put its id in the arm's
``results_dataset`` column of zeroshot/experiments.txt (evaluate.sh mounts
``<dataset>:training_checkpoints/<ckpt>.tar``).

The rewritten state is rebuilt with ``Stepper.from_state`` and its weights and
EMA weights loaded before saving, so a mismatch fails here, not on the cluster.
"""

import argparse
import pathlib

import torch

from fme.ace.stepper.single_module import Stepper, load_ema_params_if_available


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--lat-pad", choices=["constant", "reflect", "pole"])
    parser.add_argument("--pad-to-pool-multiple", choices=["true", "false"])
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    builder = ckpt["stepper"]["config"]["step"]["config"]["builder"]
    if builder["type"] != "Samudra":
        raise ValueError(f"expected a Samudra builder, got {builder['type']}")
    print("before:", {k: builder["config"].get(k) for k in _KEYS})
    builder["config"]["lat_pad"] = args.lat_pad
    builder["config"]["pad_to_pool_multiple"] = args.pad_to_pool_multiple == "true"
    print("after: ", {k: builder["config"].get(k) for k in _KEYS})

    stepper = Stepper.from_state(ckpt["stepper"])  # rebuilds + loads weights
    load_ema_params_if_available(ckpt, stepper.modules, args.src)
    module = stepper.modules[0]
    torch_module = getattr(module, "module", module)
    print("built module:", type(torch_module).__name__)

    pathlib.Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.dst)
    print("wrote", args.dst)


_KEYS = ("zonally_periodic_upsample", "lat_pad", "pad_to_pool_multiple")

if __name__ == "__main__":
    main()

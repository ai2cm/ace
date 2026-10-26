"""Replay a FME_NAN_DEBUG dump in eager vs compiled to localize a non-finite op.

Usage: python scripts/nan_debug_replay.py <dump_dir> [n_repeats]

Loads ``module.pt`` (the pickled eager NoiseConditionedModel) and each
``rank*_event*.pt`` (records + state dict), then for every recorded call runs
the module with grad enabled in train mode:

- eager, restoring the recorded CUDA RNG state, ``n_repeats`` times;
- ``torch.compile``d, restoring the recorded CUDA RNG state, ``n_repeats``
  times;
- compiled per top-level stage (each child of ``conditional_model`` compiled on
  its own, the rest eager), with forward hooks on the stage boundaries to find
  the first stage whose compiled output is non-finite.
"""

import glob
import os
import sys

import torch

from fme.core.nan_debug import _stats


def _run(module, rec, device, restore_rng=True):
    if restore_rng and rec["cuda_rng"] is not None:
        torch.cuda.set_rng_state(rec["cuda_rng"])
    labels = None if rec["labels"] is None else rec["labels"].to(device)
    x = rec["input"].to(device)
    out = module(x, labels=labels)
    return out


def main():
    dump_dir = sys.argv[1]
    n_rep = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    torch.set_float32_matmul_precision(os.environ.get("MATMUL_PRECISION", "high"))
    device = torch.device("cuda")
    module = torch.load(os.path.join(dump_dir, "module.pt"), weights_only=False)
    module = module.to(device)
    module.train()
    compiled = torch.compile(module)
    for path in sorted(glob.glob(os.path.join(dump_dir, "rank*_event*.pt"))):
        dump = torch.load(path, weights_only=False)
        module.load_state_dict(dump["state_dict"])
        print(f"=== {path}: loss={dump['loss']}", flush=True)
        for rec in dump["records"]:
            print(
                f"call {rec['call_index']} orig output_finite={rec['output_finite']} "
                f"input={_stats(rec['input'])}",
                flush=True,
            )
            for name, m in (("eager", module), ("compiled", compiled)):
                for restore in (True, False):
                    n_bad = 0
                    for _ in range(n_rep):
                        out = _run(m, rec, device, restore_rng=restore)
                        if not torch.isfinite(out).all():
                            n_bad += 1
                        del out
                    print(
                        f"  {name} restore_rng={restore}: non-finite "
                        f"{n_bad}/{n_rep}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()

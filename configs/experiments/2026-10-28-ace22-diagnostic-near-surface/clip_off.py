"""Copy a checkpoint with latent global-mean clipping disabled.

The clip acts only at inference, and only while its envelope (_gm_min/_gm_max) is
finite, so setting the envelope to -inf/+inf turns it off and leaves the weights
unchanged.

    python clip_off.py <input checkpoint> <output checkpoint>
"""

import sys

import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = ckpt["stepper"]["step"]["module"]
keys = [k for k in state if k.endswith(("._gm_min", "._gm_max"))]
if sorted(k.rsplit(".", 1)[1] for k in keys) != ["_gm_max", "_gm_min"]:
    sys.exit(f"ERROR: expected one _gm_min and one _gm_max buffer, found {keys}")
for k in keys:
    if not torch.isfinite(state[k]).all():
        sys.exit(f"ERROR: {k} is already non-finite")
    state[k].fill_(float("-inf") if k.endswith("_gm_min") else float("inf"))
torch.save(ckpt, sys.argv[2])
print(f"epoch {ckpt['epoch']}: set {', '.join(keys)} to -inf/+inf")

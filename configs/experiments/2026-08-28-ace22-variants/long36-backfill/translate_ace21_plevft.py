"""Translate ACE2.1's plev-FT checkpoint to the current fme stepper schema.

ACE2.1 was trained at ace ref 70c966ed5, when the separate-decoder mechanism used two
flat fields on the step config. The mechanism was later restructured into a nested
`secondary_decoder` block. The semantics are unchanged: the docstring "diagnosed
directly from outputs without access to latent variables" is carried over verbatim.
A current-code load therefore fails with:

    UnexpectedDataError: can not match "additional_diagnostic_hidden_dim",
    "additional_diagnostic_names" to any data class field

Two changes, both mechanical:

  config   stepper.config.step.config.additional_diagnostic_names  (69 names)
           stepper.config.step.config.additional_diagnostic_hidden_dim = 256
        -> stepper.config.step.config.secondary_decoder = {
               secondary_diagnostic_names: <the 69 names>,
               network: {type: MLP, config: {hidden_dim: 256, depth: 2}}}

  weights  stepper.step["additional_diagnostic_module"]
             keys module.net.{0,2}.{weight,bias}
        -> stepper.step["secondary_decoder"]
             keys module.{0,2}.{weight,bias}

Only the "net." component comes out. The runtime wraps the decoder in
DistributedDataParallel, which contributes the surviving "module." prefix --
stripping it too yields "Missing key(s): module.0.weight / Unexpected key(s):
0.weight" at load time.
The check at the end of this script catches that class of error locally.

depth=2 is not inferred: the saved module holds exactly net.0 and net.2, i.e. two Linear
layers, and a freshly built MLP(hidden_dim=256, depth=2) produces the same key set
and the same shapes (256,44,1,1) / (69,256,1,1).
"""

import sys

import torch

SRC, DST = sys.argv[1], sys.argv[2]
ck = torch.load(SRC, map_location="cpu", weights_only=False)

step_cfg = ck["stepper"]["config"]["step"]["config"]
names = step_cfg.pop("additional_diagnostic_names")
hidden = step_cfg.pop("additional_diagnostic_hidden_dim")
step_cfg["secondary_decoder"] = {
    "secondary_diagnostic_names": names,
    "network": {"type": "MLP", "config": {"hidden_dim": hidden, "depth": 2}},
}

old = ck["stepper"]["step"].pop("additional_diagnostic_module")
PREFIX = "module.net."  # -> "module." : the runtime wraps the decoder in DDP
assert all(k.startswith(PREFIX) for k in old), f"unexpected keys: {sorted(old)}"
ck["stepper"]["step"]["secondary_decoder"] = {
    "module." + k[len(PREFIX) :]: v for k, v in old.items()
}

torch.save(ck, DST)
print(f"wrote {DST}")
print(f"  {len(names)} secondary diagnostics, hidden_dim={hidden}, depth=2")
print(f"  weight keys: {sorted(ck['stepper']['step']['secondary_decoder'])}")


# --- verify the result loads, rather than finding out in a 6-hour job ---------------
# Build the decoder the way SecondaryDecoderConfig will, wrap it as the runtime does,
# and load the translated weights strictly. A prefix or shape error fails in seconds.
import torch.nn as nn

from fme.core.dataset_info import DatasetInfo
from fme.core.registry.module import ModuleSelector


class _DDPLike(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.module = m


n_in = len(step_cfg["out_names"])
built = ModuleSelector(type="MLP", config={"hidden_dim": hidden, "depth": 2}).build(
    n_in_channels=n_in, n_out_channels=len(names), dataset_info=DatasetInfo()
)
sd = {
    k: v
    for k, v in ck["stepper"]["step"]["secondary_decoder"].items()
    if k != "label_encoding"
}
_DDPLike(built.torch_module).load_state_dict(sd, strict=True)
print(f"  verified: loads into a DDP-wrapped MLP({n_in} -> {hidden} -> {len(names)})")

"""Walk one hybridufsft training window step by step and name the first NaN.

Single process, batch 1: NaN-audit the batch, the loss-normalizer scales,
and each of the four predicted steps' outputs per variable.
"""

import sys

import dacite
import torch
import yaml

from fme.ace.train.train_config import TrainConfig
from fme.core.optimization import NullOptimization

c = yaml.safe_load(open(sys.argv[1]))
c["train_loader"]["batch_size"] = 1
c["train_loader"]["num_data_workers"] = 0
cfg = dacite.from_dict(TrainConfig, c, config=dacite.Config(strict=True))
train_data = cfg._get_train_data()
ts = cfg._get_stepper(dataset_info=train_data.dataset_info)
batch = next(iter(train_data.loader))
data = batch


def nanreport(td, label, top=8):
    bad = {}
    for k, v in dict(td).items():
        if torch.is_tensor(v) and torch.isnan(v).any():
            bad[k] = round(float(torch.isnan(v).float().mean()), 4)
    print(
        label,
        "->",
        dict(sorted(bad.items(), key=lambda kv: -kv[1])[:top]) if bad else "no NaN",
        flush=True,
    )


nanreport(data.data, "batch")

# loss normalizer scales audit
for attr in ("loss_normalizer", "_loss_normalizer"):
    ln = getattr(ts, attr, None)
    if ln is not None:
        stds = {k: float(v) for k, v in ln.stds.items()}
        weird = {k: v for k, v in stds.items() if not (v == v) or v <= 0}
        print(
            "loss normalizer stds <=0 or NaN:", weird if weird else "none", flush=True
        )
        print(
            "smallest stds:",
            dict(sorted(stds.items(), key=lambda kv: kv[1])[:5]),
            flush=True,
        )
        break

stepper = ts._stepper
prognostic_names = (
    ts._prognostic_names
    if hasattr(ts, "_prognostic_names")
    else stepper.prognostic_names
)
input_data = data.get_start(prognostic_names, 1).as_batch_data()
nanreport(input_data.data, "initial condition")
gen = stepper.predict_generator(
    input_data.data,
    data.data,
    4,
    NullOptimization(),
    labels=input_data.labels,
    data_mask=data.data_mask,
    stepper_state=input_data.stepper_state,
)
for step in range(4):
    out = next(gen)
    nanreport(out.output, f"step {step} output")
    deltas = dict(out.corrector_diagnostics.delta)
    if deltas:
        nanreport(deltas, f"step {step} corrector deltas")

"""Run one hybridufsft training step with NaN introspection.

Single process (world size 1), batch size 1: report NaN fractions in the
loaded batch, then run train_on_batch without loss validation and print
every per-channel loss so the NaN-carrying variables are named.
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
print("building train data...", flush=True)
train_data = cfg._get_train_data()
stepper = cfg._get_stepper(dataset_info=train_data.dataset_info)
print("pulling one batch...", flush=True)
batch = next(iter(train_data.loader))
bad = {
    k: round(float(torch.isnan(v).float().mean()), 4)
    for k, v in dict(batch.data).items()
    if torch.is_tensor(v) and torch.isnan(v).any()
}
print("batch NaN fractions:", bad if bad else "none", flush=True)
try:
    out = stepper.train_on_batch(batch, NullOptimization())
    metrics = getattr(out, "metrics", {}) or {}
    nan_metrics = {k: v for k, v in metrics.items() if v != v}
    fin = {k: round(float(v), 4) for k, v in metrics.items() if v == v}
    print("NaN metrics:", sorted(nan_metrics)[:40], flush=True)
    print("sample finite metrics:", dict(list(fin.items())[:10]), flush=True)
except Exception as e:
    print("train_on_batch raised:", type(e).__name__, str(e)[:300], flush=True)
    raise

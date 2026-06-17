"""Measure peak GPU memory of the 4deg/daily residual SFNO at mlp_ratio 1 vs 2.

Builds the *real* training stepper from a train-config YAML (the production
TrainBuilders path, so normalization/corrector/ocean/global-mean-removal are
all faithful), then measures peak GPU memory in two regimes for each
mlp_ratio:

  - training : one real ``TrainStepper.train_on_batch`` (forward x n_ensemble
               + backward + optimizer step), with the config's real optimizer
               (FusedAdam) and EnsembleLoss.
  - inference: one ``Stepper.predict`` step under ``torch.no_grad()`` (the
               forward-only path used at inference time).

The peak is captured with ``fme.core.benchmark.memory.benchmark_memory``
(wraps ``torch.cuda.reset_peak_memory_stats`` + ``max_memory_allocated`` /
``max_memory_reserved``). Both regimes run on one real training batch
(batch_size and ensemble taken from the config), so the only thing that
changes between the two columns is the MLP hidden width (embed_dim *
mlp_ratio).

A short warmup runs before each measurement so that cuda/cudnn workspace and
allocator caching do not land in the measured peak.

Usage (1 GPU):
    python scripts/benchmark_mlp_ratio_memory.py <train-config.yaml>
"""

import argparse
import copy
import gc

import dacite
import torch

import fme
from fme.core.benchmark.memory import benchmark_memory
from fme.core.cli import prepare_config
from fme.core.timing import GlobalTimer
from fme.ace.train.train_config import TrainBuilders, TrainConfig

GB = 1024.0**3


def _build_train_config(config_data: dict, mlp_ratio: float) -> TrainConfig:
    data = copy.deepcopy(config_data)
    builder_config = data["stepper"]["step"]["config"]["builder"]["config"]
    builder_config["mlp_ratio"] = mlp_ratio
    return dacite.from_dict(
        data_class=TrainConfig, data=data, config=dacite.Config(strict=True)
    )


def _measure(config_data: dict, mlp_ratio: float, batch, dataset_info, warmup: int):
    """Return (train_result, infer_result) MemoryResults for one mlp_ratio."""
    config = _build_train_config(config_data, mlp_ratio)
    builder = TrainBuilders(config)

    train_stepper = builder.get_stepper(dataset_info=dataset_info)
    optimization = builder.get_optimization(train_stepper.modules)
    stepper = train_stepper._stepper

    n_params = sum(p.numel() for m in train_stepper.modules for p in m.parameters())
    print(f"\n=== mlp_ratio={mlp_ratio} | {n_params/1e6:.1f}M params ===", flush=True)

    # --- training: forward + backward + optimizer step ---
    with GlobalTimer():
        for _ in range(warmup):
            train_stepper.train_on_batch(batch, optimization)
        torch.cuda.synchronize()
        with benchmark_memory() as bm_train:
            train_stepper.train_on_batch(batch, optimization)
    train_mem = bm_train.result
    print(
        f"  train     : max_alloc {train_mem.max_alloc/GB:6.3f} GB | "
        f"max_reserved {train_mem.max_reserved/GB:6.3f} GB",
        flush=True,
    )

    # --- inference: forward-only predict under no_grad ---
    n_ic = stepper.n_ic_timesteps
    initial_condition = batch.get_start(
        prognostic_names=stepper.prognostic_names, n_ic_timesteps=n_ic
    )
    with GlobalTimer(), torch.no_grad():
        for _ in range(warmup):
            stepper.predict(initial_condition, batch, compute_derived_variables=False)
        torch.cuda.synchronize()
        with benchmark_memory() as bm_infer:
            stepper.predict(initial_condition, batch, compute_derived_variables=False)
    infer_mem = bm_infer.result
    print(
        f"  inference : max_alloc {infer_mem.max_alloc/GB:6.3f} GB | "
        f"max_reserved {infer_mem.max_reserved/GB:6.3f} GB",
        flush=True,
    )

    del train_stepper, optimization, stepper, initial_condition
    gc.collect()
    torch.cuda.empty_cache()
    return train_mem, infer_mem


def main(config_path: str, warmup: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to measure peak GPU memory.")
    device = fme.get_device()
    print(f"device: {device} ({torch.cuda.get_device_name()})", flush=True)

    config_data = prepare_config(config_path, override=None)

    # Build one real training batch from the production train loader, shared
    # across both mlp_ratio measurements so the input is byte-identical.
    base_config = _build_train_config(config_data, mlp_ratio=2.0)
    train_data = TrainBuilders(base_config).get_train_data()
    dataset_info = train_data.dataset_info
    batch = next(iter(train_data.loader))
    print(
        f"batch: {batch.time.shape[0]} samples x {batch.time.shape[1]} timesteps; "
        f"img_shape={dataset_info.img_shape}",
        flush=True,
    )

    results = {}
    for mlp_ratio in (2.0, 1.0):
        results[mlp_ratio] = _measure(
            config_data, mlp_ratio, batch, dataset_info, warmup
        )

    def _table(attr):
        print(f"{'regime':<12}{'mlp_ratio=2':>14}{'mlp_ratio=1':>14}{'delta':>12}{'reduction':>12}")
        for i, regime in enumerate(("train", "inference")):
            a = getattr(results[2.0][i], attr) / GB
            b = getattr(results[1.0][i], attr) / GB
            pct = 100.0 * (a - b) / a if a > 0 else 0.0
            print(f"{regime:<12}{a:>14.3f}{b:>14.3f}{b-a:>+12.3f}{pct:>11.1f}%")

    print("\n================ PEAK GPU MEMORY (max_alloc, GB) ================")
    _table("max_alloc")
    print("\n================ PEAK GPU MEMORY (max_reserved, GB) =============")
    _table("max_reserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to a train-config YAML")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    main(args.config, args.warmup)

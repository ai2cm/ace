"""Time the real ACE training step (data, stepper, EMA) for an FM config.

Builds the trainer from a training YAML exactly as ``fme.ace.train`` does
(wandb in offline mode, so ``wandb.watch`` hooks are installed as in
production) and runs its per-batch loop on one GPU with the per-GPU batch
size of a 2-GPU run, reporting wall time per step split into data wait,
``train_on_batch`` and the EMA update, plus a profiler summary.

Usage:
    python bench_real.py <config.yaml> [--steps N] [--set key=value ...]
"""

import argparse
import dataclasses
import json
import os
import tempfile
import time
from collections import defaultdict

import dacite
import torch

from fme.ace.train.train_config import TrainConfig
from fme.core.cli import prepare_config


def _sync():
    torch.cuda.synchronize()


def _foreach_ema_call(self, model):
    """EMATracker.__call__ with the same math, as three foreach kernels and
    no host sync (the decay schedule stays on device).
    """
    self.num_updates += 1
    decay = self.decay
    if self._faster_decay_at_start:
        decay = torch.minimum(
            self.decay, (1 + self.num_updates) / (10 + self.num_updates)
        )
    with torch.no_grad():
        ema, params = [], []
        for key, p in model.named_parameters():
            if p.requires_grad:
                ema.append(self._ema_params[self._module_name_to_ema_name[key]])
                params.append(p)
        diff = torch._foreach_sub(ema, params)
        torch._foreach_mul_(diff, 1.0 - decay)
        torch._foreach_sub_(ema, diff)


def _patch_compile_mode(mode: str):
    import fme.core.registry.module as module_mod

    original = module_mod.compile_module

    def compile_with_mode(module, **kwargs):
        kwargs.setdefault("mode", mode)
        return original(module, **kwargs)

    module_mod.compile_module = compile_with_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--name", default="run")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--ema-impl", choices=["repo", "foreach"], default="repo")
    parser.add_argument("--compile-mode", default=None)
    args = parser.parse_args()
    if args.ema_impl == "foreach":
        from fme.core.ema import EMATracker

        EMATracker.__call__ = _foreach_ema_call  # type: ignore[method-assign]
    if args.compile_mode is not None:
        _patch_compile_mode(args.compile_mode)

    os.environ.setdefault("WANDB_MODE", "offline")
    overrides = [
        "train_loader.batch_size=4",
        f"experiment_dir={tempfile.mkdtemp()}",
        "logging.log_to_wandb=true",
        *args.set,
    ]
    data = prepare_config(args.config, override=overrides)
    config = dacite.from_dict(
        data_class=TrainConfig, data=data, config=dacite.Config(strict=True)
    )
    config.set_random_seed()
    torch.backends.cudnn.benchmark = True
    config.logging.configure_logging(
        config.experiment_dir,
        log_filename="out.log",
        config=dataclasses.asdict(config),
        resumable=False,
    )
    trainer = config.build_trainer()
    stepper = trainer.stepper
    optimization = trainer.optimization
    ema = trainer._ema
    train_data = trainer.train_data
    train_data.set_epoch(1)
    stepper.set_epoch(1)
    stepper.set_train()
    batches = iter(train_data.subset_loader(start_batch=0))

    def run(n, phases: dict | None = None):
        t_start = time.time()
        for _ in range(n):
            t0 = time.time()
            batch = next(batches)
            if phases is not None:
                _sync()
                t1 = time.time()
            stepper.train_on_batch(batch, optimization)
            if phases is not None:
                _sync()
                t2 = time.time()
            ema(model=stepper.modules)
            optimization.step_scheduler(is_iteration=True)
            if phases is not None:
                _sync()
                t3 = time.time()
                phases["data"] += t1 - t0
                phases["train_on_batch"] += t2 - t1
                phases["ema"] += t3 - t2
        _sync()
        return (time.time() - t_start) / n

    t0 = time.time()
    run(5)
    compile_s = time.time() - t0
    run(20)
    free_ms = 1000 * run(args.steps)
    phases: dict[str, float] = defaultdict(float)
    n_phase = 30
    run(n_phase, phases)
    result = dict(
        name=args.name,
        overrides=args.set,
        ema_impl=args.ema_impl,
        compile_mode=args.compile_mode,
        gpu=torch.cuda.get_device_name(),
        compile_s=round(compile_s),
        ms_per_step=round(free_ms, 1),
        samples_per_s_per_gpu=round(train_data.batch_size / (free_ms / 1000), 1),
        synced_phase_ms={k: round(1000 * v / n_phase, 1) for k, v in phases.items()},
        n_param_tensors=len(list(stepper.modules.parameters())),
    )
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        n_prof = 10
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            run(n_prof)
        kernel_ms = 0.0
        n_kernels = 0
        for evt in p.events():
            if (
                evt.device_type == torch.autograd.DeviceType.CUDA
                and not evt.name.startswith("##")
            ):
                kernel_ms += evt.device_time / 1000.0
                n_kernels += 1
        result["gpu_kernel_ms_per_step"] = round(kernel_ms / n_prof, 1)
        result["kernels_per_step"] = n_kernels // n_prof
        print(
            p.key_averages().table(sort_by="self_cpu_time_total", row_limit=30),
            flush=True,
        )
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

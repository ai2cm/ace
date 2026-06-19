"""Per-degree (l) gradient-magnitude probe for the SFNO spectral filter.

Loads a trained ACE checkpoint, pulls a handful of real training batches,
computes the exact single-step training loss (EnsembleLoss: CRPS + energy
score), backpropagates, and aggregates the gradient magnitude of the SFNO
spectral-filter weights as a function of spherical-harmonic degree l.

Motivation: train/power_spectrum/smallest_scale_norm_bias converges very
slowly. Hypothesis: small scales (high l) contribute little to the loss, so
their per-l filter weights see proportionally smaller gradients and update
slower than large-scale (low l) weights. The dhconv "linear" filter weight is
isotropic: shape (num_groups, modes_lat, c//g, c//g, 2), where dim 1 indexes
degree l and is shared across azimuthal order m. The per-l parameter count is
constant, so per-l gradient norms are directly comparable across l.

Outputs (to --out-dir):
  per_l_gradient_profile.npz  raw per-layer, per-l grad/weight arrays + metadata
  per_l_gradient_profile.png  log-log plot of per-l gradient magnitude
"""

import argparse
import contextlib
import os

import numpy as np
import torch
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import dacite  # noqa: E402

from fme.ace.train.train_config import TrainConfig, TrainBuilders  # noqa: E402
from fme.ace.stepper.single_module import load_stepper, TrainStepper  # noqa: E402
from fme.core.optimization import NullOptimization  # noqa: E402
from fme.core.device import get_device  # noqa: E402


class ProbeOptimization(NullOptimization):
    """Like NullOptimization but actually backpropagates and never zeroes grads.

    Training uses gradient accumulation, so the real Optimization backprops in
    accumulate_loss (per step) and detaches state between steps. We mirror that
    so the computed gradients match a real training step, but step_weights is a
    no-op so the .grad tensors survive for inspection and weights are unchanged.
    """

    def accumulate_loss(self, loss: torch.Tensor):
        self._accumulated_loss = self._accumulated_loss + loss.detach()
        loss.backward()

    def detach_if_using_gradient_accumulation(self, state):
        # Match training (use_gradient_accumulation=True): detach between steps.
        return {k: v.detach() for k, v in state.items()}

    def step_weights(self):
        # Do NOT step or zero grads: we want to read them.
        return

    def set_mode(self, modules):
        # Training runs the modules in train() mode (noise sampling active).
        for m in modules:
            m.train()


FILTER_SUFFIX = "filter.filter.weight"


def per_l_grad_sumsq(grad: torch.Tensor) -> torch.Tensor:
    """Sum of squared grad elements per degree l (dim 1), over all other dims."""
    dims = tuple(d for d in range(grad.ndim) if d != 1)
    return grad.detach().pow(2).sum(dim=dims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="resolved training config.yaml")
    ap.add_argument("--checkpoint", required=True, help="ckpt.tar")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-batches", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"device: {device}", flush=True)

    with open(args.config) as f:
        config_data = yaml.safe_load(f)
    config = dacite.from_dict(
        data_class=TrainConfig, data=config_data, config=dacite.Config(strict=False)
    )

    stepper = load_stepper(args.checkpoint)
    train_stepper = TrainStepper(stepper, config.stepper_training)
    modules = train_stepper.modules
    module = modules[0]
    module.to(device)

    # Identify the spectral-filter weight params and their l-extent.
    filter_names = [
        n for n, _ in module.named_parameters() if n.endswith(FILTER_SUFFIX)
    ]
    assert filter_names, "no spectral-filter weights found"
    params = dict(module.named_parameters())
    modes_lat = params[filter_names[0]].shape[1]
    print(f"found {len(filter_names)} filter weights, modes_lat (l)={modes_lat}",
          flush=True)
    for n in filter_names:
        print(f"  {n}: {tuple(params[n].shape)}", flush=True)

    # Static (trained) weight magnitude per l, per layer.
    weight_sumsq = {n: per_l_grad_sumsq(params[n].data).cpu().numpy()
                    for n in filter_names}

    builders = TrainBuilders(config)
    train_data = builders.get_train_data()
    loader = iter(train_data.loader)
    optimization = ProbeOptimization()

    # Accumulate per-batch per-l grad sum-of-squares per layer.
    grad_sumsq = {n: np.zeros(modes_lat) for n in filter_names}
    loss_vals = []
    n_done = 0
    for i in range(args.n_batches):
        try:
            batch = next(loader)
        except StopIteration:
            break
        module.zero_grad(set_to_none=True)
        out = train_stepper.train_on_batch(batch, optimization)
        loss = float(out.metrics["loss"])
        loss_vals.append(loss)
        for n in filter_names:
            g = params[n].grad
            if g is None:
                raise RuntimeError(f"no grad for {n}")
            grad_sumsq[n] += per_l_grad_sumsq(g).cpu().numpy()
        n_done += 1
        print(f"batch {i}: loss={loss:.5f}", flush=True)

    assert n_done > 0
    # Per-l RMS gradient (root mean over batches and over params at that l).
    # params-per-l is constant within a layer, so divide by that count too for
    # an interpretable per-parameter RMS.
    params_per_l = {n: params[n].numel() // modes_lat for n in filter_names}
    grad_rms = {n: np.sqrt(grad_sumsq[n] / n_done / params_per_l[n])
                for n in filter_names}
    weight_rms = {n: np.sqrt(weight_sumsq[n] / params_per_l[n])
                  for n in filter_names}

    layers = sorted(filter_names)
    grad_stack = np.stack([grad_rms[n] for n in layers])      # (n_layers, L)
    weight_stack = np.stack([weight_rms[n] for n in layers])
    grad_mean = grad_stack.mean(axis=0)
    weight_mean = weight_stack.mean(axis=0)
    rel = grad_mean / weight_mean

    ll = np.arange(modes_lat)
    np.savez(
        os.path.join(args.out_dir, "per_l_gradient_profile.npz"),
        l=ll, layers=np.array(layers),
        grad_rms=grad_stack, weight_rms=weight_stack,
        grad_rms_mean=grad_mean, weight_rms_mean=weight_mean,
        rel_update=rel, loss_vals=np.array(loss_vals),
        n_batches=n_done, modes_lat=modes_lat,
    )

    # Plot.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    for j, n in enumerate(layers):
        ax.loglog(ll[1:], grad_stack[j, 1:], color="0.7", lw=0.8)
    ax.loglog(ll[1:], grad_mean[1:], "k-", lw=2.2, label="mean over 8 layers")
    ax.set_xlabel("spherical-harmonic degree l")
    ax.set_ylabel("per-parameter gradient RMS")
    ax.set_title(f"Per-l gradient magnitude\n(mean of {n_done} train batches)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.loglog(ll[1:], weight_mean[1:], "b-", lw=2, label="weight RMS")
    ax.loglog(ll[1:], grad_mean[1:], "r-", lw=2, label="grad RMS")
    ax.set_xlabel("spherical-harmonic degree l")
    ax.set_ylabel("RMS magnitude")
    ax.set_title("Weight vs gradient magnitude")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    ax.loglog(ll[1:], rel[1:], "g-", lw=2)
    ax.set_xlabel("spherical-harmonic degree l")
    ax.set_ylabel("grad RMS / weight RMS")
    ax.set_title("Relative update rate per l\n(proxy for fractional step size)")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "SFNO spectral-filter per-l gradient profile "
        "(j8r0z322, epoch 120, single-step CRPS loss)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(args.out_dir, "per_l_gradient_profile.png"), dpi=130)
    print("wrote outputs to", args.out_dir, flush=True)

    # Verbal summary to stdout.
    print("\n=== SUMMARY ===", flush=True)
    print(f"l=1 grad RMS:        {grad_mean[1]:.3e}", flush=True)
    print(f"l={modes_lat-1} grad RMS:       {grad_mean[-1]:.3e}", flush=True)
    print(f"ratio low/high l:    {grad_mean[1]/grad_mean[-1]:.1f}x", flush=True)
    print(f"l=1 weight RMS:      {weight_mean[1]:.3e}", flush=True)
    print(f"l={modes_lat-1} weight RMS:     {weight_mean[-1]:.3e}", flush=True)
    print(f"rel update l=1:      {rel[1]:.3e}", flush=True)
    print(f"rel update l={modes_lat-1}:     {rel[-1]:.3e}", flush=True)


if __name__ == "__main__":
    main()

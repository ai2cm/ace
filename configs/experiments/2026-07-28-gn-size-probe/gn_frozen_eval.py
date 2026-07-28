"""Test whether GroupNorm's extent sensitivity explains downscaling bias.

Follow-up to gn_probe.py, which established that shrinking the input extent
below the trained ``coarse_shape`` moves GroupNorm's statistics substantially
(median |dmean| 0.75 over 122 layers at 4 deg vs 16 deg) while enlarging it does
not (0.11 at 32 deg). This script asks the causal question: if the small-input
run is handed the statistics the model would have seen at the trained extent,
does its bias go away?

Arms, run in order in one process so the captured statistics can be replayed:

  tc_16deg_capture  16x16 deg, live GroupNorm. In-distribution control, and the
                    source of the reference statistics.
  tc_04deg_live     4x4 deg, live GroupNorm. The biased case.
  tc_04deg_frozen   4x4 deg, GroupNorm statistics replaced by tc_16deg_capture's.
  tc_32deg_single   32x32 deg generated as ONE patch, no composite prediction.

Bias is reported against the fine truth over the common central 4x4 deg
footprint, so every arm is scored on identical ground.

Two guards on the machinery:

- ``check_input_shape_supported`` is never reached. It rejects extents below
  coarse_shape outright, and above it demands composite prediction; both are
  what we are deliberately measuring. Driving the predictor directly rather
  than through Downscaler/EventDownscaler sidesteps it without patching, and
  also means every arm generates as a single patch. The model is fully
  convolutional here (attn_resolutions=[], bottleneck_attention=false, no
  additive_pos_embed), so it runs at any extent whose fine grid divides by
  2**5.
- The frozen replacement recomputes the whole layer -- normalization, affine,
  and the activation some of these GroupNorms fuse. During the capture arm it
  recomputes each layer's output from the LIVE statistics and compares against
  what the layer actually returned, so any mismatch in eps, affine or fused
  activation handling is caught before the frozen numbers are believed.

Usage:
    python gn_frozen_eval.py <config.yaml>
"""

import argparse
import collections
import logging
import os

import dacite
import numpy as np
import torch
import yaml
from einops import rearrange
from torch.nn.functional import elu, gelu, leaky_relu, relu, sigmoid, silu, tanh

from fme.downscaling.evaluator import EvaluatorConfig

# The footprint every arm is scored on: the central 4x4 deg of the 16 deg event.
COMMON_LAT = (13.0, 17.0)
COMMON_LON = (136.0, 140.0)

_ACTIVATIONS = {
    "silu": silu,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "gelu": gelu,
    "elu": elu,
}


def _group_norm_modules(root: torch.nn.Module):
    """Yield ``(qualified_name, module, num_groups)`` for every GroupNorm."""
    for name, module in root.named_modules():
        if "groupnorm" not in type(module).__name__.lower():
            continue
        num_groups = getattr(module, "num_groups", None)
        if num_groups is None:
            logging.warning(
                f"skipping {name}: {type(module).__name__} exposes no num_groups"
            )
            continue
        yield name, module, int(num_groups)


def _activation_of(module: torch.nn.Module):
    """The activation a GroupNorm fuses into its output, or None.

    The vendored GroupNorm exposes the resolved callable as ``act_fn``; Apex's
    keeps only the name. Anything unrecognized raises rather than silently
    dropping an activation from the frozen path.
    """
    act_fn = getattr(module, "act_fn", None)
    if act_fn is not None:
        return act_fn
    act = getattr(module, "act", None)
    if act is None:
        return None
    if callable(act):
        return act
    resolved = _ACTIVATIONS.get(str(act).lower())
    if resolved is None:
        raise ValueError(f"unknown fused activation {act!r} on {type(module).__name__}")
    return resolved


def _normalize(x, num_groups, mean, var, module):
    """Reproduce a GroupNorm's eval-mode forward with supplied statistics.

    ``mean``/``var`` broadcast against the grouped tensor, so they may be either
    the live per-sample statistics or frozen per-group vectors.
    """
    grouped = rearrange(x, "b (g c) h w -> b g c h w", g=num_groups)
    normed = (grouped - mean) * (var + module.eps).rsqrt()
    out = rearrange(normed, "b g c h w -> b (g c) h w")
    weight = module.weight.to(out.dtype).reshape(1, -1, 1, 1)
    bias = module.bias.to(out.dtype).reshape(1, -1, 1, 1)
    out = out * weight + bias
    act_fn = _activation_of(module)
    return out if act_fn is None else act_fn(out)


class _Capture:
    """Records per-group statistics per call, and self-checks the replacement.

    GroupNorm reduces over (C/G, H, W) per sample; statistics are averaged over
    the sample dimension, which is a repeat of identical conditions, but kept
    separate per call because they vary strongly with noise level.
    """

    def __init__(self) -> None:
        self.stats: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = (
            collections.defaultdict(list)
        )
        self.max_reconstruction_error = 0.0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, name: str, module: torch.nn.Module, num_groups: int) -> None:
        def hook(mod, args, output):
            x = args[0]
            if not torch.is_tensor(x) or x.dim() != 4:
                return None
            grouped = rearrange(x, "b (g c) h w -> b g c h w", g=num_groups)
            live_mean = grouped.mean(dim=[2, 3, 4], keepdim=True)
            live_var = grouped.var(dim=[2, 3, 4], keepdim=True)

            # Self-check: rebuild this layer's output from its own live
            # statistics. Any gap means the replacement mishandles eps, the
            # affine parameters or a fused activation.
            rebuilt = _normalize(x, num_groups, live_mean, live_var, mod)
            err = (rebuilt - output).abs().max().item()
            self.max_reconstruction_error = max(self.max_reconstruction_error, err)

            # (B, G, 1, 1, 1) -> (G,), averaged over the sample dimension.
            self.stats[name].append(
                (
                    live_mean.mean(dim=0).detach().clone(),
                    live_var.mean(dim=0).detach().clone(),
                )
            )
            return None

        self._handles.append(module.register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class _Replay:
    """Replaces each GroupNorm's output using previously captured statistics."""

    def __init__(self, stats: dict[str, list[tuple[torch.Tensor, torch.Tensor]]]):
        self.stats = stats
        self.calls: dict[str, int] = collections.defaultdict(int)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, name: str, module: torch.nn.Module, num_groups: int) -> None:
        def hook(mod, args, output):
            x = args[0]
            if not torch.is_tensor(x) or x.dim() != 4:
                return None
            idx = self.calls[name]
            self.calls[name] += 1
            cached = self.stats.get(name)
            if cached is None or idx >= len(cached):
                raise RuntimeError(
                    f"no captured statistics for {name} call {idx} "
                    f"(captured {0 if cached is None else len(cached)}); "
                    "capture and replay arms must use the same n_samples"
                )
            mean, var = cached[idx]
            return _normalize(
                x,
                num_groups,
                mean.to(x.device, x.dtype),
                var.to(x.device, x.dtype),
                mod,
            )

        self._handles.append(module.register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _crop_index(coord: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    values = coord.detach().cpu()
    return torch.nonzero((values >= lo) & (values <= hi), as_tuple=True)[0]


def _common_footprint_bias(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    lat: torch.Tensor,
    lon: torch.Tensor,
) -> dict[str, float]:
    """Mean (prediction - truth) over the common central footprint, per variable."""
    y_idx = _crop_index(lat, *COMMON_LAT)
    x_idx = _crop_index(lon, *COMMON_LON)
    if len(y_idx) == 0 or len(x_idx) == 0:
        raise RuntimeError(
            "common footprint empty: "
            f"lat range {float(lat.min())}..{float(lat.max())}, "
            f"lon range {float(lon.min())}..{float(lon.max())}"
        )
    biases = {}
    for name, pred in prediction.items():
        if name not in target:
            continue
        p = pred.detach().float()
        # (B, S, H, W) -> average the sample dimension; (B, H, W) -> as is.
        if p.dim() == 4:
            p = p.mean(dim=1)
        t = target[name].detach().float()
        p = p[0][y_idx][:, x_idx]
        t = t[0][y_idx][:, x_idx]
        biases[name] = float((p - t).mean())
    return biases


def _run_arm(event, config: EvaluatorConfig, model, requirements, captured):
    """Run one arm; returns (bias dict, extra diagnostics)."""
    data = event.get_paired_gridded_data(
        base_data_config=config.data, requirements=requirements
    )
    mode = (
        "capture"
        if event.name.endswith("_capture")
        else "frozen"
        if event.name.endswith("_frozen")
        else "live"
    )
    logging.info(f"{event.name}: mode={mode}, n_samples={event.n_samples}")

    batch = next(iter(data.get_generator()))
    base_model = model.with_rolled_lon(batch[0].coarse.latlon_coordinates.lon)

    hooks: _Capture | _Replay | None = None
    if mode == "capture":
        hooks = _Capture()
    elif mode == "frozen":
        if not captured:
            raise RuntimeError("frozen arm ran before any capture arm")
        hooks = _Replay(captured)
    if hooks is not None:
        for name, module, num_groups in _group_norm_modules(base_model.modules):
            hooks.attach(name, module, num_groups)

    try:
        with torch.no_grad():
            outputs = base_model.generate_on_batch(batch, n_samples=event.n_samples)
    finally:
        if hooks is not None:
            hooks.remove()

    extra = {}
    if isinstance(hooks, _Capture):
        captured.update(hooks.stats)
        extra["max_reconstruction_error"] = hooks.max_reconstruction_error
        logging.info(
            f"{event.name}: captured {len(hooks.stats)} layers; "
            f"max replacement reconstruction error "
            f"{hooks.max_reconstruction_error:.3e}"
        )

    fine_coords = batch[0].fine.latlon_coordinates
    bias = _common_footprint_bias(
        outputs.prediction, batch[0].fine.data, fine_coords.lat, fine_coords.lon
    )
    logging.info(f"{event.name}: bias {bias}")
    return bias, extra


def main(config_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    config: EvaluatorConfig = dacite.from_dict(
        data_class=EvaluatorConfig, data=raw, config=dacite.Config(strict=True)
    )
    os.makedirs(config.experiment_dir, exist_ok=True)

    model = config.model.build()
    requirements = config.model.data_requirements
    logging.info(
        f"model coarse_shape={model.coarse_shape} fine_shape={model.fine_shape} "
        f"downscale_factor={model.downscale_factor}"
    )

    captured: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    results: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict] = {}

    for event in config.events or []:
        try:
            bias, extra = _run_arm(event, config, model, requirements, captured)
            results[event.name] = bias
            diagnostics[event.name] = extra
        except Exception:
            logging.exception(f"{event.name}: arm failed, continuing")

    variables = sorted({v for b in results.values() for v in b})
    lines = [f"common footprint: lat {COMMON_LAT}, lon {COMMON_LON}", ""]
    header = f"{'arm':<22}" + "".join(f"{v:>28}" for v in variables)
    lines += [header, "-" * len(header)]
    for arm, bias in results.items():
        lines.append(
            f"{arm:<22}"
            + "".join(f"{bias.get(v, float('nan')):>28.6g}" for v in variables)
        )
    for arm, extra in diagnostics.items():
        if "max_reconstruction_error" in extra:
            lines += [
                "",
                f"{arm}: max GroupNorm replacement reconstruction error "
                f"{extra['max_reconstruction_error']:.3e} "
                "(should be ~0; validates the frozen path)",
            ]

    summary = "\n".join(lines)
    with open(os.path.join(config.experiment_dir, "gn-frozen-summary.txt"), "w") as f:
        f.write(summary)
    np.savez_compressed(
        os.path.join(config.experiment_dir, "gn-frozen-bias.npz"),
        **{
            f"{arm}|{var}": np.array(val)
            for arm, b in results.items()
            for var, val in b.items()
        },
    )
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().config_path)

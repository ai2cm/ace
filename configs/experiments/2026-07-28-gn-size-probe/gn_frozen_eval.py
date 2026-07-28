"""Test whether GroupNorm's extent sensitivity explains downscaling bias.

Follow-up to gn_probe.py, which established that shrinking the input extent
below the trained ``coarse_shape`` moves GroupNorm's statistics substantially
(median |dmean| 0.75 over 122 layers at 4 deg vs 16 deg) while enlarging it does
not (0.11 at 32 deg). This script asks the causal question: if the small-input
run is handed the statistics the model would have seen at the trained extent,
does its bias go away?

Arms are events in the config, run in order in one process so captured
statistics can be replayed. The name SUFFIX selects the mode:

  *_capture  unmodified; also records reference statistics for the frozen arm.
             Use this on the arm whose extent equals the trained coarse_shape.
  *_live     unmodified. The case under test, as the model actually runs it.
  *_recomp   GroupNorm recomputed from its own LIVE statistics -- numerics
             control for the frozen arm.
  *_frozen   GroupNorm recomputed from the capture arm's statistics.
  (anything else runs unmodified, like *_live)

Because the frozen arm replays whatever the capture arm recorded in the same
process, one config = one location. gn-frozen-eval.yaml is the cyclone case and
gn-bland-eval.yaml the quiet-ocean control; they are separate jobs.

``recomp`` exists because this model runs under AMP bf16 against a fused Apex
GroupNorm kernel, and an unfused Python recomputation of the same arithmetic
does not reproduce it bit-for-bit. Comparing ``frozen`` against ``live`` would
therefore confound the frozen statistics with that kernel difference.
``recomp`` and ``frozen`` take the identical code path and differ only in which
statistics they use, so **frozen - recomp is the clean measurement**; live is
kept as the reference for how large the kernel difference actually is.

Bias is reported against the fine truth over the common central 4x4 deg
footprint, so every arm is scored on identical ground.

``check_input_shape_supported`` is never reached: it rejects extents below
coarse_shape outright and above it demands composite prediction, both of which
are what we are measuring. Driving the predictor directly rather than through
Downscaler/EventDownscaler sidesteps it without patching, and also means every
arm generates as a single patch. The model is fully convolutional here
(attn_resolutions=[], bottleneck_attention=false, no additive_pos_embed), so it
runs at any extent whose fine grid divides by 2**5.

Usage:
    python gn_frozen_eval.py <config.yaml>
"""

import argparse
import collections
import dataclasses
import logging
import os
from datetime import datetime, timedelta

import dacite
import numpy as np
import torch
import yaml
from einops import rearrange
from torch.nn.functional import elu, gelu, leaky_relu, relu, sigmoid, silu, tanh

from fme.core.dataset.time import TimeSlice
from fme.downscaling.evaluator import EvaluatorConfig

# The coarse data is 6-hourly.
TIMESTEP = timedelta(hours=6)

# The footprint every arm is scored on is derived from the config: the
# intersection of every event's extent, i.e. the smallest arm when they are
# concentric. Keeps the script location-agnostic so the same code runs the
# cyclone case and the quiet-ocean control.

# Bias is also profiled against distance from the domain boundary, in fine grid
# cells (1 cell ~ 1/32 deg ~ 3.5 km at downscale_factor 32). Every conv in this
# UNet zero-pads (F.conv2d(..., padding=N) throughout layers.py), so if that
# padding drives the small-domain bias, bias should fall off with distance from
# the edge -- and the 4 deg and 16 deg runs should trace the SAME curve over
# their shared range, the 4 deg domain simply having no cells beyond bin 32-64.
# A flat profile at 4 deg falsifies the padding hypothesis.
BIN_EDGES = [0, 4, 8, 16, 32, 64, 128, 256, 512]

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
    dropping an activation from the recomputed path.
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


def _group_stats(x: torch.Tensor, num_groups: int):
    """Per-sample, per-group mean and variance, shaped to broadcast."""
    grouped = rearrange(x, "b (g c) h w -> b g c h w", g=num_groups)
    return (
        grouped.mean(dim=[2, 3, 4], keepdim=True),
        grouped.var(dim=[2, 3, 4], keepdim=True),
    )


def _normalize(x, num_groups, mean, var, module):
    """Reproduce a GroupNorm's eval-mode forward with supplied statistics.

    Computed in float32 regardless of the incoming dtype: under AMP bf16 an
    unfused recomputation accumulates visibly more error than the fused kernel,
    and every arm that uses this path uses it identically, so the cast keeps the
    frozen-vs-recomputed comparison clean.
    """
    dtype = x.dtype
    x32 = x.float()
    grouped = rearrange(x32, "b (g c) h w -> b g c h w", g=num_groups)
    normed = (grouped - mean.float()) * (var.float() + module.eps).rsqrt()
    out = rearrange(normed, "b g c h w -> b (g c) h w")
    out = out * module.weight.float().reshape(1, -1, 1, 1)
    out = out + module.bias.float().reshape(1, -1, 1, 1)
    act_fn = _activation_of(module)
    if act_fn is not None:
        out = act_fn(out)
    return out.to(dtype)


class _Capture:
    """Records per-group statistics per call, and measures replacement fidelity.

    GroupNorm reduces over (C/G, H, W) per sample; statistics are averaged over
    the sample dimension, which is a repeat of identical conditions, but kept
    separate per call because they vary strongly with noise level.

    The recorded fidelity numbers say how closely the unfused recomputation
    tracks the model's own GroupNorm. They are diagnostics, not gates: the
    frozen measurement is made against the ``recomp`` arm, which shares this
    code path, so a nonzero gap here does not bias it.
    """

    def __init__(self) -> None:
        self.stats: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = (
            collections.defaultdict(list)
        )
        self.max_abs_error = 0.0
        self.max_rel_error = 0.0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, name: str, module: torch.nn.Module, num_groups: int) -> None:
        def hook(mod, args, output):
            x = args[0]
            if not torch.is_tensor(x) or x.dim() != 4:
                return None
            live_mean, live_var = _group_stats(x, num_groups)

            rebuilt = _normalize(x, num_groups, live_mean, live_var, mod)
            err = (rebuilt.float() - output.float()).abs().max().item()
            scale = output.float().abs().max().item()
            self.max_abs_error = max(self.max_abs_error, err)
            self.max_rel_error = max(self.max_rel_error, err / max(scale, 1e-12))

            # (B, G, 1, 1, 1) -> (G, 1, 1, 1), averaged over the sample dim.
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


class _Recompute:
    """Replaces each GroupNorm's output with an unfused recomputation.

    With ``stats=None`` it uses each call's own live statistics -- the numerics
    control. With captured statistics it replays them by call ordinal, which is
    the frozen condition. Both paths run identical arithmetic.
    """

    def __init__(
        self, stats: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] | None = None
    ):
        self.stats = stats
        self.calls: dict[str, int] = collections.defaultdict(int)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, name: str, module: torch.nn.Module, num_groups: int) -> None:
        def hook(mod, args, output):
            x = args[0]
            if not torch.is_tensor(x) or x.dim() != 4:
                return None
            if self.stats is None:
                mean, var = _group_stats(x, num_groups)
            else:
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
                mean = mean.to(x.device)
                var = var.to(x.device)
            return _normalize(x, num_groups, mean, var, mod)

        self._handles.append(module.register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _to_hw(t: torch.Tensor) -> torch.Tensor:
    """Collapse any leading batch/sample dimensions, averaging them."""
    t = t.detach().float()
    if t.dim() < 2:
        raise ValueError(
            f"expected at least 2 spatial dims, got shape {tuple(t.shape)}"
        )
    return t.reshape(-1, *t.shape[-2:]).mean(dim=0)


def _crop_index(coord: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    values = coord.detach().cpu()
    return torch.nonzero((values >= lo) & (values <= hi), as_tuple=True)[0]


def _common_footprint(events) -> tuple[tuple[float, float], tuple[float, float]]:
    """Intersection of every event's extent -- the ground all arms share."""
    lat = (
        max(float(e.lat_extent.start) for e in events),
        min(float(e.lat_extent.stop) for e in events),
    )
    lon = (
        max(float(e.lon_extent.start) for e in events),
        min(float(e.lon_extent.stop) for e in events),
    )
    if lat[0] >= lat[1] or lon[0] >= lon[1]:
        raise ValueError(f"event extents do not overlap: lat {lat}, lon {lon}")
    return lat, lon


def _footprint_stats(
    diffs: dict[str, torch.Tensor],
    lat: torch.Tensor,
    lon: torch.Tensor,
    footprint: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Domain-mean bias and spatial RMSE of the time-mean difference field.

    RMSE is reported alongside bias because a domain mean hides cancellation:
    offsetting errors of either sign average to nearly zero bias while leaving
    the field wrong everywhere.
    """
    y_idx = _crop_index(lat, *footprint[0])
    x_idx = _crop_index(lon, *footprint[1])
    if len(y_idx) == 0 or len(x_idx) == 0:
        raise RuntimeError(
            "common footprint empty: "
            f"lat range {float(lat.min())}..{float(lat.max())}, "
            f"lon range {float(lon.min())}..{float(lon.max())}"
        )
    bias, rmse = {}, {}
    for name, diff in diffs.items():
        cropped = diff[y_idx][:, x_idx]
        bias[name] = float(cropped.mean())
        rmse[name] = float(cropped.pow(2).mean().sqrt())
    return bias, rmse


def _build_paired_data(event, config: EvaluatorConfig, requirements, n_times: int):
    """Paired data for ``event``, spanning ``n_times`` steps centered on its date.

    EventConfig.get_paired_gridded_data hardcodes a 12h slice ("Event evaluation
    only load the first snapshot"), so a time mean needs its own construction.
    Mirrors that method otherwise.
    """
    center = datetime.strptime(event.date, event.date_format)
    half = (n_times - 1) // 2
    time_slice = TimeSlice(
        (center - half * TIMESTEP).strftime(event.date_format),
        (center + (n_times - 1 - half) * TIMESTEP).strftime(event.date_format),
    )
    fine = dataclasses.replace(config.data.fine[0])
    fine.update_subset(time_slice)
    coarse = dataclasses.replace(config.data.coarse_full_config[0])
    coarse.update_subset(time_slice)
    data_config = dataclasses.replace(
        config.data,
        fine=[fine],
        coarse=[coarse],
        repeat=1,
        batch_size=1,
        lat_extent=event.lat_extent,
        lon_extent=event.lon_extent,
    )
    return data_config.build(train=False, requirements=requirements)


def _edge_distance(height: int, width: int) -> torch.Tensor:
    """Distance from each cell to the nearest domain boundary, in cells."""
    rows = torch.arange(height).reshape(-1, 1).expand(height, width)
    cols = torch.arange(width).reshape(1, -1).expand(height, width)
    return torch.minimum(
        torch.minimum(rows, height - 1 - rows), torch.minimum(cols, width - 1 - cols)
    )


def _bias_profile(diffs: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    """Mean bias per edge-distance bin, over the arm's whole domain.

    Unlike the footprint scalars, this uses every cell the arm generated: the
    question is how bias varies with proximity to that arm's own boundary.
    Bins beyond the domain's reach are NaN.
    """
    profiles: dict[str, list[float]] = {}
    for name, diff in diffs.items():
        flat = diff.cpu().flatten()
        radius = _edge_distance(*diff.shape).flatten()
        row = []
        for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
            mask = (radius >= lo) & (radius < hi)
            row.append(float(flat[mask].mean()) if bool(mask.any()) else float("nan"))
        profiles[name] = row
    return profiles


def _run_arm(event, config, model, requirements, captured, footprint, n_times):
    """Run one arm over ``n_times`` steps; returns (bias, rmse, extra, profile)."""
    data = _build_paired_data(event, config, requirements, n_times)
    if event.name.endswith("_capture"):
        mode = "capture"
    elif event.name.endswith("_frozen"):
        mode = "frozen"
    elif event.name.endswith("_recomp"):
        mode = "recomp"
    else:
        mode = "live"
    logging.info(f"{event.name}: mode={mode}, n_samples={event.n_samples}")

    hooks: _Capture | _Recompute | None = None
    if mode == "capture":
        hooks = _Capture()
    elif mode == "recomp":
        hooks = _Recompute(None)
    elif mode == "frozen":
        if not captured:
            raise RuntimeError("frozen arm ran before any capture arm")
        hooks = _Recompute(captured)

    # Accumulate the difference field over time. Averaging (prediction - truth)
    # over time equals the difference of the time means, so this is the bias of
    # the time-mean field.
    totals: dict[str, torch.Tensor] = {}
    fine_coords = None
    attached = False
    n_seen = 0

    for step, batch in enumerate(data.get_generator()):
        if step >= n_times:
            break
        base_model = model.with_rolled_lon(batch[0].coarse.latlon_coordinates.lon)
        # Hooks bind to modules, which are shared across steps: attach once, and
        # let the capture/replay call ordinals run continuously so that step k of
        # the frozen arm replays step k of the capture arm.
        if hooks is not None and not attached:
            for name, module, num_groups in _group_norm_modules(base_model.modules):
                hooks.attach(name, module, num_groups)
            attached = True

        with torch.no_grad():
            outputs = base_model.generate_on_batch(batch, n_samples=event.n_samples)

        for name, pred in outputs.prediction.items():
            if name not in outputs.target:
                continue
            diff = _to_hw(pred) - _to_hw(outputs.target[name])
            totals[name] = diff if name not in totals else totals[name] + diff
        fine_coords = batch[0].fine.latlon_coordinates
        n_seen += 1

    if hooks is not None:
        hooks.remove()
    if n_seen == 0 or fine_coords is None:
        raise RuntimeError(f"{event.name}: generator yielded no timesteps")
    if n_seen < n_times:
        logging.warning(f"{event.name}: only {n_seen} of {n_times} steps available")

    diffs = {k: v / n_seen for k, v in totals.items()}

    extra: dict[str, float] = {"n_times": float(n_seen)}
    if isinstance(hooks, _Capture):
        captured.update(hooks.stats)
        extra["max_abs_error"] = hooks.max_abs_error
        extra["max_rel_error"] = hooks.max_rel_error
        logging.info(
            f"{event.name}: captured {len(hooks.stats)} layers; recomputation "
            f"vs fused kernel max abs {hooks.max_abs_error:.3e}, "
            f"max rel {hooks.max_rel_error:.3e}"
        )

    bias, rmse = _footprint_stats(diffs, fine_coords.lat, fine_coords.lon, footprint)
    logging.info(f"{event.name}: {n_seen} steps; bias {bias}; rmse {rmse}")
    return bias, rmse, extra, _bias_profile(diffs)


def main(config_path: str, n_times: int) -> None:
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
    footprint = _common_footprint(config.events or [])
    logging.info(
        f"common footprint: lat {footprint[0]}, lon {footprint[1]}; "
        f"time mean over {n_times} steps"
    )

    captured: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    biases: dict[str, dict[str, float]] = {}
    rmses: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    profiles: dict[str, dict[str, list[float]]] = {}

    for event in config.events or []:
        try:
            bias, rmse, extra, profile = _run_arm(
                event, config, model, requirements, captured, footprint, n_times
            )
            biases[event.name] = bias
            rmses[event.name] = rmse
            diagnostics[event.name] = extra
            profiles[event.name] = profile
        except Exception:
            logging.exception(f"{event.name}: arm failed, continuing")

    variables = sorted({v for b in biases.values() for v in b})
    lines = [
        f"common footprint: lat {footprint[0]}, lon {footprint[1]}",
        f"time mean over {n_times} steps (6-hourly), centered on each event date",
        "frozen vs recomp is the clean contrast; RMSE is spatial, over the",
        "time-mean field, and catches error that a domain mean cancels away",
    ]
    for label, table in (
        ("bias = mean(prediction - truth)", biases),
        ("spatial RMSE", rmses),
    ):
        lines += ["", label]
        header = f"{'arm':<22}" + "".join(f"{v:>30}" for v in variables)
        lines += [header, "-" * len(header)]
        for arm in biases:
            row = table.get(arm, {})
            lines.append(
                f"{arm:<22}"
                + "".join(f"{row.get(v, float('nan')):>30.6g}" for v in variables)
            )
    for arm, extra in diagnostics.items():
        if "max_abs_error" in extra:
            lines += [
                "",
                f"{arm}: unfused recomputation vs fused GroupNorm kernel -- "
                f"max abs {extra['max_abs_error']:.3e}, "
                f"max rel {extra['max_rel_error']:.3e}",
            ]

    bin_labels = [f"{lo}-{hi}" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]
    lines += [
        "",
        "",
        "bias vs distance from domain boundary, in fine cells (1 cell ~ 1/32 deg)",
    ]
    for var in variables:
        lines += ["", f"--- {var} ---"]
        header = f"{'arm':<22}" + "".join(f"{b:>14}" for b in bin_labels)
        lines += [header, "-" * len(header)]
        for arm in biases:
            profile_row = profiles.get(arm, {}).get(var)
            if profile_row is None:
                continue
            cells = "".join(
                "             -" if np.isnan(v) else f"{v:>14.5g}" for v in profile_row
            )
            lines.append(f"{arm:<22}{cells}")

    summary = "\n".join(lines)
    with open(os.path.join(config.experiment_dir, "gn-frozen-summary.txt"), "w") as f:
        f.write(summary)
    arrays: dict[str, np.ndarray] = {}
    for kind, table in (("bias", biases), ("rmse", rmses)):
        arrays.update(
            {
                f"{kind}|{arm}|{var}": np.array(val)
                for arm, d in table.items()
                for var, val in d.items()
            }
        )
    arrays.update(
        {
            f"profile|{arm}|{var}": np.array(val)
            for arm, pr in profiles.items()
            for var, val in pr.items()
        }
    )
    arrays["profile_bin_edges"] = np.array(BIN_EDGES)
    np.savez_compressed(
        os.path.join(config.experiment_dir, "gn-frozen-bias.npz"), **arrays
    )
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=str, help="Path to the config file")
    parser.add_argument(
        "--n-times",
        type=int,
        default=1,
        help=(
            "Number of 6-hourly steps to average, centered on each event date. "
            "1 reproduces the single-snapshot behaviour."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    main(_args.config_path, _args.n_times)

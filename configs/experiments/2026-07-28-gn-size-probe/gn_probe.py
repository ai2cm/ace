"""Probe GroupNorm input statistics as a function of input spatial extent.

Diagnostic for the hypothesis that a downscaling SongUNetv2's GroupNorm layers
see out-of-sample statistics when the inference input extent differs from the
extent the model was trained on (its ``coarse_shape``). GroupNorm reduces over
(C/G, H, W), so its per-sample statistics are a summary over whatever spatial
extent it is handed. On a statistically inhomogeneous field, changing the extent
changes those statistics -- and every downstream activation with them.

For each event in the config -- here, one tropical-cyclone snapshot cropped to
several extents about the same center -- this runs the real generation path and
records, for every GroupNorm layer and every denoiser call, the per-group mean
and variance of that layer's *input*.

Two things this deliberately does not do:

- It does not modify the model. Statistics are recomputed in a forward-pre hook
  from the input tensor, exactly as GroupNorm would compute them internally, so
  Apex GroupNorm stays enabled and the numerics are those of a normal run.
- It does not go through ``EventDownscaler``, which would apply
  ``check_input_shape_supported`` and reject any extent below the trained
  ``coarse_shape``. Sub-``coarse_shape`` extents are the point of the probe, so
  the predictor is driven directly.

Writes one ``gn-stats-<event>.npz`` per event into ``experiment_dir``, plus a
``gn-probe-summary.txt`` comparing every extent against the reference extent
(the one matching the model's ``coarse_shape``).

Usage:
    python gn_probe.py <config.yaml>
"""

import argparse
import collections
import logging
import os

import dacite
import numpy as np
import torch
import yaml

from fme.downscaling.predict import DownscalerConfig

# Written next to the per-event npz files; see module docstring.
SUMMARY_FILENAME = "gn-probe-summary.txt"


def _group_norm_modules(root: torch.nn.Module):
    """Yield ``(qualified_name, module, num_groups)`` for every GroupNorm.

    Matches on the type name so both the vendored ``GroupNorm`` and Apex's
    ``GroupNorm`` are picked up. ``num_groups`` is read off the module because
    both classes may adjust the requested group count to satisfy
    ``min_channels_per_group``.
    """
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


class _StatRecorder:
    """Records per-group input statistics for each GroupNorm, per call.

    One "call" is one invocation of the layer, i.e. one denoiser evaluation.
    The sampler makes several per generated sample (18 steps, and a 2nd-order
    sampler evaluates twice per step), and GroupNorm statistics vary strongly
    with noise level, so calls are kept separate rather than pooled. The sample
    dimension *is* averaged over -- it is a repeat of the same conditions.
    """

    def __init__(self) -> None:
        self.records: dict[str, list[tuple[np.ndarray, np.ndarray, int, int]]] = (
            collections.defaultdict(list)
        )
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, name: str, module: torch.nn.Module, num_groups: int) -> None:
        def hook(_module, args):
            if not args or not torch.is_tensor(args[0]):
                return
            x = args[0]
            if x.dim() != 4:
                return
            n, _, height, width = x.shape
            grouped = x.detach().float().reshape(n, num_groups, -1)
            self.records[name].append(
                (
                    grouped.mean(dim=-1).mean(dim=0).cpu().numpy(),
                    grouped.var(dim=-1, unbiased=False).mean(dim=0).cpu().numpy(),
                    height,
                    width,
                )
            )

        self._handles.append(module.register_forward_pre_hook(hook))

    @property
    def n_hooked(self) -> int:
        return len(self._handles)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for name, calls in self.records.items():
            arrays[f"{name}|mean"] = np.stack([c[0] for c in calls])
            arrays[f"{name}|var"] = np.stack([c[1] for c in calls])
            arrays[f"{name}|hw"] = np.array([[c[2], c[3]] for c in calls])
        return arrays


def _summarize(
    per_event: dict[str, dict[str, np.ndarray]], reference: str
) -> list[str]:
    """Compare each event's statistics against the reference event's.

    Reports, per layer, the mean absolute shift in group mean and the mean ratio
    of group standard deviations. A layer whose statistics are extent-invariant
    scores a shift near 0 and a ratio near 1.
    """
    lines = [f"reference extent: {reference}", ""]
    ref = per_event[reference]
    layers = sorted({k.rsplit("|", 1)[0] for k in ref})
    for event, arrays in per_event.items():
        if event == reference:
            continue
        lines.append(f"=== {event} vs {reference} ===")
        lines.append(f"{'layer':<58} {'d_mean':>10} {'std_ratio':>10} {'calls':>6}")
        for layer in layers:
            mean_key, var_key = f"{layer}|mean", f"{layer}|var"
            if mean_key not in arrays:
                continue
            n_calls = min(len(ref[mean_key]), len(arrays[mean_key]))
            if n_calls == 0:
                continue
            d_mean = float(
                np.abs(arrays[mean_key][:n_calls] - ref[mean_key][:n_calls]).mean()
            )
            ref_std = np.sqrt(ref[var_key][:n_calls]) + 1e-12
            std_ratio = float((np.sqrt(arrays[var_key][:n_calls]) / ref_std).mean())
            lines.append(f"{layer:<58} {d_mean:>10.4f} {std_ratio:>10.4f} {n_calls:>6}")
        lines.append("")
    return lines


def main(config_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    config: DownscalerConfig = dacite.from_dict(
        data_class=DownscalerConfig, data=raw, config=dacite.Config(strict=True)
    )
    os.makedirs(config.experiment_dir, exist_ok=True)

    model = config.model.build()
    requirements = config.model.data_requirements
    logging.info(
        f"model coarse_shape = {model.coarse_shape}, "
        f"fine_shape = {model.fine_shape}, "
        f"downscale_factor = {model.downscale_factor}"
    )

    per_event: dict[str, dict[str, np.ndarray]] = {}
    reference: str | None = None

    for event in config.events or []:
        # The largest extents are the ones at risk of exhausting memory. Isolate
        # each event so a late failure still leaves the earlier extents' npz
        # files (already written) and a summary over whatever completed.
        try:
            per_event[event.name], is_reference = _probe_event(
                event, config, model, requirements
            )
            if is_reference:
                reference = event.name
        except Exception:
            logging.exception(f"{event.name}: probe failed, continuing")

    if reference is None:
        logging.warning(
            "no event matched the model coarse_shape; skipping the comparison "
            "summary (the per-event npz files are still written)"
        )
        return

    lines = _summarize(per_event, reference)
    summary_path = os.path.join(config.experiment_dir, SUMMARY_FILENAME)
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logging.info(f"wrote {summary_path}")
    print("\n".join(lines))


def _probe_event(
    event, config: DownscalerConfig, model, requirements
) -> tuple[dict[str, np.ndarray], bool]:
    """Run one extent and return its statistics plus whether it is the reference."""
    data = event.get_gridded_data(
        base_data_config=config.data, requirements=requirements
    )
    is_reference = tuple(data.shape) == tuple(model.coarse_shape)
    logging.info(
        f"{event.name}: input shape {data.shape}, "
        f"model coarse_shape {model.coarse_shape}"
        f"{' -- reference' if is_reference else ''}"
    )

    # Drive the predictor directly: EventDownscaler would reject any extent
    # below coarse_shape, which is exactly what we want to measure.
    base_model = model.with_rolled_lon(data.coarse_extent_latlon_coords.lon)

    recorder = _StatRecorder()
    for name, module, num_groups in _group_norm_modules(base_model.modules):
        recorder.attach(name, module, num_groups)
    logging.info(f"{event.name}: hooked {recorder.n_hooked} GroupNorm layers")

    batch = next(iter(data.get_generator()))
    try:
        with torch.no_grad():
            base_model.generate_on_batch_no_target(batch, n_samples=event.n_samples)
    finally:
        recorder.remove()

    arrays = recorder.to_arrays()
    out_path = os.path.join(config.experiment_dir, f"gn-stats-{event.name}.npz")
    np.savez_compressed(out_path, **arrays)
    logging.info(f"{event.name}: wrote {out_path}")
    return arrays, is_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().config_path)

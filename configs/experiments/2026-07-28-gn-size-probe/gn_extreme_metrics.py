"""Do frozen GroupNorm statistics change cyclone extremes at off-training extents?

Question: generating around a cyclone at 4x4 and 32x32 deg, with GroupNorm
statistics frozen or live, do max 10m wind speed and mean precipitation rate
within the central 4x4 deg region differ significantly from the 16x16 deg
generation -- the control, being the extent the model was trained on?

Domain-mean bias, used by gn_frozen_eval.py, is the wrong metric for this: it
dilutes a cyclone across mostly-ambient cells. These metrics target the feature.

Arms come from the config; the name suffix selects the GroupNorm mode, as in
gn_frozen_eval.py (*_capture / *_live / *_frozen). Every arm must share
n_samples and SAMPLE_GROUP so that the frozen arms' call ordinals line up with
the capture arm's.

Significance is across the diffusion ensemble: each sample is an independent
draw, so per-sample metric values give a distribution per arm, compared against
the control with Welch's t-test. This tests whether an arm's extremes differ
from the control's for THIS snapshot; snapshot-to-snapshot robustness is a
separate question needing several cyclones.

Usage:
    python gn_extreme_metrics.py <config.yaml>
"""

import argparse
import logging
import os

import dacite
import numpy as np
import torch
import yaml
from gn_frozen_eval import (
    _Capture,
    _common_footprint,
    _crop_index,
    _group_norm_modules,
    _Recompute,
)
from scipy import stats

from fme.downscaling.evaluator import EvaluatorConfig

# Samples per generate_on_batch call. Bounds peak memory at 32x32 deg, where the
# fine grid is 1024x1024. Must match across arms: it sets how many denoiser
# calls each layer sees, which is what the frozen replay is keyed on.
SAMPLE_GROUP = 8

U_NAME = "eastward_wind_at_ten_meters"
V_NAME = "northward_wind_at_ten_meters"
PRECIP_NAME = "PRATEsfc"


def _crop(field: torch.Tensor, y_idx, x_idx) -> torch.Tensor:
    """Crop trailing spatial dims to the footprint, preserving leading dims."""
    return field[..., y_idx, :][..., :, x_idx]


def _sample_metrics(
    prediction: dict[str, torch.Tensor], y_idx, x_idx
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample (max wind speed, mean precip) over the footprint.

    Prediction tensors are (batch, sample, y, x) with batch 1 for event data.
    """
    u = _crop(prediction[U_NAME][0].detach().float(), y_idx, x_idx)
    v = _crop(prediction[V_NAME][0].detach().float(), y_idx, x_idx)
    speed = torch.sqrt(u * u + v * v)
    max_wind = speed.amax(dim=(-2, -1))
    precip = _crop(prediction[PRECIP_NAME][0].detach().float(), y_idx, x_idx).mean(
        dim=(-2, -1)
    )
    return max_wind.cpu().numpy(), precip.cpu().numpy()


def _truth_metrics(
    target: dict[str, torch.Tensor], y_idx, x_idx
) -> tuple[float, float]:
    """The same two metrics computed on the fine truth."""
    u = _crop(target[U_NAME][0, 0].detach().float(), y_idx, x_idx)
    v = _crop(target[V_NAME][0, 0].detach().float(), y_idx, x_idx)
    precip = _crop(target[PRECIP_NAME][0, 0].detach().float(), y_idx, x_idx)
    return float(torch.sqrt(u * u + v * v).max()), float(precip.mean())


def _run_arm(event, config, model, requirements, captured, footprint):
    """Generate ``event.n_samples`` samples in groups; return per-sample metrics."""
    data = event.get_paired_gridded_data(
        base_data_config=config.data, requirements=requirements
    )
    if event.name.endswith("_capture"):
        mode = "capture"
    elif event.name.endswith("_frozen"):
        mode = "frozen"
    elif event.name.endswith("_recomp"):
        mode = "recomp"
    else:
        mode = "live"

    batch = next(iter(data.get_generator()))
    base_model = model.with_rolled_lon(batch[0].coarse.latlon_coordinates.lon)
    fine = batch[0].fine.latlon_coordinates
    y_idx = _crop_index(fine.lat, *footprint[0])
    x_idx = _crop_index(fine.lon, *footprint[1])
    logging.info(
        f"{event.name}: mode={mode}, n_samples={event.n_samples}, "
        f"footprint {len(y_idx)}x{len(x_idx)} fine cells"
    )

    hooks: _Capture | _Recompute | None = None
    if mode == "capture":
        hooks = _Capture()
    elif mode == "recomp":
        hooks = _Recompute(None)
    elif mode == "frozen":
        if not captured:
            raise RuntimeError("frozen arm ran before any capture arm")
        hooks = _Recompute(captured)
    if hooks is not None:
        for name, module, num_groups in _group_norm_modules(base_model.modules):
            hooks.attach(name, module, num_groups)

    winds: list[np.ndarray] = []
    precips: list[np.ndarray] = []
    truth: tuple[float, float] | None = None
    try:
        for start in range(0, event.n_samples, SAMPLE_GROUP):
            group = min(SAMPLE_GROUP, event.n_samples - start)
            with torch.no_grad():
                outputs = base_model.generate_on_batch(batch, n_samples=group)
            w, p = _sample_metrics(outputs.prediction, y_idx, x_idx)
            winds.append(w)
            precips.append(p)
            if truth is None:
                truth = _truth_metrics(outputs.target, y_idx, x_idx)
    finally:
        if hooks is not None:
            hooks.remove()

    if isinstance(hooks, _Capture):
        captured.update(hooks.stats)
        logging.info(
            f"{event.name}: captured {len(hooks.stats)} layers; recomputation vs "
            f"fused kernel max rel {hooks.max_rel_error:.3e}"
        )

    assert truth is not None
    return np.concatenate(winds), np.concatenate(precips), truth


def _compare(values: dict[str, np.ndarray], control: str) -> list[str]:
    """Mean +/- sd per arm, with Welch's t-test against the control arm."""
    lines = [
        f"{'arm':<20}{'mean':>12}{'sd':>10}{'n':>5}"
        f"{'vs control':>13}{'p':>10}{'signif':>8}"
    ]
    lines.append("-" * len(lines[0]))
    ref = values[control]
    for arm, vals in values.items():
        delta = float(vals.mean() - ref.mean())
        if arm == control:
            row = f"{'--':>13}{'--':>10}{'--':>8}"
        else:
            p = float(stats.ttest_ind(vals, ref, equal_var=False).pvalue)
            row = f"{delta:>13.4g}{p:>10.3g}{('yes' if p < 0.05 else 'no'):>8}"
        lines.append(
            f"{arm:<20}{vals.mean():>12.5g}{vals.std(ddof=1):>10.4g}{len(vals):>5}{row}"
        )
    return lines


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
    footprint = _common_footprint(config.events or [])
    logging.info(
        f"coarse_shape={model.coarse_shape}; scoring footprint "
        f"lat {footprint[0]}, lon {footprint[1]}"
    )

    captured: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    winds: dict[str, np.ndarray] = {}
    precips: dict[str, np.ndarray] = {}
    truth: tuple[float, float] | None = None

    for event in config.events or []:
        try:
            w, p, t = _run_arm(event, config, model, requirements, captured, footprint)
            winds[event.name] = w
            precips[event.name] = p
            truth = truth or t
        except Exception:
            logging.exception(f"{event.name}: arm failed, continuing")

    control = next((a for a in winds if "16deg" in a), None)
    if control is None:
        raise RuntimeError("no 16deg control arm succeeded; nothing to compare against")

    lines = [
        f"cyclone extremes over the central footprint lat {footprint[0]} "
        f"lon {footprint[1]}",
        f"control = {control} (the extent the model was trained on)",
        f"ensemble of {len(winds[control])} diffusion samples per arm; "
        "Welch t-test vs control",
    ]
    if truth is not None:
        lines.append(f"fine truth: max wind {truth[0]:.5g}, mean precip {truth[1]:.5g}")
    lines += ["", "=== max 10m wind speed (m/s) ==="] + _compare(winds, control)
    lines += ["", "=== mean precipitation rate ==="] + _compare(precips, control)

    summary = "\n".join(lines)
    with open(os.path.join(config.experiment_dir, "gn-extreme-summary.txt"), "w") as f:
        f.write(summary)
    arrays = {f"wind|{a}": v for a, v in winds.items()}
    arrays.update({f"precip|{a}": v for a, v in precips.items()})
    if truth is not None:
        arrays["truth"] = np.array(truth)
    np.savez_compressed(
        os.path.join(config.experiment_dir, "gn-extreme-metrics.npz"), **arrays
    )
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().config_path)

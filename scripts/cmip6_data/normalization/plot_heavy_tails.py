"""Per-model histograms for heavy-tailed variables (T5).

For pr, hus (per plev), siconc, rsus we overlay one histogram per model,
in linear and log space side-by-side. The point is to decide:

- Are model distributions the same shape but scaled differently? Then a
  shared log/Box-Cox + shared standardization handles them.
- Or do model distributions diverge in shape (e.g., one model bimodal,
  another unimodal)? Then per-model transforms / scales become the
  natural answer for that variable.

We sample timesteps to keep the script fast — a uniform sample of ~30
days from each dataset's `historical` 2010 zarr (the only experiment
processed in the pilot subset; ssp585 is also one year and would just
duplicate the picture). Spatial dimensions are kept fully so tail
fidelity is good.

Output: ``outputs/heavy_tail_histograms/{variable}.png`` (single panel)
or ``{variable}_plev{i}.png`` for 3D. Each figure has two panes: linear
on the left, log on the right (with a small offset added before logging
so zero-floor variables like rsus and pr don't blow up).
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ProcessConfig  # noqa: E402

LOG = logging.getLogger("normalization.heavy_tails")

# (variable, plev_indices) — None means scalar (no plev). For hus we hit
# every plev level.
_TARGETS: list[tuple[str, list[int] | None]] = [
    ("pr", None),
    ("rsus", None),
    ("siconc", None),
    ("hus", list(range(8))),
]

_N_TIMESTEPS = 30  # uniform sample per dataset
_HIST_BINS = 80
# Small additive offsets to keep log10 well-defined for zero-floored
# variables. Picked one per variable (~0.1% of typical "active" value).
_LOG_OFFSET = {
    "pr": 1e-7,  # kg/m²/s; ~0.01 mm/day equivalent
    "rsus": 1.0,  # W/m²; ~floor of nighttime
    "siconc": 0.1,  # %
    "hus": 1e-9,  # kg/kg; below any physical value
}


def _open(out_dir: str, source_id: str, experiment: str, variant: str) -> xr.Dataset:
    p = Path(out_dir) / source_id / experiment / variant / "data.zarr"
    if not p.exists():
        raise FileNotFoundError(f"missing dataset: {p}")
    return xr.open_zarr(p, consolidated=True)


def _list_pilot_datasets(out_dir: Path) -> list[tuple[str, str, str]]:
    """Yield (source_id, experiment, variant) for each ok dataset.

    We pick one variant per (source_id, experiment) — the lexicographically
    smallest, which in CMIP6 is r1i1p1f1 for almost every model. Multiple
    realizations of the same model duplicate the picture for
    distribution-shape questions.
    """
    out: list[tuple[str, str, str]] = []
    for source_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        for exp_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
            variants = sorted(p for p in exp_dir.iterdir() if p.is_dir())
            if not variants:
                continue
            v = variants[0]
            if (v / "data.zarr").exists():
                out.append((source_dir.name, exp_dir.name, v.name))
    return out


def _sample_values(
    ds: xr.Dataset, var: str, plev_index: int | None, n_t: int
) -> np.ndarray:
    if var not in ds:
        return np.array([])
    da = ds[var]
    if plev_index is not None:
        if "plev" not in da.dims:
            return np.array([])
        da = da.isel(plev=plev_index)
    if "time" in da.dims and da.sizes["time"] > n_t:
        idx = np.linspace(0, da.sizes["time"] - 1, n_t, dtype=int)
        da = da.isel(time=idx)
    arr = da.values.ravel()
    return arr[np.isfinite(arr)]


def _plot_var(
    var: str,
    plev_index: int | None,
    samples_by_label: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    if not samples_by_label:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_lin, ax_log = axes

    all_vals = np.concatenate(list(samples_by_label.values()))
    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    bins_lin = np.linspace(lo, hi, _HIST_BINS)

    offset = _LOG_OFFSET.get(var, 0.0)
    log_all = np.log10(np.maximum(all_vals, 0) + offset)
    log_lo, log_hi = float(np.nanmin(log_all)), float(np.nanmax(log_all))
    bins_log = np.linspace(log_lo, log_hi, _HIST_BINS)

    n_models = len(samples_by_label)
    cmap = plt.get_cmap("turbo")
    for i, (label, vals) in enumerate(sorted(samples_by_label.items())):
        color = cmap(i / max(n_models - 1, 1))
        ax_lin.hist(
            vals,
            bins=bins_lin,
            density=True,
            histtype="step",
            color=color,
            alpha=0.85,
            linewidth=1.0,
            label=label,
        )
        log_vals = np.log10(np.maximum(vals, 0) + offset)
        ax_log.hist(
            log_vals,
            bins=bins_log,
            density=True,
            histtype="step",
            color=color,
            alpha=0.85,
            linewidth=1.0,
        )

    ax_lin.set_yscale("log")
    ax_lin.set_xlabel(var)
    ax_lin.set_ylabel("density (log)")
    title = var if plev_index is None else f"{var} (plev_index={plev_index})"
    ax_lin.set_title(f"{title} — linear")

    ax_log.set_xlabel(f"log10({var} + {offset:g})")
    ax_log.set_ylabel("density")
    ax_log.set_title(f"{title} — log10")

    if n_models <= 12:
        ax_lin.legend(fontsize=7, loc="upper right", ncol=1)
    else:
        ax_lin.legend(fontsize=6, loc="upper right", ncol=2)

    fig.suptitle(
        f"Per-model distribution of {title} — "
        f"{n_models} datasets, "
        f"{len(all_vals):,} cell-time samples"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--out-dir",
        default="normalization/outputs/heavy_tail_histograms",
    )
    parser.add_argument(
        "--experiment",
        default="historical",
        help="Use one experiment for the overlay (default: historical)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ProcessConfig.from_file(args.config)
    out_dir = Path(args.out_dir)
    pilot_root = Path(cfg.output_directory)

    triples = [t for t in _list_pilot_datasets(pilot_root) if t[1] == args.experiment]
    LOG.info("Found %d %s datasets", len(triples), args.experiment)

    # Pre-pick which variables exist; saves repeat I/O.
    sample_by_var: dict[tuple[str, int | None], dict[str, np.ndarray]] = {}
    for var, plevs in _TARGETS:
        if plevs is None:
            sample_by_var[(var, None)] = {}
        else:
            for p in plevs:
                sample_by_var[(var, p)] = {}

    for source_id, exp, variant in triples:
        try:
            ds = _open(str(pilot_root), source_id, exp, variant)
        except FileNotFoundError:
            continue
        label = source_id  # one r per model, no need for full variant tag
        for (var, plev), bucket in sample_by_var.items():
            arr = _sample_values(ds, var, plev, _N_TIMESTEPS)
            if arr.size == 0:
                continue
            bucket[label] = arr
        ds.close()
        LOG.info("Sampled %s/%s/%s", source_id, exp, variant)

    for (var, plev), bucket in sample_by_var.items():
        if not bucket:
            LOG.warning("No data for %s plev=%s", var, plev)
            continue
        suffix = "" if plev is None else f"_plev{plev}"
        _plot_var(var, plev, bucket, out_dir / f"{var}{suffix}.png")


if __name__ == "__main__":
    main()

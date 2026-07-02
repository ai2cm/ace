"""Generate diagnostic figures for the README's known data quirks.

Outputs into ``scripts/cmip6_data/figures/``:

- ``inm_cm4_8_zg_top.png`` — INM-CM4-8 ``zg`` at 10 hPa shows
  anomalously low values in a subset of cells (~22 km vs the
  expected ~32 km). Side-by-side with CanESM5 as a reference and a
  histogram makes the anomaly obvious.
- ``cesm2_fv2_sftlf_overshoot.png`` — CESM2-FV2's regridded land
  fraction reaches ~114% in some cells. Overlaid histogram + map
  with cells > 100% highlighted in red.

Usage:
    python make_figures.py --config configs/pilot.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from config import ProcessConfig  # noqa: E402


def _open(out_dir: str, source_id: str, experiment: str, variant: str) -> xr.Dataset:
    p = Path(out_dir) / source_id / experiment / variant / "data.zarr"
    if not p.exists():
        raise FileNotFoundError(f"missing dataset: {p}")
    return xr.open_zarr(p, consolidated=True)


def figure_inm_zg_top(out_dir: str, fig_path: Path) -> None:
    inm = _open(out_dir, "INM-CM4-8", "historical", "r1i1p1f1")
    can = _open(out_dir, "CanESM5", "historical", "r1i1p1f1")

    # Time-min surfaces the bad-cell story: time-averaging smooths the
    # episodic low values into a 500 m cool patch, but ``min over time``
    # captures the worst case per cell, which is what makes the
    # hypsometric layer-6 derivation collapse.
    inm_min = inm["zg"].isel(plev=-1).min("time")
    can_min = can["zg"].isel(plev=-1).min("time")

    inm_v = inm_min.values
    can_v = can_min.values
    vmin = float(min(inm_v.min(), can_v.min()))
    vmax = float(max(inm_v.max(), can_v.max()))

    lat = inm.lat.values
    lon = inm.lon.values
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0], hspace=0.55, wspace=0.25)

    ax_inm = fig.add_subplot(gs[0, 0])
    ax_can = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, :])

    for ax, da, title in (
        (ax_inm, inm_v, "INM-CM4-8 r1 (excluded from training)"),
        (ax_can, can_v, "CanESM5 r1 (reference)"),
    ):
        im = ax.imshow(
            da,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(f"{title}\n2010 minimum zg at 10 hPa (m)")
        ax.set_xlabel("longitude (deg)")
        ax.set_ylabel("latitude (deg)")
        plt.colorbar(im, ax=ax, label="m")

    # Histogram across every cell-time value with bins wide enough to
    # cover the low-end tail (down to ~22 km in INM-CM4-8 cells).
    inm_all = inm["zg"].isel(plev=-1).values.ravel()
    can_all = can["zg"].isel(plev=-1).values.ravel()
    lo = float(min(inm_all.min(), can_all.min()))
    hi = float(max(inm_all.max(), can_all.max()))
    bins = np.linspace(lo - 200, hi + 200, 100)
    ax_hist.hist(inm_all, bins=bins, alpha=0.55, label="INM-CM4-8 r1", color="#cc3333")
    ax_hist.hist(can_all, bins=bins, alpha=0.55, label="CanESM5 r1", color="#3aa845")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("zg at 10 hPa (m)")
    ax_hist.set_ylabel("cell-time count (log)")
    ax_hist.set_title(
        "Per-cell distribution of zg at 10 hPa across the 2010 year — "
        f"INM-CM4-8 has values down to ~{int(inm_all.min()):,} m "
        "vs CanESM5's clean cluster around 31 km"
    )
    ax_hist.legend()

    fig.suptitle(
        "Known model-side quirk: INM-CM4-8 reports unphysically low zg at 10 hPa "
        "in some grid cells"
    )
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", fig_path)


def figure_cesm2fv2_sftlf(out_dir: str, fig_path: Path) -> None:
    fv2 = _open(out_dir, "CESM2-FV2", "historical", "r1i1p1f1")
    can = _open(out_dir, "CanESM5", "historical", "r1i1p1f1")

    fv2_v = fv2["sftlf"].values
    can_v = can["sftlf"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"wspace": 0.25})
    ax_hist, ax_map = axes

    bins = np.linspace(-2.0, 120.0, 60)
    ax_hist.hist(
        fv2_v.ravel(), bins=bins, alpha=0.55, label="CESM2-FV2", color="#cc3333"
    )
    ax_hist.hist(
        can_v.ravel(),
        bins=bins,
        alpha=0.55,
        label="CanESM5 (reference)",
        color="#3aa845",
    )
    ax_hist.axvline(
        100, color="k", linestyle=":", alpha=0.7, label="100% (max physical)"
    )
    ax_hist.set_xlabel("sftlf (%)")
    ax_hist.set_ylabel("cell count")
    ax_hist.set_title("sftlf cell-value distribution")
    ax_hist.legend(loc="upper center")

    lat = fv2.lat.values
    lon = fv2.lon.values
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    base = ax_map.imshow(
        np.clip(fv2_v, 0, 100),
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0,
        vmax=100,
        aspect="auto",
    )
    over = np.where(fv2_v > 100, fv2_v, np.nan)
    over_im = ax_map.imshow(
        over,
        origin="lower",
        extent=extent,
        cmap="Reds",
        vmin=100,
        vmax=115,
        aspect="auto",
    )
    ax_map.set_title("CESM2-FV2 sftlf — cells > 100% highlighted in red")
    ax_map.set_xlabel("longitude (deg)")
    ax_map.set_ylabel("latitude (deg)")
    plt.colorbar(base, ax=ax_map, label="% land (clipped to 100)", pad=0.02)
    plt.colorbar(over_im, ax=ax_map, label="% land (overshoot region)", pad=0.08)

    fig.suptitle(
        "Known model-side quirk: CESM2-FV2 sftlf overshoots 100% along the "
        "southernmost (~-87 to -90 deg) row of the F22.5 target grid"
    )
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", fig_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to process YAML")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ProcessConfig.from_file(args.config)
    out_dir = cfg.output_directory
    fig_dir = Path(__file__).parent / "figures"

    figure_inm_zg_top(out_dir, fig_dir / "inm_cm4_8_zg_top.png")
    figure_cesm2fv2_sftlf(out_dir, fig_dir / "cesm2_fv2_sftlf_overshoot.png")


if __name__ == "__main__":
    main()

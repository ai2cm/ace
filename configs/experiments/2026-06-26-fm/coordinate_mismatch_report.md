# Vertical Coordinate Mismatch Report — `2026-06-26-fm`

**Date:** 2026-07-09
**Scope:** FM (foundation-model) training / eval / cooldown configs in this directory.

## Summary

The FM training configs (`ace-train-config-4deg-AIMIP-nc-sfno-fm-*.yaml`) pool ERA5
and SHiELD (c96) datasets that do **not** share the same hybrid sigma-pressure
vertical coordinate (`ak`/`bk`). The horizontal grid matches exactly; only the
vertical coordinate differs. Two config flags cause this mismatch to be swallowed
silently rather than raised as an error. The magnitude of the mismatch turns out to
be **small** (layer-center reference pressures within ~10 hPa, ISA reference
temperature within ~1.4 K), so this is a physics-consistency wart worth harmonizing,
not a result-wrecking bug.

## Datasets involved

| Role | Path (`/climate-default/...`) | Vertical coord |
|---|---|---|
| Inference / validation / cooldown | `2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr` | **ERA5** |
| FM train, concat member #0 | `2026-01-28-...c96-4deg-daily-shield-amip-ensemble-dataset/ic_000*.zarr` | SHiELD |
| FM train | `2026-06-08-...c96-shield-ramped-climSST-random-CO2-...-4deg-daily/*.zarr` | SHiELD |
| FM train | `2026-06-08-...4deg-daily-c96-shield-som-ensemble-fme-dataset/*.zarr` | SHiELD |
| FM train, concat member (last) | `2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr` | ERA5 |

Bucket root: `gs://vcm-ml-intermediate/`. All three SHiELD datasets share one
bit-identical `ak`/`bk`; only ERA5 differs. Both grids are 45×90 (4°) with matching
`lat`/`lon` (ERA5 uses `latitude`/`longitude`, SHiELD uses `grid_yt`/`grid_xt`, values
`allclose`). `ak`/`bk` are stored as scalar variables `ak_0..ak_8`, `bk_0..bk_8`
(9 interfaces = 8 layers).

## Coordinate comparison

### Raw `ak` (Pa) / `bk`

| i | ERA5 `ak` | SHiELD `ak` | ERA5 `bk` | SHiELD `bk` |
|--:|--:|--:|--:|--:|
| 0 | 1.0 | 300.0 | 0.0000 | 0.0000 |
| 1 | 5119.9 | 5247.9 | 0.0000 | 0.0000 |
| 2 | 13881.3 | 12990.4 | 0.0054 | 0.0090 |
| 3 | 19343.5 | 14738.1 | 0.0597 | 0.1138 |
| 4 | 20087.1 | 12854.1 | 0.2035 | 0.2864 |
| 5 | 15596.7 | 9156.1 | 0.4384 | 0.5101 |
| 6 | 8880.5 | 5484.4 | 0.6806 | 0.7103 |
| 7 | 3057.3 | 2261.7 | 0.8739 | 0.8797 |
| 8 | 0.0 | 0.0 | 1.0000 | 1.0000 |

`max|Δak| = 7233 Pa`, `max|Δbk| = 0.083`. Looks large, but `ak` and `bk` trade off —
see effective pressure below.

### Effective reference pressure at `ps = 1013.25 hPa` (`p = ak + bk·ps`)

Interface pressures (hPa):

| i | ERA5 | SHiELD | Δ |
|--:|--:|--:|--:|
| 0 | 0.01 | 3.00 | −2.99 |
| 1 | 51.20 | 52.48 | −1.28 |
| 2 | 144.28 | 139.02 | +5.26 |
| 3 | 253.93 | 262.69 | −8.76 |
| 4 | 407.07 | 418.74 | −11.67 |
| 5 | 600.18 | 608.42 | −8.24 |
| 6 | 778.42 | 774.56 | +3.87 |
| 7 | 916.05 | 913.97 | +2.08 |
| 8 | 1013.25 | 1013.25 | 0.00 |

Layer-center pressures (hPa):

| k | ERA5 | SHiELD | Δ |
|--:|--:|--:|--:|
| 0 | 25.60 | 27.74 | −2.14 |
| 1 | 97.74 | 95.75 | +1.99 |
| 2 | 199.11 | 200.86 | −1.75 |
| 3 | 330.50 | 340.71 | −10.22 |
| 4 | 503.62 | 513.58 | −9.96 |
| 5 | 689.30 | 691.49 | −2.19 |
| 6 | 847.24 | 844.26 | +2.97 |
| 7 | 964.65 | 963.61 | +1.04 |

Layer centers agree within ~10 hPa; worst at mid-troposphere (layers 3–4). Same 8
layers, monotonic, shared top and surface interfaces.

### ISA reference temperature per layer center (US Standard Atmosphere 1976)

| k | p ERA5 (hPa) | p SHiELD (hPa) | T ERA5 (K) | T SHiELD (K) | ΔT (E−S) |
|--:|--:|--:|--:|--:|--:|
| 0 | 25.60 | 27.74 | 221.52 | 221.00 | +0.52 |
| 1 | 97.74 | 95.75 | 216.65 | 216.65 | 0.00 |
| 2 | 199.11 | 200.86 | 216.65 | 216.65 | 0.00 |
| 3 | 330.50 | 340.71 | 232.83 | 234.19 | −1.35 |
| 4 | 503.62 | 513.58 | 252.26 | 253.20 | −0.94 |
| 5 | 689.30 | 691.49 | 267.78 | 267.95 | −0.16 |
| 6 | 847.24 | 844.26 | 278.50 | 278.32 | +0.19 |
| 7 | 964.65 | 963.61 | 285.47 | 285.41 | +0.06 |

`max|ΔT| = 1.35 K`, `rms ΔT = 0.62 K`. Layers 1–2 fall in the isothermal tropopause
band (216.65 K both) → exactly 0. Everywhere else sub-1 K except layer 3.

## How the mismatch is masked (code path)

1. **Concat base = first member.** `fme/core/dataset/concat.py:30` sets
   `self._properties = datasets[0].properties.copy()`; later members merged via
   `update()`. In the FM configs, concat member #0 is a SHiELD dataset and ERA5 is
   last, so the recorded training vertical coordinate is **SHiELD's**.
2. **`strict: false` downgrades error to warning.**
   `fme/core/dataset/properties.py:55-74`: on `vertical_coordinate != other` it
   raises only when `strict`; otherwise emits `warnings.warn`. The FM train loaders
   set `strict: false`, so the ERA5 member's differing coordinate is swallowed.
3. **Checkpoint stores the training coordinate.** `fme/ace/train/train.py:271`
   `dataset_info = train_data.dataset_info` → FM base checkpoint records SHiELD `ak`/`bk`.
4. **Eval bypasses the compatibility guard.** `fme/core/dataset_info.py:117-120`
   (`assert_compatible_with`) would raise on the coordinate diff, but
   `fme/ace/inference/evaluator.py:378` and `inference.py:348` skip it when
   `allow_incompatible_dataset: true`, which the eval suite sets on every entry.

## Effect per phase

- **FM base training:** recorded coordinate = SHiELD. With `group_weights`
  `start_value: [0.5, 0.5]`, ~50% of samples are ERA5 fields, corrected with SHiELD
  `ak`/`bk`. Corrector (`conserve_dry_air`, `moisture_budget_correction`) computes
  layer thickness `dp = Δak + ps·Δbk` with the wrong grid for those samples.
- **Eval:** FM checkpoint (SHiELD coordinate) evaluated on ERA5 data, guard suppressed.
- **Cooldown** (`*-cooldown`, `*-bestinfcooldown`): trains on ERA5-only (all SHiELD
  dropped). `parameter_init.weights_path` loads **weights only**, so
  `dataset_info` is re-derived from ERA5 → cooldown coordinate = **ERA5**, matching
  the eval target. The coordinate therefore flips SHiELD → ERA5 between base training
  and cooldown while inheriting SHiELD-trained weights, and the model re-fits the
  ERA5 budget over the (short) cooldown schedule.

## Severity and recommendation

Small. Layer identity is preserved, effective pressures differ ≤~10 hPa, and the ISA
reference temperature a given `air_temperature_k` implies differs by <1.4 K between
the grids — well under natural variance. Corrector `dp` is off by a few percent for
the affected samples.

Recommended:

1. Harmonize the vertical coordinate before pooling — regrid the ERA5-8layer dataset
   onto SHiELD's `ak`/`bk` (or vice versa) so every training/eval/cooldown dataset
   shares one coordinate.
2. Until harmonized, keep `strict: false` / `allow_incompatible_dataset: true` only
   with awareness that they are masking this specific mismatch; do not rely on them to
   catch a future, larger coordinate error.
3. If harmonizing is not feasible, consider ordering ERA5 first in the concat so the
   recorded coordinate matches the eval/cooldown target, removing the base→cooldown flip.

## Reproduction

```python
import xarray as xr, numpy as np
paths = {
 "ERA5":         "gs://vcm-ml-intermediate/2026-04-17-era5-4deg-8layer-daily-1940-2025/2026-03-19-era5-4deg-8layer-1940-2025.zarr",
 "SHiELD-amip":  "gs://vcm-ml-intermediate/2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-dataset/ic_0001.zarr",
 "SHiELD-ramped":"gs://vcm-ml-intermediate/2026-06-08-vertically-resolved-c96-shield-ramped-climSST-random-CO2-ensemble-fme-dataset-4deg-daily/ramped-sst-1xCO2-random-perturbation.zarr",
 "SHiELD-som":   "gs://vcm-ml-intermediate/2026-06-08-vertically-resolved-4deg-daily-c96-shield-som-ensemble-fme-dataset/1xCO2-ic_0001.zarr",
}
for name, p in paths.items():
    ds = xr.open_zarr(p, decode_times=False, chunks=None)
    n = len([k for k in ds.variables if k.startswith("ak_")])
    ak = np.array([float(ds[f"ak_{i}"].values) for i in range(n)])
    bk = np.array([float(ds[f"bk_{i}"].values) for i in range(n)])
    print(name, ak, bk)
```

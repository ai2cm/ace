# ACE2S snow-transform training, 1-deg daily

Four models testing better encodings of the snow variables, against the naive-encoding baselines
in `../2026-08-07-ace2s-snow-prognostic-daily` (which standardize `surface_snow_amount` (SWE) and
`surface_snow_area_fraction` (SCF) with raw global mean/std). Motivated by the data-level analysis
in `explore2/brianh/2026-06-09-land-atm-coupling/snow-var-distributions/report.md`: ice sheets set
the raw σ (dynamic-range ratio ~125× ERA5 / ~15× CM4), the seasonal-snow tail is heavily skewed,
and SCF is bounded with point masses at 0 and 1.

## Arms

Each config is its dataset's baseline config plus exactly: a `transforms:` block under both
`normalization.network` and `normalization.residual`, and a stats mount whose snow entries are
transformed-space statistics (all other variables byte-identical). Recipes are alternatives, never
composed.

| recipe | SWE encoding | SCF encoding | notes |
|---|---|---|---|
| **log1p** | `z = (log1p(x) − μ)/σ` | `z = (logit(clip(p, ε, 1−ε)) − μ)/σ`, `scale` 1 (ERA5) / 100 (CM4, percent) | SCF dropped from `force_positive_names` — the sigmoid decode already bounds it |
| **quantile** | `z = Φ⁻¹(F̂(x))` via fitted knot table | same | tables fit on land cells; μ=0, σ=1 by construction |

The transforms are applied inside the normalizer (`fme/core/field_transform.py`,
`TransformedNormalizer`), so model inputs/outputs, all inline metrics, and W&B keys stay in
physical units and compare directly against the baselines (same W&B group).

## Stats

Fitted by `fit_snow_transform_stats.py` (see its docstring) from consecutive-day pairs of the daily
zarrs: ERA5 1990–2019 at 2-day stride, CM4 full record at 8-day stride. Patched stats datasets
(only the two snow entries differ from the baseline stats):

| mount | contents |
|---|---|
| `brianhenn/2026-08-07-era5-1deg-8layer-daily-snow-log1p-stats-1990-2019` | patched stats |
| `brianhenn/2026-08-07-era5-1deg-8layer-daily-snow-quantile-stats-1990-2019` | patched stats + `quantile-<var>.nc` knot tables |
| `brianhenn/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-snow-log1p-stats` | patched stats |
| `brianhenn/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-snow-quantile-stats` | patched stats + knot tables |

## Launch

```bash
./run-ace-train.sh                    # ai2/jupiter, 8 GPUs; all four
CLUSTER=ai2/titan ./run-ace-train.sh  # ai2/titan, 4 GPUs
```

## Evaluation

Overlay against the baselines on identical W&B keys: `skill_map` R²/RMSE for the snow channels
(does the ice-sheet margin ring improve?), per-channel validation loss, snow climatology and
seasonal cycle in inline inference, whether predicted SCF stays in range, and the predicted
snow-free fraction against the observed ~65% (ERA5) / ~72% (CM4).

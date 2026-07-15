# ACE2S snow-prognostic training

Training runs that give ACE2S **prognostic snow**: snow water equivalent becomes a predicted,
fed-back state variable, and snow-covered area a predicted diagnostic. This is the counterpart to
the land-forcing follow-up (`exp/ace2s-land-forcing-training`), which instead prescribed snow as a
*forcing* input. Here we ask whether ACE2S can carry and evolve its own snow state, naively
replicating the standard two-stage ACE2S recipe otherwise.

## Design

- **Two datasets**, one model each, trained from scratch: `era5-snow-prognostic`,
  `cm4-snow-prognostic`.
- **Added variables** (identical names in both datasets):
  | variable | name | role | in `in_names` | in `out_names` |
  |---|---|---|---|---|
  | snow water equivalent | `surface_snow_amount` | **prognostic** | yes | yes |
  | snow-covered area | `surface_snow_area_fraction` | **diagnostic** | no | yes |

  A variable is prognostic iff it is in both `in_names` and `out_names`, and diagnostic iff it is
  in `out_names` only (`fme/core/step/single_module.py`). Both are added to
  `corrector.force_positive_names` (non-negative). Adding an input channel changes the encoder
  input dims, so these models train **from scratch** (no warm-start from the deployed controls).
- **Two-stage recipe** per model: 1-step pretrain → multi-step finetune (warm-started from the
  pretrain checkpoint). **Both stages are run** (stage 2 is launched after stage 1 finishes; see
  Launch order). The finetune configs are variable-agnostic — they inherit
  `in_names`/`out_names`/corrector from the mounted pretrain checkpoint.
- **Frozen-precipitation corrector enabled**: the moisture-budget corrector's
  `clip_frozen_precipitation: true` clips predicted `total_frozen_precipitation_rate` to the
  (corrected) `PRATEsfc`, since frozen precip is a component of total precip. (Merged from
  `feature/frozen-precip-corrector`.)
- **One-step skill-map validation metric**: the default-on `skill_map` aggregator (per-gridcell R²
  and RMSE at the first forecast step) logs automatically during one-step validation, with the R²
  panel on a fixed [-1, 1] colorbar so near-constant snow variables don't wash it out. (Merged from
  `feature/one-step-r2-metric`.)
- **Short-lead, many-IC inline inference**: each config runs the standard long climate rollout
  **plus** a second `inference-short-lead` probe (40 held-out ICs as explicit timestamps — ERA5
  quarterly 1996-2005, CM4 across validation years 0306-0310 — each a ~10-day rollout, 10-member
  ensemble). It logs a discrete short-range skill ladder: `step_means` RMSE/bias (denorm per-variable
  + norm `channel_mean`) and `ensembles` CRPS / spread-skill / ensemble-mean RMSE at leads
  1/2/4/8/20/40 (6h→10d). `weight: 0` (diagnostic — does not drive checkpoint selection);
  `forward_steps_in_memory: 8` bounds memory under the 10× ensemble broadcast; climate metrics
  (`power_spectrum`, `zonal_mean`) are disabled as meaningless over 10 days. Inline inference cannot
  log the continuous lead-time curve (`enable_time_series=False`); the discrete ladder is the
  workaround. The long rollout is explicitly named `inference` so its W&B keys match the control/other
  runs — with >1 inference entry, an unnamed first entry defaults to `inference_0` (not `inference`)
  and breaks cross-run comparison.
- **Reading the 10-day skill gain vs control**: the gain is read by overlaying a control run against
  the treatment on the shared `inference-short-lead` keys (`mean_step_{k}/...`, `ensemble_step_{k}/...`).
  This requires the control to be retrained with a **byte-identical** `inference-short-lead` block
  (same name, IC timestamps, `step_means`/`ensembles`, `n_ensemble_per_ic`) so the keys and ICs
  match — otherwise the W&B overlay silently breaks. **TODO when control-rerun configs are created:
  copy this entry verbatim into them.** (A continuous lead-time curve, if ever wanted, needs the
  offline evaluator with `enable_time_series=True`.)

## Base recipes (ported)

- ERA5 pretrain/finetune ← `feature/add-ACE2S-ERA5-baseline` (via
  `exp/ace2s-land-forcing-training`).
- CM4 pretrain/finetune ← `exp/ace2s-cm4-piControl-train` (via
  `exp/ace2s-land-forcing-training`).

## Inputs

| | value |
|---|---|
| ERA5 train/inference data | `/climate-default/2026-03-19-era5-1deg-8layer-1940-2025.zarr` (Weka) |
| CM4 train/inference data | `/climate-default/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr` (Weka) |
| ERA5 stats (network norm) | beaker `andrep/2026-03-19-era5-1deg-8layer-stats-1990-2019` → `/statsdata` |
| CM4 stats (network norm) | beaker `jamesd/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-stats` → `/statsdata` |

Stats datasets are mounted via each config's `# arg:` header; the launcher extracts them.

## Prerequisites to verify BEFORE launching

1. **Normalization stats contain the new variables — the main risk.** `surface_snow_amount` is
   **prognostic**, so it is residual-scaled and must appear in **all three** stats files:
   `centering.nc`, `scaling-full-field.nc`, **and** `scaling-residual.nc`. `surface_snow_area_fraction`
   is diagnostic and needs only `centering.nc` + `scaling-full-field.nc`. Missing entries raise a
   hard error (`fme/core/normalizer.py`, `Variable ... not found`). Check both stats datasets. The
   land-forcing branch verified only `surface_snow_area_fraction` in the network files, so
   `surface_snow_amount` and the residual file are unverified here. If `scaling-residual.nc` lacks
   `surface_snow_amount`, regenerate the stats to include it.
2. **No-NaN check** on `surface_snow_amount` in both zarrs (both snow fields are expected NaN-free —
   0 over ocean — matching the land-forcing branch's finding for `surface_snow_area_fraction`). If a
   snow field has NaN, output-side masking would be needed (out of scope for this naive
   replication).
3. **Confirm early-epoch loss is finite (non-NaN)** on a short run before scaling up.

## Launch order

```bash
./run-ace-train.sh                 # submit both pretrain jobs
# when the pretrain jobs finish, paste their beaker dataset IDs into run-ace-train.sh
# (PRETRAIN_DATASET_ERA5 / PRETRAIN_DATASET_CM4), uncomment the finetune block, then:
./run-ace-train.sh                 # submit both finetune jobs
# (add a substring, e.g. `./run-ace-train.sh era5`, to launch a single model)
```

Each job runs `torchrun --nproc_per_node 4 -m fme.ace.train <config>` on `ai2/titan`, logging to
W&B (project `ace`, group `ace2s-snow-prognostic`).

## Evaluation

After a model finetunes, compare against the deployed ACE2S control checkpoints. Focus on snow
skill (the new `surface_snow_amount`/`surface_snow_area_fraction` metrics and the one-step
skill_map panels), the `inference-short-lead` section for short-range skill, and near-surface T/Q
to check for regressions from the added prognostic state.

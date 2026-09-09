# Per-dataset normalization ablation

## Goal

Test whether per-dataset normalization statistics improve ERA5 performance when
training jointly on ERA5 and C96 SHiELD synthetic data.

Hypothesis: **feature-space alignment.** Pooled stats leave
`specific_total_water_0` with disjoint marginals between the two sources (JS =
0.6930, overlap 0.000, 5.3σ mean offset), and per-group scalar standardization
is exactly the operation that closes a mean-and-scale offset like that.

This is **not** a test of whether the model can tell the sources apart.
`global_mean_co2` stays pooled (see Pinned variables) and remains a perfect
discriminator in every arm. A result here should not be read as evidence about
source-partitioning behavior.

## Axes

**Normalization arm** — how many groups the network's normalization constants
are split into:

| ID | Strategy | Groups |
|----|----------|--------|
| **A1** | Shared (control) | 1 — all data |
| **A2** | Per-source | `c96`, `era5` |
| **A3** | Per-config | `amip`, `ramped`, `som`, `era5` |

CO2 levels stay pooled *within* group in every arm, so forced response is
preserved.

**Module conditioning** — off / on. A second, independent use of the same
labels: with `conditional: true` the module also consumes them through its
adaLN (swin) or CLN (sfno) layers, giving the model an explicit source signal
rather than only an aligned input space. Crossed with the arms so the two
mechanisms can be attributed separately.

**Data regime** — c96-only, era5-only, fm (both).

**Input masking** — off / `mask10`. A third mechanism, independent of the
labels: with `input_dropout.default.max_masked_vars: 10` the step drops
`k ~ U[0, 10]` of its 45 packed input channels (44 `in_names` plus the shared
GMR sentinel) on every training step, and `include_channel_mask_inputs: true`
appends a per-channel presence indicator so the network can tell a dropped
channel from one sitting at its climatological mean.

Masking is a candidate answer to the same question the arms ask, by a different
route: a model that cannot count on any particular channel being present has
less opportunity to key on the ones that identify the source. It is crossed
with the arms so the two can be compared and, later, combined.

Two properties of the mechanism matter when reading results. A dropped channel
is set to **zero in normalized space**, which under A1 is the pooled
climatological mean and under A2/A3 is the *group's* mean — so masking and the
grouping arms interact by construction, not only statistically. And masking is
**training-only**: `SingleModuleStep._draw_input_dropout_mask` returns `None` in
eval mode, so validation and every inference entry see the full input.

## Cell grid

Each config is composed from two base configs: the **regime source** supplies
datasets, validation and inference entries; the **architecture source**
supplies the module builder, `residual_prediction`, and `in_names` ordering.

| | c96-only | era5-only | fm |
|---|---|---|---|
| **nc-sfno** | `sfno-c96-v3` | `sfno-v2` | fm data + sfno builder |
| **nc-swin-v2** | c96 data + swin builder | era5 data + swin builder | `swin-v2-fm-random-v1` |

Regime sources: c96 → `sfno-c96-v3`, era5 → `sfno-v2`, fm →
`swin-v2-fm-random-v1`. Architecture sources: `sfno-v2` and
`swin-v2-fm-random-v1`. `sfno-fm-random-v3` is not used.

The three bases are already near-identical outside the builder: `out_names`,
`next_step_forcing_names`, `corrector`, `ocean`, `optimization`, `scheduler`,
`stepper_training`, `max_epochs`, and `global_mean_removal` are byte-identical,
and `in_names` is the same 44-name set in all three (swin orders
`global_mean_co2` last). `train_aggregator` and `ema_checkpoint_save_epochs`,
which the swin base omits, are applied uniformly to all cells.

### 22 unmasked configs, 44 in total

Cells that would train a model identical to a cheaper one are skipped. Two
independent collapses, both from a regime having too few labels for an axis to
vary anything:

- an arm with a single group has nothing to select per sample → it *is* A1;
- conditioning on a single label feeds every sample the same constant one-hot,
  and that constant scale and shift is absorbed by the normalization layers'
  own affine parameters.

| regime | kept cells | count |
|---|---|---|
| c96 | A1, A1-cond, A3, A3-cond (A2 ≡ A1) | **4** |
| era5 | A1 (A2 ≡ A3 ≡ A1; conditioning a no-op) | **1** |
| fm | A1, A2, A3, each off/on | **6** |

11 per architecture, **22 total** unmasked. The masking axis multiplies that
by the number of `MASKINGS` entries plus one for the unmasked cells: nothing
collapses under it (masking changes the training distribution in every cell),
so it adds 22 `mask10` twins for **44 configs written**, and the generator logs
28 skipped rather than 14.

Written is not submitted. `submit_norm_ablation_jobs.py --masking` takes one
variant per invocation and defaults to the unmasked cells, so the 22 masked
configs sit on disk until they are asked for. As of the first masked
submission, four are queued:

| arch | regime | arm | cond | masking |
|---|---|---|---|---|
| nc-sfno | fm | A1 | off | mask10 |
| nc-sfno | fm | A1 | on | mask10 |
| nc-swin-v2 | fm | A1 | off | mask10 |
| nc-swin-v2 | fm | A1 | on | mask10 |

A1 first because it is the cell where masking has to stand on its own: if it
lands between A1 and A3, the `mask10` twins of A2 and A3 answer whether the two
mechanisms compose or substitute, and they are already generated. `fm` because
it is the only regime holding the two sources whose alignment is at issue. `--include-degenerate` writes them anyway, and the same flag on
`submit_norm_ablation_jobs.py` submits them (only useful as a seed-variance
estimate, and only if the seeds are then changed). A degenerate arm has one
group covering the regime's whole label set, so that group reads the regime's
root pooled stats rather than a `groups/{name}/` directory the stats run never
writes — the two are pooled over the same stores.

## Pinned variables

Always use pooled stats regardless of sample group.

| Variable | JS div | overlap | Reason |
|----------|--------|---------|--------|
| `global_mean_co2` | 0.6931 | 0.000 | **Numerical.** Near-constant within group → σ→0 → normalized input explodes. Pooled σ spans 1x-4x via the between-group term (`get_pooled_stats.py`). Splitting also puts ERA5 and C96 in disjoint CO2 input spaces — kills transfer. |
| `HGTsfc` | 0.0539 | 0.833 | **Static.** Same field every sample; spatial fingerprint survives any scalar normalization. (950 m max diff between sources is a real regridding difference, but not fixable by scalar stats.) |
| `land_fraction` | 0.0077 | 0.933 | Static, same reasoning. |
| `ocean_fraction` | 0.0219 | 0.860 | Static, same reasoning. |
| `sea_ice_fraction` | 0.0143 | 0.937 | Not static, but rms_score 0.24 — the gap is well inside ERA5's own temporal variance. |
| `DSWRFtoa` | 0.0000 | 0.995 | Pure orbital geometry, identical by construction. |

`specific_total_water_0` is **not** pinned — it is the variable the experiment
exists to move.

Also pooled, as a consequence of where the seam was placed (below): global mean
removal, the loss/residual normalizer, spatial masking fill values, and every
aggregator's `*_norm` metric.

Global mean removal has one consequence worth stating outright. It runs before
normalization and shifts its fields to the **pooled** climatological mean; the
group normalizer then subtracts the **group** mean. The normalized input for
those fields therefore carries a constant `(pooled_mean - group_mean) /
group_std` that differs by group — per-group normalization does not fully align
them. This is accepted rather than fixed: GMR covers only `surface_temperature`,
`TMP2m`, `TMP850` and `air_temperature_0..7`, whose between-source gap is small
(`air_temperature_3`: rms_score 0.258), and `specific_total_water_0` — the
variable the experiment exists to move — is not a GMR field. Making GMR
per-group instead would put its offsets on a different scale in each arm and
break the fixed yardstick the seam was placed to preserve.

## Discriminability data

Source: `~/Git/explore2/alexeyy/reports/dataset_label_discriminator_summary.ipynb`

Only two variables are real discriminators (JS max = ln2 = 0.693 = fully disjoint):

| Variable | JS div | overlap | rms_score |
|----------|--------|---------|-----------|
| `global_mean_co2` | 0.6931 | 0.0000 | 11.79 |
| `specific_total_water_0` | 0.6930 | 0.0000 | 5.31 |
| *air_temperature_3* (next) | *0.0954* | *0.733* | *0.258* |

7× cliff after #2. Every variable except those two has `rms_score < 1.0` — the
ERA5-vs-SHiELD gap is smaller than ERA5's own temporal variance.

## Implementation

### Where the seam is

`GroupedNormalizer` (`fme/core/normalizer.py`) holds the pooled normalizer plus
one `StandardNormalizer` per group. `bind(labels)` resolves each sample's group
and returns a plain `StandardNormalizer` whose non-pinned constants have shape
`[n_samples, *(1,) * n_spatial_dims]`; those broadcast against the step's
`[n_batch, *spatial]` tensors, so `step_with_adjustments` needed no signature
change and the other three step types are untouched. `n_spatial_dims` comes
from `DatasetInfo.n_spatial_dims` (2 for lat/lon, 3 for HEALPix, which carries
a leading face dimension) rather than being hard-coded to 2.

Resolving the group index costs a device sync, and `bind` is called once per
forward step while the labels are fixed for the whole window, so `bind` keeps a
single-entry cache keyed on the `BatchLabels` instance: one resolve per batch,
not per step.

`NetworkAndLossNormalizationConfig` is shared with three step types that do not
bind a grouped normalizer (`SeparateRadiationStepConfig`,
`SecondaryModuleStepConfig`, `FCN3StepConfig`). Each rejects a `grouped` block
in `__post_init__` rather than parsing it and silently training on pooled
constants.

It is bound **only** at `SingleModuleStep.step`, for the network's inputs and
outputs. `step.normalizer` still returns the pooled normalizer, and the
`NormalizeFn` protocol is unchanged.

That choice is about measurement validity, not just blast radius: if the
aggregators normalized per group, every `*_norm` metric would be in different
units in A1 vs A2 vs A3 and the arms could not be compared on them. Keeping
them pooled fixes the yardstick — and makes the GMR and masking pins fall out
for free rather than being special cases.

Group membership is resolved from the multi-hot label tensor. A sample
resolving to zero or to more than one group raises, rather than silently
picking one.

### Labels

Every dataset entry in every config is labeled with its finest-grained group
(`amip` / `ramped` / `som` / `era5`), including the A1 controls, so cells differ
only in the two axes under test. The coarser A2 grouping is expressed in the
normalization config rather than by relabeling the data.

The vocabulary is per-regime, not global: the c96 regime has no `era5` label.
Train, validation and every inference loader must agree on whether labels are
in use (`train_config.py`), so the generator labels all of them.

`default_group` is a required field, used when a batch carries no labels (e.g.
standalone inference on an unlabeled dataset). It is named explicitly because
an implicit choice would silently normalize against the wrong distribution.
`bind` never falls back to the pooled constants: a model trained under A2 or A3
never saw its network inputs on the pooled scale, so pooled is not a safe
default — it is simply a fourth, untrained distribution.

**Post-hoc eval and inference configs must set labels.** All 44 generated
training configs label every loader, so `default_group` is unreachable during the runs
themselves. It only becomes reachable later, when a checkpoint is evaluated
against a config whose datasets carry no `labels:`. There, the two grouped arms
behave differently:

- **A2/A3 `-cond`** fails loudly — `TypeError: Labels are required for
  conditional models`.
- **A2/A3 without conditioning** is **silent**: every sample is normalized
  against `default_group` with no warning. For the c96 regime that group is
  `amip`, picked alphabetically rather than for any physical reason.

So label the eval datasets, or set `labels:` on `InferenceEvaluatorConfig` /
`InferenceConfig`, which override whatever the dataset carries. A1 is exempt —
it has no `grouped` block and normalizes with the pooled constants either way.

### Unconditional builds no longer see labels

Adding labels surfaced a latent bug: `NoiseConditionedSFNO` with
`conditional: false` on a labeled dataset sized its CLN label weights from
`dataset_info.all_labels`, was then never given labels, and raised
`ValueError: labels must be provided`. The swin adaLN path silently no-opped
instead. Nothing hit this before because no config in the repo had ever set
`labels:`.

`ModuleSelector.build` now builds unconditional modules against
`dataset_info.without_labels()`. Labels reach the normalizer; the module is
built as if there were none. This is what makes the conditioning axis clean —
`conditional: false` now means no label machinery at all, so an A1 cell with
labels is identical to one without.

The two frozen checkpoints in `fme/core/registry/testdata/` do contain label
weights with `label_encoding: None` — built under the old accidental behavior.
Their fixtures are now marked `conditional=True`, which matches what the
checkpoints actually hold; verified they still load byte-compatibly, and
neither `.pt` was regenerated.

### Statistics

`get_pooled_stats.py` gained a `group` / `groups` tag per dataset pair and a
`groups:` list on the config. One `compute` run writes the root pooled stats
plus `groups/{name}/`, all pooled from the same per-store moments. `n_samples`
is written into the netCDF attrs so a later re-pooling has its weights without
re-reading any zarr.

Three stats configs, one per regime — the regimes' store lists do not nest (the
`era5` group of the fm regime covers different time windows than the era5
regime's own data):

| config | stores | groups |
|---|---|---|
| `norm-ablation-c96-stats.yaml` | 11 | amip, ramped, som |
| `norm-ablation-era5-stats.yaml` | 6 | none (single source) |
| `norm-ablation-fm-stats.yaml` | 12 | c96, era5, amip, ramped, som |

Stores are listed explicitly rather than by directory, so held-out ensemble
members (`ic_0003`+) are not swept in — unlike `pooled_stats_0`, which globs
directory roots and therefore includes data no model trains on.

Output goes to a new `norm_ablation_0/{regime}/`, written first to
`gs://vcm-ml-intermediate/alexeyy/` and then copied to
`/climate-default/alexeyy/` on weka. Two hops rather than one because the
training jobs read weka while the analysis notebooks read GCS over `gsutil`,
and the per-member subdirectories `compute` writes are what those notebooks
resolve back to `(store, window)` pairs. This is the same path
`pooled_stats_0` and `shield_random_co2_stats_0` took; both are left frozen as
historical artifacts.

## Running it

The three stats jobs are independent and run in parallel. Everything after
them is gated on the hand verification in step 3.

```bash
# 1. Statistics to GCS (three CPU jobs, run in parallel)
cd scripts/data_process
for regime in c96 era5 fm; do
  python get_pooled_stats.py submit configs/norm-ablation-$regime-stats.yaml \
    gs://vcm-ml-intermediate/alexeyy/norm_ablation_0/$regime \
    --name norm-ablation-$regime
done

# 2. Copy to weka, where the training jobs read from
for regime in c96 era5 fm; do
  ./gcs_to_weka.sh gs://vcm-ml-intermediate/alexeyy/norm_ablation_0/$regime \
    /climate-default/alexeyy/norm_ablation_0/$regime
done

# 3. Verify the statistics by hand -- see below. Blocking.

# 4. Configs (already generated; regenerate only after editing the generator)
cd ../../configs/experiments/2026-06-26-fm
python generate_norm_ablation_configs.py

# 5. Training (unmasked cells; --masking defaults to these)
python submit_norm_ablation_jobs.py --dry-run   # inspect first
python submit_norm_ablation_jobs.py

# 6. Training, masked cells. One masking variant per invocation.
python submit_norm_ablation_jobs.py --masking mask10 --regime fm --arm a1 --dry-run
python submit_norm_ablation_jobs.py --masking mask10 --regime fm --arm a1

# 7. ERA5 fine-tuning of the finished nc-sfno fm cells. Requires the source
#    runs to have finished (update_beaker_map.py), and requires this branch --
#    configs and the fme change alike -- to be pushed: gantry clones HEAD.
python generate_norm_ablation_finetune_configs.py
python submit_norm_ablation_finetune_jobs.py --dry-run
python submit_norm_ablation_finetune_jobs.py
```

### Verifying the statistics

Nothing in the repo checks the written stats, and a bad constant does not
necessarily crash training -- it trains to completion on the wrong scale. So
step 3 is a manual gate: the stats are inspected in the notebooks under
`~/Git/explore2/alexeyy/foundation-model/` before any training is submitted.
`build_member_pool.py` there reads the per-member subdirectories from GCS,
which is why step 1 writes there rather than straight to weka.

Two failure modes are worth looking for specifically, since neither shows up
in the job logs:

- a near-zero group std on a variable outside the pinned list -- the same
  σ→0 blowup `global_mean_co2` is pinned to avoid, but on a variable nothing
  pins, which would poison only the arms that use that group and read as
  "A3 is worse";
- group `n_samples` attrs that do not sum to the root's over a partition
  (`amip + ramped + som + era5` for fm/A3, `c96 + era5` for fm/A2), which
  means a store was dropped from a group or double-tagged.

`submit_norm_ablation_jobs.py` filters with `--arch`, `--regime`, `--arm`, and
`--conditional` / `--no-conditional`. `submit_fm_jobs.py` is untouched.

## ERA5 fine-tuning

Six 10-epoch runs, one per `nc-sfno` fm cell (A1/A2/A3 × conditioning),
warm-started from each source run's last-epoch checkpoint and continued on ERA5
alone. Written by `generate_norm_ablation_finetune_configs.py`, submitted by
`submit_norm_ablation_finetune_jobs.py` into wandb group
`ace2-fm-norm-ablation-finetune-2026-06-26`.

**What it asks.** Not transfer — every source model already saw ERA5 in the
mixture — but *specialization*: does the arm a model was pretrained under leave
it better positioned to be specialized onto ERA5. The fine-tuning phase itself
differentiates nothing. With only `era5` in the train_loader, A2 and A3 both
bind their `era5` group, which is the same `groups/era5/` directory in both, and
the conditional cells see a constant one-hot. Any difference between the six is
entirely a difference between the weights they start from.

That constant one-hot is not the `degenerate_reason` collapse: those CLN weights
were trained with all four labels varying, so it selects an already-learned
column rather than a constant a fresh affine could absorb.

**The transformation.** Each config is its source with:

| field | value |
|---|---|
| `# arg:` header | `--dataset <ID>:/checkpoints`, from `wandb_to_beaker_map.json` |
| `parameter_init` | `/checkpoints/training_checkpoints/ckpt.tar`, both overrides on |
| `train_loader` concat | the 2 ERA5 members; the 10 c96 members dropped |
| `max_epochs` / `lr` | 10 / 1e-5, single `PolynomialLR(power 0.5)`, no warmup |
| `ema_checkpoint_save_epochs` | `{start: 1, step: 1}` |
| `inference[].epochs` | removed, so all 13 entries run every epoch |
| non-ERA5 `inference[].weight` | `0.0` |
| `evaluate_before_training` | `true` |

**Three things are load-bearing.**

*Statistics are copied verbatim* — every path still points at
`norm_ablation_0/fm/`. `parameter_init` loads module weights only and the
normalizer is rebuilt from the YAML with **nothing checking the two agree**, so
re-deriving statistics from the ERA5-only data would silently retrain on a
shifted input space. It would also collapse the experiment: all three arms would
read one identical set of constants. Copying them verbatim means ERA5 samples
get exactly the constants they got in pretraining — pooled under A1, the `era5`
group under A2/A3 — so the input scale is continuous across the warm start.

*The label vocabulary comes from the checkpoint.* `DatasetInfo.all_labels` is
built from the train loader alone, so an ERA5-only loader shrinks it from four
labels to one. That sizes a conditional module's label-dependent weights (CLN
`W_scale_labels` / `W_bias_labels`, `label_pos_embed`) narrower than the
checkpoint's and `overwrite_weights` refuses to load them; and since that path
matches weights positionally against sorted labels, a surviving label would land
in another's column. It also matters for the unconditional cells, whose modules
build through `without_labels()` but whose checkpoints would otherwise record a
different vocabulary than their siblings'. Fixed by
`parameter_init.override_labels_from_weights`, added for this and mirroring the
existing `override_vertical_coordinate_from_weights` — which is set for the same
reason, so the coordinate is not re-derived from the restricted loader.

*Checkpoint selection moves onto ERA5.* Both `10year` (ERA5) and
`10year_insample_ensemble_varying_co2` (C96) carry weight 1.0 in the source
configs, so `best_inference_ckpt` would be selected half on the data the
fine-tune has just stopped training on. The C96 entries stay in the suite as
weight-0 forgetting diagnostics, and still normalize correctly: the `grouped`
config retains all its groups regardless of what is in the train set, so
`amip`/`ramped`/`som` resolve to the `c96` group as before.

**Reading the results.** The loss, global mean removal and every `*_norm` metric
stay on the fm-pooled constants, so those remain comparable to the base runs and
to the existing figures. They are *not* comparable to the `era5-a1` specialist,
which trained under `norm_ablation_0/era5`: fm-pooled
`specific_total_water_0` σ is inflated by the between-source term, so q0 errors
carry less loss weight here. Compare against the specialist on native-unit
metrics (`time_mean/rmse/<field>`, `time_mean/bias/<field>`), not on `loss` or
`*_norm`.

Cost is dominated by inference, not training: ERA5-only takes an epoch from ~163
dataset-years to ~73, while the 13-entry suite (~81k forward steps) runs 11
times rather than the base run's 15 in 150 epochs.

## Caveats

- **No A1 control reproduces its base run's normalization.** Every regime
  reads the new `norm_ablation_0/{regime}` stats instead of its base config's:
  c96 leaves `shield_random_co2_stats_0` (ramped stores only, while training on
  AMIP + ramped + SOM), era5 leaves
  `2026-04-17-era5-4deg-8layer-daily-stats-1990-2019/`, and fm leaves
  `pooled_stats_0` (which globs directory roots and so covers held-out
  members). The A1 arms are proper controls *for this experiment* — same stats
  as their A2/A3 siblings — but none is a rerun of a prior job.
- **Inference weights are unchanged** from each regime source, so checkpoint
  selection is regime-matched and comparable to prior runs in the project. The
  `long_46year` / `long_43year_ensemble_varying_co2` entries already exist at
  weight 0.0 and run as diagnostics.
- **No primary metric is pre-registered.** With 22 runs (26 once the first
  masked cells land), ~40 variables and multiple lead times, something will
  look better by chance; pick the decision metric before reading results.
- **YAML anchors are expanded** by the `safe_load`/`dump` round-trip, so the
  generated configs repeat the `inference_variables` block. Cosmetic; the
  cooldown generator does the same.
- **Fine-tuning is implemented for the fm regime only** (see ERA5
  fine-tuning below). The c96 → ERA5 arm the original plan named is still
  not implemented, and is a different experiment: those cells have never
  seen ERA5, and A3 has no `era5` group to bind to.
- **A masked cell is not weight-shaped like its unmasked twin.**
  `include_channel_mask_inputs` doubles the module's input channels (45 → 90),
  so a `mask10` run differs from its baseline in parameter count as well as in
  training distribution. The comparison is behavioral, not a weight diff, and
  no masked run can warm-start from an unmasked checkpoint. Setting the flag
  everywhere would fix that but would change the 22 configs already trained.
- **Masking is uniform over all 45 channels.** No counterpart to the pinned
  variable list: `global_mean_co2`, the statics and `DSWRFtoa` are all in the
  default pool, matching `nc-sfno-fm-random-v2-mask10`. Dropping
  `global_mean_co2` teaches "unknown CO2 → pooled-mean forcing" on a stream
  that spans 1x/2x/4x, which is the one channel whose masked value is
  physically wrong rather than merely absent. Left in for comparability with
  the existing mask10 runs; a `rate: 0` override group would carve it out.

## Deviations from the original plan

Recorded so the reasoning survives. Everything else was built as specified.

**Hypothesis restated.** The plan said pooled stats let the model "partition
behavior by source instead of learning shared physics." But `global_mean_co2`
is pinned and stays a perfect discriminator (JS = 0.6931, overlap 0.000), so
the model can still partition by source for free. The experiment cannot test
denial of discriminability; it tests input-distribution alignment. Reworded
rather than dropping the CO2 pin, since the pin is load-bearing numerically.

**Base configs: 2 → 3, split into regime and architecture roles.** The plan's
line 3 named `...nc-swin-v2-fm-random-v3.yaml`, which does not exist, and its
config list named two bases. Only 2 of the 6 (arch × regime) cells actually
had a base. Added `sfno-v2` as the era5 regime source and made the composition
explicit: regime source supplies data/validation/inference, architecture source
supplies builder, `residual_prediction` and `in_names` ordering. The three
bases are byte-identical outside the builder, so this composes cleanly.

**Config count: 18 → 22, and conditioning became a real axis.** The plan's
`3*3*2=18` counted every arm in every regime. Six of those cells train models
identical to their A1 control (an arm with one group has nothing to select per
sample), so they are skipped. Separately, the plan's "dataset label (on, off)"
axis became module conditioning on/off: after the `without_labels()` fix below,
label *presence* no longer changes an unconditional model at all, so on/off
would have been duplicate trainings, whereas conditioning is a genuinely
different mechanism. Skipping degenerate cells in both axes leaves 11 per
architecture.

**Naming: `nc-{arch}-{regime}-{arm}[-cond]`.** The plan's `nc-sfno-<data>` had
no field for the arm or for conditioning, so it could not name 11 cells.

**A separate `GroupedNormalizer` instead of modifying `StandardNormalizer`.**
The plan put the `[n_samples]` gather inside `StandardNormalizer`. That class is
read by ~10 consumers outside the network — the loss, global mean removal,
spatial masking fill values, and every aggregator's `*_norm` metric. Making
them per-group would put those metrics in different units in A1 vs A2 vs A3 and
make the arms incomparable on exactly the numbers used to judge them. So the
gather lives in a separate class, bound only at the step seam, and
`step.normalizer` still returns the pooled normalizer. `bind()` returns a plain
`StandardNormalizer` holding `[n_samples, 1, 1]` constants, which also meant
`step_with_adjustments` needed no signature change and the three other step
types sharing it were untouched.

**`grouped` sits alongside `network`, not inside it.** The pooled constants are
needed anyway (pins, GMR, masking, aggregators, unlabeled fallback), so they
stay in `network` and `grouped` layers on top. A1 therefore omits `grouped`
entirely and runs the pre-existing code path unchanged — a stronger control
than a one-group config would be.

**`default_group` added as a required field.** Not in the plan. `BatchLabels`
is `None` for inference on an unlabeled dataset, so a fallback group is
unavoidable; making it required avoids silently normalizing against the wrong
distribution. Both no-label cases route to it — `labels=None` and a
`BatchLabels` carrying zero names — since falling back to pooled would hand an
A2/A3 model a scale it never trained on.

**Unconditional builds no longer see labels.** The plan said to route labels to
the normalizer and noted non-conditional builders hard-error on labels. The
real defect was worse: `NoiseConditionedSFNO` sized its CLN label weights from
`dataset_info.all_labels` regardless of `conditional`, was then never handed
labels, and raised `ValueError: labels must be provided`. Fixed centrally in
`ModuleSelector.build`. This changed existing behavior, so: two swin tests that
asserted an unconditional model *does* allocate label weights now assert it
does not, and the two frozen checkpoint fixtures were marked `conditional=True`
to match what their `.pt` files actually contain (neither file regenerated,
both verified to still load).

**`InferenceEvaluatorConfig.labels` added.** Not in the plan. The evaluator had
no way to supply labels, unlike `InferenceConfig`.

**Labels on every loader, not only `train_loader`.** `train_config.py` requires
the train, validation and inference loaders to agree on whether labels are in
use, so the generator labels all of them.

**Stats regenerated from scratch into a new directory.** The plan expected
re-pooling subsets of the existing `pooled_stats_0` moments with "no zarr
reads." Two problems. Sample counts were never persisted, so the pooling
weights would have had to be re-derived from the config's time bounds — now
fixed by writing `n_samples` into the netCDF attrs. More seriously,
`pooled_stats_0` was built from a config that globs directory roots, so it
includes held-out `ic_0003`+ members that appear in no `train_loader`; the A1
control would have been normalized by statistics computed over a superset of
its own training data. The new configs list every store explicitly. Three runs
rather than one, because the regimes' store lists do not nest (the fm regime's
`era5` group covers different time windows than the era5 regime's own data).
`pooled_stats_0` is left untouched.

**Input masking added as a third axis.** Not in the plan at all; added after
the first 22 runs, to ask whether synthetic input dropout buys the same
out-of-sample generalization the grouping arms were built for, and whether the
two compose. Modelled as an axis (`MASKINGS`) rather than as a fourth arm
because it is not a grouping strategy and has to be able to cross with A2 and
A3. The generator writes every cell of the cross; submission is filtered.

**Not deviations.** Pinned variable list, pooled GMR, pooled loss/residual
normalizer, and deferring the c96 → ERA5 fine-tuning arm are all as planned.

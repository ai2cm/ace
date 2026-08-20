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

### 22 configs

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

11 per architecture, **22 total**. The generator logs the 14 it skips with the
reason. `--include-degenerate` writes them anyway, and the same flag on
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

**Post-hoc eval and inference configs must set labels.** The 22 training
configs label every loader, so `default_group` is unreachable during the runs
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

# 5. Training
python submit_norm_ablation_jobs.py --dry-run   # inspect first
python submit_norm_ablation_jobs.py
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
- **No primary metric is pre-registered.** With 22 runs, ~40 variables and
  multiple lead times, something will look better by chance; pick the decision
  metric before reading results.
- **YAML anchors are expanded** by the `safe_load`/`dump` round-trip, so the
  generated configs repeat the `inference_variables` block. Cosmetic; the
  cooldown generator does the same.
- **Fine-tuning (c96 → ERA5) is not implemented**, as planned.

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

**Not deviations.** Pinned variable list, pooled GMR, pooled loss/residual
normalizer, and deferring the c96 → ERA5 fine-tuning arm are all as planned.

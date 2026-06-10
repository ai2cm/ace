# Distillation Architecture: how ACE plugs into FastGen

This document captures how the distillation code is wired into
[FastGen](https://github.com/NVlabs/FastGen) and, in particular, where every
teacher-dependent parameter lives.  Read this before changing teacher
checkpoints, noise schedules, or spike configs.  See `README.md` for how to
run the spikes.

## Component map

| File | Role |
|---|---|
| `fastgen_train.py` | Entry point. Loads the teacher, builds the ACE data pipeline, injects both into a FastGen config, runs FastGen's `Trainer`. |
| `fastgen_teacher.py` | `AceDiffusionTeacher` — adapter making an ACE `DiffusionModel` (or `DenoisingMoEPredictor`) look like a `FastGenNetwork`. Also `AceEDMNoiseSchedule` (adds `"loguniform"`). |
| `fastgen_loader.py` | `AceConditionBuilder` — turns coarse-only ACE batches into FastGen `{real, condition}` batches by *sampling the teacher* for x0 targets. |
| `configs/*_spike.py` | Per-method FastGen configs (SFT, f-distill, sCM, DMD2). Python modules exposing `create_config()`, parameterized by `ACE_*` env vars. |
| `best_student_callback.py` | Validation-time CRPS/tail scoring against a pre-generated teacher zarr; saves `best_student.ckpt` in ACE format. |
| `student_checkpoint.py` | Exports student weights as a drop-in ACE checkpoint (`sampler_type: "fastgen"`). |
| `generate_val_dataset.py` | One-off job that samples the teacher over the validation split and writes the zarr used by the callback. |
| `configs/experiments/2026-05-18-distillation-with-val/run.sh` (repo root) | Beaker/gantry submission. Chooses teacher checkpoint, val dataset, and sets the `ACE_*` env vars per teacher. |

FastGen itself is a **clean, unmodified clone of NVlabs/FastGen** at the repo
root (`Fastgen/`, copied into the distillation Docker image by
`docker/Dockerfile`).  Do not patch it — anything ACE-specific belongs in an
adapter or subclass in this package (e.g. `AceEDMNoiseSchedule`).

## How FastGen consumes the ACE model

FastGen's `Trainer` instantiates `config.model.net` (a `LazyCall`) once for
the student and once for the frozen teacher copy.  `fastgen_train.py` sets

```python
config.model.net = L(_copy_ace_teacher)(teacher=teacher)
```

so each `instantiate()` returns an independent `deepcopy` of the loaded
`AceDiffusionTeacher`.  Consequences:

- The student is **initialized from teacher weights** (no
  `pretrained_model_path` loading; it is set to `""`).
- `AceDiffusionTeacher.__deepcopy__` deliberately drops `_moe_experts` so
  student/teacher copies carry only the **primary expert's** weights, not the
  whole MoE.  Only the *original* instance (used by `AceConditionBuilder` for
  target generation) keeps the full expert list and sigma-dispatch.
- `forward(x_t, t, condition)` maps directly onto ACE's
  `EDMPrecond`-preconditioned UNet: FastGen's `t` *is* EDM sigma, and the
  return is an x0 prediction (`net_pred_type="x0"`, `schedule_type="edm"`,
  `sigma_data=1` from standard-score normalization).

The DMD2 discriminator is auto-wired in `fastgen_train.py` by introspecting
the teacher UNet's encoder blocks (`encoder_feature_info()`), replacing
FastGen's CIFAR-sized default.  For an MoE teacher, introspection uses the
primary expert (the sigma-dispatch wrapper is not an `nn.Module`).

## Noise schedule: the three places sigma parameters live

This is the part that has bitten us.  There are **three independent layers**,
and only one of them follows the teacher checkpoint automatically.

### 1. The schedule object (automatic — from the checkpoint)

`AceDiffusionTeacher.__init__` reads `sigma_min` / `sigma_max` from the loaded
model (`DiffusionModel.config`, or `DenoisingMoEPredictor._sigma_schedule_min/max`
for a bundle) and passes them as `min_t` / `max_t` to the
`FastGenNetwork` constructor, which builds the noise schedule.  This controls:

- the Karras sigma grid (`schedule.sigmas`, used by `"polynomial"` t-sampling),
- initial-noise scaling (`schedule.latents`: `noise * sigma(max_t)`),
- the student's sampling `t_list` (`get_t_list` spans `[max_t, 0]`).

Caveat fixed in `AceDiffusionTeacher.set_noise_schedule`: FastGen's
`reset_parameters()` (FSDP path) rebuilds the schedule **with no kwargs**,
i.e. EDM defaults sigma ∈ [0.002, 80].  Our override remembers the
construction kwargs so the teacher's range survives.

### 2. `sample_t_cfg` (manual — spike configs + `ACE_*` env vars)

Every FastGen method draws training noise levels via

```python
self.net.noise_scheduler.sample_t(n, **convert_cfg_to_dict(self.config.sample_t_cfg))
```

so **all fields of `SampleTConfig` are forwarded as kwargs**, including
`min_t` / `max_t`, whose FastGen defaults are `[0.002, 80]`.  `sample_t`
clamps its output to these — with the defaults, training silently never
samples sigma above 80 regardless of what the teacher supports.  The spike
configs therefore must set:

- `sample_t_cfg.time_dist_type` — the distribution family,
- `sample_t_cfg.train_p_mean/train_p_std` — lognormal parameters (ignored by
  loguniform/polynomial),
- `sample_t_cfg.min_t/max_t` — must span the teacher's full sigma range; they
  truncate the lognormal and *parameterize* the loguniform.

These are env-driven (`ACE_NOISE_DIST`, `ACE_SIGMA_MIN`, `ACE_SIGMA_MAX`) with
defaults matching the original single-model teacher; `run.sh` overrides them
for `--moe-teacher`.

### 3. The teacher's actual training distribution (ground truth)

On the ACE side, `DiffusionModelConfig.training_noise_distribution` is either
`LogNormalNoiseDistribution(p_mean, p_std)` or
`LogUniformNoiseDistribution(p_min, p_max)` (`fme/downscaling/noise.py`).
Distillation should sample t where the teacher is competent, so layer 2 must
mirror this.  FastGen's stock `EDMNoiseSchedule` has no log-uniform option —
`AceEDMNoiseSchedule` in `fastgen_teacher.py` adds `time_dist_type="loguniform"`
(`t = exp(uniform(log min_t, log max_t))`).

### Known teacher parameter sets

| Teacher | Checkpoint | Out channels | Training noise | sigma range | Gen. steps |
|---|---|---|---|---|---|
| Single-model CONUS (default) | `01KNM6H3JB1ZNS76HX17AAZRF7` / `best_histogram_tail.ckpt` | 1 (`PRATEsfc`) | lognormal(−1.2, 1.8) | [0.002, 150] | 15 |
| Multivariate MoE (2026-05) | `01KTCHVDHY0SATWH9E0AW2PDS6` / `bundled_moe_multivariate.ckpt` | 5 (u10, v10, PRMSL, PRATEsfc, T2m) | loguniform | [0.005, 200] | 18 |

When adding a teacher, update the corresponding branch in `run.sh` (env vars
`ACE_C_OUT`, `ACE_NOISE_DIST`, `ACE_SIGMA_MIN`, `ACE_SIGMA_MAX`, and
`--teacher-num-steps`) — the spike config defaults only cover the
single-model teacher.

## `input_shape` / `ACE_C_OUT` matters beyond logging

`config.model.input_shape = [C_OUT, H_FINE, W_FINE]` is used by the methods
to draw generator noise (`torch.randn(batch, *input_shape)` in DMD2, which
f-distill inherits; similarly SFT/causvid/ladd) and by `fastgen_train.py` to
derive the coarse patch extent (`H_FINE // downscale_factor`).  A channel
mismatch with the teacher's `out_names` breaks training — it is *not*
auto-derived from the checkpoint.

## Data flow (no fine data needed)

The training data yaml is **coarse-only**.  `AceConditionBuilder` runs the
(frozen, full-MoE) teacher's EDM sampler on each coarse batch at training
time to produce the `real` (x0 target) tensor, and builds the dense
`condition` tensor the same way ACE's `DiffusionModel` does (interpolated
coarse inputs + optional fine topography).  `--teacher-num-steps` controls
the sampler steps for target generation and should match the teacher's
configured `num_diffusion_generation_steps`.

Patching: when the domain is larger than `input_shape // downscale_factor`,
`AceInfiniteDataLoader` iterates shuffled, randomly-offset patches; ACE's
distributed sampler logic is bypassed (`Distributed(force_non_distributed=True)`)
because FastGen's DDP wrapper handles replication — every rank sees the full
dataset.

## Runtime patches in `fastgen_train.py` (easy to forget they exist)

- `torch.load` is globally patched to `weights_only=False` (ACE checkpoints
  contain numpy scalars FastGen's allowlist rejects).
- `fastgen.callbacks.wandb.to_wandb` is patched to replicate channel 0 to RGB
  (FastGen asserts C==3 for sample images).
- EMA callbacks from the method defaults are stripped and replaced
  (`EMA_CONST_CALLBACKS` in the spike configs; `config.model.use_ema = []` is
  forced in `fastgen_train.py` since EMA objects don't exist for the injected
  net).
- `grad_accum_rounds` is recomputed from the *actual* per-GPU batch size in
  the data yaml versus `trainer.batch_size_global`.
- `_CheckpointPruner` keeps only the most recent 20 raw `.pth` checkpoints.
- `BestStudentCheckpointCallback` (when `--val-dataset`/`--val-data-yaml` are
  given) scores the student against the pre-generated teacher zarr and writes
  ACE-format `student_checkpoints/best_student{,_tail}.ckpt`; for an MoE
  teacher it receives the primary expert as the reference `DiffusionModel`.

## Config override mechanics

Spike configs are Python modules (`create_config()`), imported by dotted path
from `--config <path>.py`.  They are parameterized by `ACE_*` env vars **read
at import time on the training host** — so for Beaker runs the values must be
passed as `gantry --env` flags (see `run.sh`), not exported locally.
CLI overrides use FastGen's `- key=value` remainder syntax and are applied
immediately after `create_config()` — i.e. *before* `fastgen_train.py` injects
the teacher, dataloader, and discriminator, so those injections cannot be
overridden from the CLI.

## MoE teacher caveats

- The student is a **single model** distilled from the full MoE: target
  generation uses sigma-dispatch over all experts, but the student (and the
  frozen teacher copy used inside the FastGen loss) is initialized from the
  primary expert only.
- DMD2 + MoE is not fully supported: discriminator feature extraction hooks
  the primary expert's UNet only (see note in `run.sh`).

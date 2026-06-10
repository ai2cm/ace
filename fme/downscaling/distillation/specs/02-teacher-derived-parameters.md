# 02 — Derive teacher-dependent parameters from the checkpoint

## Goal

Eliminate the failure mode where distillation silently trains against the
wrong noise distribution, sigma range, or channel count because an operator
forgot an env var.  Everything derivable from the teacher checkpoint is
derived; anything not derivable is validated against the checkpoint at
startup.  The env vars `ACE_C_OUT`, `ACE_H_FINE`, `ACE_W_FINE`,
`ACE_NOISE_DIST`, `ACE_SIGMA_MIN`, `ACE_SIGMA_MAX` are deleted.

## Why this exists (history)

The MoE multivariate teacher (loguniform σ∈[0.005, 200], 5 channels) was
added with the spike configs still hardcoding the old single-model teacher's
parameters (lognormal(−1.2, 1.8), 1 channel) — and FastGen's `SampleTConfig`
defaults (`min_t=0.002, max_t=80`) were silently clamping all training noise
to σ≤80 on top of that.  The 2026-06 fix made these env-configurable; this
task makes them automatic.

## Verified mechanics (do not re-derive)

- Every FastGen method draws training noise via
  `self.net.noise_scheduler.sample_t(n, **convert_cfg_to_dict(self.config.sample_t_cfg))`
  — *all* `SampleTConfig` fields are forwarded, including `min_t`/`max_t`
  which clamp the output (`FastGen/fastgen/configs/config.py:77`,
  `fastgen/networks/noise_schedule.py` `EDMNoiseSchedule.sample_t`).
- The schedule object itself already follows the checkpoint:
  `AceDiffusionTeacher.__init__` (`fme/downscaling/distillation/fastgen_teacher.py`)
  reads `sigma_min`/`sigma_max` from `DiffusionModel.config` or, for MoE,
  `DenoisingMoEPredictor._sigma_schedule_min/_sigma_schedule_max`, and
  passes them as `min_t`/`max_t` to `FastGenNetwork.__init__`.
- The teacher's training distribution is available after loading:
  `DiffusionModel.config.noise_distribution` returns
  `LogNormalNoiseDistribution(p_mean, p_std)` or
  `LogUniformNoiseDistribution(p_min, p_max)`
  (`fme/downscaling/models.py:231-246`, `fme/downscaling/noise.py`).  For an
  MoE bundle use the primary expert: `model._primary.config` — but note each
  expert may have its own distribution; the primary's is the convention
  already used elsewhere (e.g. `BestStudentCheckpointCallback` receives
  `teacher_model._primary`).
- Channel/shape facts: `len(model.out_packer.names)` is the output channel
  count (already used by `AceConditionBuilder.build_fastgen_batch`,
  `fastgen_loader.py:86`).  `model.downscale_factor` and the spike's
  `coarse_patch_extent` determine `H_FINE`/`W_FINE` (currently
  `512 = 16 * 32`).  `config.model.input_shape` feeds
  `torch.randn(batch, *input_shape)` for generator noise in
  DMD2/f-distill/SFT (`FastGen/fastgen/methods/distribution_matching/dmd2.py:96`)
  and the patch-extent computation in `fastgen_train.py` — a channel
  mismatch with the teacher breaks training and is **not** currently
  validated anywhere.
- `AceEDMNoiseSchedule` (in `fastgen_teacher.py`) already supports
  `time_dist_type="loguniform"`.

## Design

Add a function in a new module `fme/downscaling/distillation/auto_config.py`:

```python
def apply_teacher_parameters(config, teacher_model, *, fine_shape_yx) -> None:
    """Mutate a FastGen config so sample_t_cfg and input_shape match the teacher."""
```

Behavior:

1. `config.model.input_shape = [len(out_names), *fine_shape_yx]`.
2. `config.model.sample_t_cfg.min_t = sigma_min`;
   `config.model.sample_t_cfg.max_t = sigma_max` (from the teacher).
3. Map the teacher's noise distribution:
   - `LogNormalNoiseDistribution` → `time_dist_type="lognormal"`,
     `train_p_mean=p_mean`, `train_p_std=p_std`.
   - `LogUniformNoiseDistribution` → `time_dist_type="loguniform"`
     (its parameters are exactly `min_t`/`max_t`; assert
     `p_min == sigma_min and p_max == sigma_max`, warn if not and prefer
     the distribution's values for `min_t`/`max_t`).
4. **Exception:** a spike/method config may deliberately choose a different
   t-distribution (DMD2 uses `"polynomial"`; the spikes historically widened
   `p_std` 1.2→1.8).  Support an opt-out: only overwrite `time_dist_type`
   and `train_p_*` when the method config sets a sentinel
   (`time_dist_type="teacher"`); always overwrite `min_t`/`max_t`
   unconditionally (there is no legitimate reason to clamp inside the
   teacher's range).  Update the four spike configs: fdistill/scm/sft use
   `"teacher"`, dmd2 keeps `"polynomial"`.
5. Call it from `fastgen_train.py` **after** teacher load and **after**
   CLI overrides, logging every field it sets (operators must be able to see
   the resolved values in the job log).
6. Delete the `ACE_C_OUT/H_FINE/W_FINE/NOISE_DIST/SIGMA_MIN/SIGMA_MAX`
   reads from `configs/*_spike.py` and the corresponding `--env` flags from
   `configs/experiments/2026-05-18-distillation-with-val/run.sh`
   (`TEACHER_ENV_FLAGS`).  `ACE_STUDENT_STEPS` and `ACE_TEACHER_CKPT` stay
   (genuinely operator-chosen).
7. `--teacher-num-steps` default: when the flag is 0/absent, the teacher's
   own `num_diffusion_generation_steps` is already used by
   `AceDiffusionTeacher.sample` — make `run.sh` stop passing it explicitly
   (delete `TEACHER_NUM_STEPS`) unless an override is wanted.

`fine_shape_yx` source: keep `ACE_H_FINE`/`ACE_W_FINE`?  No — move patch
size into the data/launch config (it is an experiment choice, not a teacher
property).  Until spec 03 lands, accept a `--fine-shape H W` CLI arg
defaulting to 512 512.

## Tests (add to `test_teacher.py` or new `test_auto_config.py`)

- Lognormal teacher → cfg gets lognormal + exact p_mean/p_std + sigma range.
- Loguniform teacher → cfg gets loguniform + range.
- `time_dist_type="polynomial"` in the method config survives (opt-out), but
  `min_t`/`max_t` are still corrected.
- Channel count mismatch raises with a message naming both values.
- Use `_build_small_model(...)` from `test_teacher.py` (already
  parameterized over `sigma_min`/`sigma_max`); build a small model with
  `training_noise_distribution=LogUniformNoiseDistribution(...)` for the
  loguniform case.

## Acceptance criteria

- A run launched with *zero* noise-related env vars against either teacher
  produces a logged `sample_t_cfg` matching that teacher's training config.
- `grep -r ACE_SIGMA fme/ configs/` returns nothing.
- New tests pass; existing `test_teacher.py` tests unchanged.

## Out of scope

YAML config schema (spec 03); validating that each MoE *expert*'s noise
distribution matches the primary (record in spec 08 instead).

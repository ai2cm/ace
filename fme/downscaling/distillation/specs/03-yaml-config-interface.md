# 03 — Replace Python spike configs with a YAML `DistillationConfig`

## Goal

Distillation is configured the way everything else in this repo is: a single
YAML file parsed into dataclasses with `dacite(strict=True)`, validated in
`__post_init__`, launched as
`torchrun ... -m fme.downscaling.distillation.train config.yaml`.
The four `configs/*_spike.py` modules stop being user-facing configuration
and become an internal "method defaults" layer.  Env-var-at-import-time
configuration is eliminated.

## Current state (verified)

- `fastgen_train.py` imports a spike `.py` by dotted path
  (`--config fme/downscaling/distillation/configs/fdistill_kl_spike.py`),
  calls its `create_config()` → FastGen attrs/omegaconf config object, then
  applies CLI overrides via FastGen's `override_config_with_opts` (the
  `- key=value` remainder convention).
- Data is a *separate* YAML (`--data-yaml`) parsed by `_load_data_config`
  into `fme.downscaling.data.config.DataLoaderConfig` (accepts a bare dict
  or one nested under `train_data:`); validation data is two more flags
  (`--val-dataset` zarr + `--val-data-yaml`).
- Operator-tunable knobs scattered across spike configs today:
  `ACE_STUDENT_STEPS` (per-method default 1/2/2/18), optimizer LRs, GAN loss
  weight (`gan_loss_weight_gen`), f-distill ratio clip params, sCM loss
  flags (`use_cd`, `use_jvp_finite_diff`), `trainer.batch_size_global`,
  `max_iter`, `save_ckpt_iter`, `logging_iter`, EMA list, `log_config.*`.
- The four methods map to FastGen default-config factories:
  `fastgen.configs.methods.config_{sft,f_distill,scm,dmd2}.create_config()`.
  Note sCM's config has a *different* `SampleTConfig` subclass and a
  `loss_config`; the methods do not share a config schema, which is why the
  spike layer exists.

## Design

New module `fme/downscaling/distillation/config.py`:

```python
@dataclasses.dataclass
class DistillationConfig:
    method: Literal["sft", "fdistill", "scm", "dmd2"]
    teacher: TeacherConfig            # checkpoint path or moe_checkpoint path
    train_data: DataLoaderConfig      # reuse existing class, coarse-only
    experiment_dir: str
    student_sample_steps: int | None = None   # None → method default
    fine_patch_shape: tuple[int, int] = (512, 512)
    teacher_num_steps: int | None = None      # None → teacher's configured steps
    validation: ValidationConfig | None = None  # val zarr + coarse val data
    logging: LoggingConfig            # project/group/name → fastgen log_config
    overrides: dict[str, Any] = field(default_factory=dict)  # raw fastgen keys
```

- `TeacherConfig.__post_init__` enforces exactly one of
  `checkpoint_path` / `moe_checkpoint_path` (mirrors the current mutually
  exclusive `--teacher-checkpoint` / `--teacher-moe-checkpoint`).
- `overrides` is the escape hatch: a flat dot-path dict applied with
  FastGen's `override_config_with_opts` *after* the method defaults and
  *before* teacher injection.  This keeps full FastGen tunability without
  promoting every knob into the schema.  Promote a knob to a typed field
  only when it is used by >1 experiment (start with the list above:
  `lr`, `gan_loss_weight_gen`, `batch_size_global`, `max_iter`).
- `build_fastgen_config(cfg: DistillationConfig)` does, in order:
  1. dispatch `method` → spike-module `create_config()` (these keep the
     per-method tuning rationale and comments — do not flatten them away);
  2. apply typed fields;
  3. apply `overrides`;
  4. (spec 02) `apply_teacher_parameters(...)` after teacher load.
- Spike modules lose all `os.environ` reads (spec 02 removes the noise ones;
  this task removes `ACE_STUDENT_STEPS`, `ACE_TEACHER_CKPT` defaults,
  `TEACHER_CKPT_PATH`, `C_OUT/H_FINE/W_FINE` constants).  Their
  `create_config()` becomes parameterless and pure.
- New entrypoint `fme/downscaling/distillation/train.py` with
  `main(yaml_path)`; keep `fastgen_train.py` as a thin deprecated shim for
  one release (it is referenced by `run.sh` and the README) or delete it in
  the same PR if spec 05 lands together — coordinate.

## Validation rules (in `__post_init__`, per AGENTS.md)

- `method` ∈ the four known methods (Literal handles it via dacite strict).
- `validation` requires both zarr path and data config.
- `fine_patch_shape` divisible by the teacher's `downscale_factor` — NOT
  checkable in `__post_init__` (needs the checkpoint); do it in
  `build_fastgen_config` with a clear error.
- Reject `overrides` keys that collide with typed fields (e.g.
  `model.student_sample_steps`) to avoid two sources of truth.

## Example YAML (put in `configs/experiments/.../distill-fdistill-moe.yaml`)

```yaml
method: fdistill
teacher:
  moe_checkpoint_path: /checkpoints/bundled_moe_multivariate.ckpt
experiment_dir: /results
student_sample_steps: 2
train_data:
  coarse: [...]      # same shape as current data-config.yaml
  batch_size: 4
  num_data_workers: 2
  strict_ensemble: false
validation:
  teacher_dataset: /climate-default/2026-06-09-.../conus_multivar_val_2023.zarr
  data: {...}        # current val-data-config.yaml contents
logging:
  project: ace-distillation
  name: fdistill-moe-2step
overrides:
  model.gan_loss_weight_gen: 1.0e-3
  trainer.max_iter: 100000
```

## Tests

- Round-trip: YAML → dacite → `build_fastgen_config` with a tiny teacher
  (use `_build_small_model` from `test_teacher.py`) produces a FastGen
  config whose fields match; overrides apply; collisions raise.
- Each method dispatches to the right FastGen factory (can be asserted via
  `type(config.model_class)` / known per-method field, no GPU needed).
- Bad YAML (unknown key) fails via dacite strict — one test as a guard.

## Acceptance criteria

- A full training run is launchable from one YAML + torchrun, no env vars
  except infra ones (`WANDB_API_KEY`, `FASTGEN_OUTPUT_ROOT`).
- `configs/*_spike.py` contain no `os.environ` and no module-level mutable
  state; README updated.
- mypy clean (the FastGen config objects are untyped omegaconf/attrs —
  isolate them behind `build_fastgen_config` so annotations stay honest).

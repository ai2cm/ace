# Productionization specs for the distillation pipeline

This folder contains self-contained task specs for turning the distillation
spike (`fme/downscaling/distillation/`) into a durable, reviewable pipeline
that can be merged to `main`.  Each spec is written to be executed
independently by an agent **without re-exploring the codebase** — required
file paths, verified behaviors, and pitfalls are stated inline.

Before starting any task, read (both co-located with this folder):

- `../ARCHITECTURE.md` — how ACE plugs into FastGen, the three layers of
  noise-schedule parameters, per-teacher parameter table, runtime patches.
- `../README.md` — current (spike-era) run instructions.

## Task index and suggested order

| Spec | Title | Depends on |
|---|---|---|
| `01-fastgen-optional-dependency.md` | Install FastGen as an optional pip extra; drop the submodule | — |
| `02-teacher-derived-parameters.md` | Derive channels/noise/sigma parameters from the teacher checkpoint; delete `ACE_*` env vars | — |
| `03-yaml-config-interface.md` | Replace Python spike configs with a YAML-loaded `DistillationConfig` | 02 |
| `04-entrypoint-refactor.md` | Decompose `fastgen_train.py`; contain the global runtime patches | 02, 03 |
| `05-experiment-launcher.md` | Replace `run.sh` with a teacher registry + durable submission path | 03 |
| `06-testing-and-ci.md` | Test strategy for an optional heavyweight dependency | 01 |
| `07-docs.md` | Consolidate README/ARCHITECTURE; user-facing docs | all |
| `08-known-gaps-and-hardening.md` | Backlog of known correctness/robustness gaps | — |
| `09-inference-sigma-schedule-fidelity.md` | Carry the trained sigma list into the exported checkpoint; stop re-deriving it at inference | — |

Tasks 01 and 02 are independent and unblock everything else; do them first.

## Shared context (verified 2026-06-10)

**Repo conventions (from `AGENTS.md`, binding for all tasks):**

- Config classes loaded from user YAML end in `Config`; validate in
  `__post_init__`, not at runtime.  YAML is parsed with
  `dacite.from_dict(..., config=dacite.Config(strict=True))`.
- Checkpoint-loading backward compatibility is critical for *inference*;
  training-side compat may be broken with deprecation warnings.
- New functionality needs tests (`python -m pytest`, conda env `fme`);
  pre-commit runs ruff, ruff-format, mypy.  `isinstance` checks and
  `type: ignore` require written justification.
- Branch naming: `feature/...`, `refactor/...`, etc.  Atomic commits.

**FastGen facts:**

- Git submodule at repo root `FastGen/`, pinned to upstream
  `NVlabs/FastGen @ 123e6a2f92d5c851403b75ad6cb5ee4337c88e3c`
  ("Remove one-logger-utils (#16)").  Working tree is clean — **we carry no
  patches**, so installing the same commit from GitHub is lossless.
- It is pip-installable (`docker/Dockerfile:60-65` does
  `uv pip install /tmp/fastgen/ --system`), with one packaging wart: the
  git-ignored `fastgen/third_party/annotators/` directory needs a `touch
  .../__init__.py` before install or `find_packages()` misses it.
- Import-time dependency chain (observed): `loguru`, `ftfy`, `einops`,
  `imageio`, `Pillow`, `omegaconf`, `attrs`, `diffusers`, `scipy`,
  `numpy`, `torch`; spike README additionally lists `hydra-core`, `boto3`,
  `torchvision`.  `diffusers` and `loguru` are **not** in the `fme` conda
  env today — distillation tests cannot currently run without stubs.
- ACE-side extensions live in `fme/downscaling/distillation/` (e.g.
  `AceEDMNoiseSchedule` adds a `"loguniform"` time distribution).  Policy:
  never patch the FastGen tree; subclass/adapt in our package.

**Current spike surface to be replaced:**

- Entry point: `python -m fme.downscaling.distillation.fastgen_train
  --config <spike>.py --teacher-checkpoint|--teacher-moe-checkpoint ...
  --data-yaml ... [--val-dataset ... --val-data-yaml ...]
  [- fastgen.key=value ...]` (the `-` remainder is FastGen's
  `override_config_with_opts` convention).
- Method configs: `configs/{sft,fdistill_kl,scm,dmd2_baseline}_spike.py`,
  Python modules exposing `create_config()`, parameterized by env vars read
  at import time: `ACE_TEACHER_CKPT`, `ACE_TEACHER_MOE_CKPT`, `ACE_C_OUT`,
  `ACE_H_FINE`, `ACE_W_FINE`, `ACE_STUDENT_STEPS`, `ACE_NOISE_DIST`,
  `ACE_SIGMA_MIN`, `ACE_SIGMA_MAX`.
- Submission: `configs/experiments/2026-05-18-distillation-with-val/run.sh`
  (gantry/Beaker), which hardcodes per-teacher datasets, env vars, and
  `--teacher-num-steps` in shell branches.

**Known teacher parameter sets** (see `../ARCHITECTURE.md` for the table):
single-model CONUS (1 channel, lognormal(−1.2, 1.8), σ∈[0.002, 150],
15 steps) vs. multivariate MoE (5 channels, loguniform, σ∈[0.005, 200],
18 steps).

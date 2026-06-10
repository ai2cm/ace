# 06 — Test strategy for an optional heavyweight dependency

## Goal

Distillation tests run in CI and locally without hand-built stubs, are
skipped cleanly where the `distillation` extra is absent, and cover the
adapter seams that have actually broken (noise schedule, channel counts,
checkpoint format) rather than FastGen internals.

## Current state (verified)

- `test_teacher.py` (10 tests) and `test_condition.py` exist and import
  fastgen **unconditionally** — in the standard `fme` conda env they fail at
  *collection* (`ModuleNotFoundError: fastgen` … then `diffusers`, `loguru`,
  `ftfy` when fastgen is on `PYTHONPATH`).  During development they were run
  with ad-hoc `sys.modules` stubs for diffusers/loguru/ftfy/imageio — that
  worked (all passed) but is not committed anywhere and should not be.
- `make test` / `test_fast` / `test_very_fast` run repo-wide pytest; the
  distillation directory currently poisons collection in envs without
  fastgen.
- `best_student_callback.py` (715 lines: ensemble-CRPS metric, histogram
  quantile estimation, zarr loading, ACE-format checkpoint writing) and
  `student_checkpoint.py` (checkpoint export/load round-trip) have **no
  tests at all**.
- `_quantile_from_histogram` and `_crps_ensemble` in
  `best_student_callback.py` are pure-torch and need no fastgen — they are
  the cheapest high-value targets.

## Design

1. **Collection gating.**  Add
   `fme/downscaling/distillation/conftest.py`:

   ```python
   collect_ignore_glob: list[str] = []
   try:
       import fastgen  # noqa: F401
   except ImportError:
       collect_ignore_glob = ["test_*.py"]
   ```

   plus a `requires_fastgen = pytest.mark.skipif(...)` helper exported for
   any future distillation tests living elsewhere.  (Prefer skip-at-
   collection over per-test importorskip: the modules under test import
   fastgen at module scope.)
2. **Dependency-light test split.**  Move pure-logic code that accidentally
   lives in fastgen-importing modules so it can be tested everywhere:
   `_quantile_from_histogram`, `_crps_ensemble` → e.g.
   `fme/downscaling/distillation/metrics.py` (no fastgen import), tested
   unconditionally.  `student_checkpoint.py` already imports no fastgen —
   verify and add tests now:
   - export → `CheckpointModelConfig`-style reload round-trip on
     `_build_small_model` (helper in `test_teacher.py`; per AGENTS.md
     promote it to a shared `testing.py` helper once used from 3+ files);
   - `sampler_type`/`num_diffusion_generation_steps` overrides land in the
     saved config;
   - `load_student_module_into_teacher` strict-load round-trip.
3. **CI job with the extra installed.**  Add a CI variant (or extend the
   distillation Docker image job) that `pip install -e .[distillation]`
   (spec 01) and runs `python -m pytest fme/downscaling/distillation/`.
   This is where `test_teacher.py`/`test_condition.py` actually execute.
   Until that job exists they only run inside the distillation image —
   acceptable, but the job is the deliverable here.
4. **Integration smoke test** (in the fastgen-gated set): the `--dryrun`
   path of the entrypoint on a tiny synthetic teacher
   (`_build_small_model`, CPU): teacher loads, one `{real, condition}`
   batch is produced with the right shapes/channel counts, one forward
   pass runs.  This is the test that would have caught both the σ-clamp
   and the `ACE_C_OUT` bugs — assert specifically:
   - `sample_t(...)` over many draws spans (0.9·σ_max, σ_max] when the
     teacher's σ_max > 80;
   - `config.model.input_shape[0] == len(out_names)`.
5. **Keep FastGen untested by us.**  No tests asserting FastGen internals
   beyond the seams we consume (`sample_t` kwargs forwarding,
   `reset_parameters` schedule rebuild) — those two are already covered in
   `test_teacher.py`; keep them, they pin our assumptions about the pinned
   commit and will fail loudly on an upgrade.

## Acceptance criteria

- `python -m pytest fme/` in a no-fastgen env: distillation tests reported
  as skipped/ignored, zero collection errors.
- In a `[distillation]` env: full distillation test suite passes, including
  new checkpoint-export and metric tests and the CPU dryrun smoke test.
- CI configuration runs the gated suite somewhere on every PR touching
  `fme/downscaling/distillation/` (path filter acceptable).
- No `sys.modules` stubbing anywhere in committed code.

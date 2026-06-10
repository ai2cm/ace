# 09 — Enforce that training sigmas are the inference sigmas

## Goal

Make the sigma list a distilled student is sampled with at inference time
**provably identical** to the sigma list it was trained against, by carrying
the trained sigmas through the exported checkpoint as a single source of
truth — instead of re-deriving them at inference from five loose parameters
(`sigma_min`, `sigma_max`, `rho`, `min/max_step_percent`,
`schedule_num_steps`) that must all coincidentally match the training-time
noise schedule.

## Why this exists (history, diagnosed 2026-06-10)

An f-distill student trained with `STUDENT_STEPS=2` matched the teacher and
target distributions *worse* at 2 inference steps than at 1.  Investigation
showed `fastgen_sampler` (`fme/downscaling/samplers.py:177`) is a faithful
replica of FastGen's `generator_fn` + `_student_sample_loop`
(`FastGen/fastgen/methods/model.py:332-436`) — the loop is not the bug.  The
bug is in the *parameters* it receives at inference:

- During training, the student's t_list comes from
  `AceEDMNoiseSchedule.get_t_list(student_sample_steps)` built with the
  teacher schedule range — for the MoE teacher, the **bundle-level**
  `DenoisingMoEPredictor._sigma_schedule_min/_max` = [0.005, 200]
  (`fastgen_teacher.py` `AceDiffusionTeacher.__init__`).
- The exported student checkpoint instead inherits its config from
  `teacher_model._primary` (`fastgen_train.py:570-574`), i.e. **the lowest-
  sigma expert's own `DiffusionModelConfig`**, whose `sigma_min`/`sigma_max`
  are that expert's sub-range, not the bundle range.
  `save_student_checkpoint` (`student_checkpoint.py`) copies it verbatim and
  also leaves `num_diffusion_generation_steps` at the teacher's value (18)
  because `BestStudentCheckpointCallback` never passes `num_sampling_steps`
  (`best_student_callback.py:237-241`).
- At inference, `DiffusionModel.generate` (`models.py:632-640`) feeds those
  checkpoint values into `fastgen_sampler`, producing a *different* t_list
  than training.

Why the symptom is asymmetric in step count: a 2-step student is trained at
exactly two sigmas (`sigmas[998]≈199.5` and `sigmas[500]≈6.32` for
[0.005, 200], rho=7).  Step 1 is robust to a wrong `sigma_max` — the input
is pure noise and `EDMPrecond`'s `c_in = 1/√(σ²+1)` normalizes the scale
away.  Step 2 is brittle — a shifted range moves the intermediate sigma off
the only low sigma the student ever saw, where it falls back to teacher-like
conditional-mean denoising (over-smoothed, bad tails).  The training-time
validation never caught this because `BestStudentCheckpointCallback` passes
the *correct* range (`student._sigma_min/_sigma_max`,
`best_student_callback.py:471-478`) — only the exported-checkpoint path is
wrong.

## Verified mechanics (do not re-derive)

- Training t_list: `DMD2Model._generate_noise_and_time`
  (`FastGen/fastgen/methods/distribution_matching/dmd2.py:107-116`) draws
  `t_student` via `sample_from_t_list(..., t_list=sample_t_cfg.t_list)`;
  `sample_t_cfg.t_list` defaults to `None`, so the effective list is
  `EDMNoiseSchedule.get_t_list(sample_steps)`
  (`FastGen/fastgen/networks/noise_schedule.py:940-973`): a rho-spaced grid
  of 1000 sigmas over `[min_t, max_t]`, indexed at
  `linspace(998, 2, steps+1).long()`, last entry clamped to 0.  f-distill
  inherits this unchanged.
- `fastgen_sampler` re-implements exactly that grid from its own kwargs and
  hardcoded defaults (`rho=7.0`, `schedule_num_steps=1000`,
  `min_step_percent=0.002`, `max_step_percent=0.998`).  Equality with
  training therefore currently rests on `sigma_min`/`sigma_max` in the
  checkpoint config matching the training schedule — which the MoE export
  path violates.
- `AceDiffusionTeacher` keeps the authoritative training range as
  `self._sigma_min` / `self._sigma_max` and its `noise_scheduler` is the
  very `AceEDMNoiseSchedule` used by the FastGen method; `get_t_list` on it
  *is* the training t_list.
- Backward-compat constraint (AGENTS.md): trained-checkpoint loading for
  inference must keep working; new config fields need safe defaults and
  old checkpoints must load unchanged (see
  `test_models.py::test_diffusion_model_config_backwards_compatible_no_sampler_type`
  for the existing pattern).

## Design

Single source of truth: the training noise schedule object.  Export its
sigmas; never re-derive them at inference.

1. **New config field.** Add `generation_sigmas: list[float] | None = None`
   to `DiffusionModelConfig` (`fme/downscaling/models.py`).  Semantics: when
   `sampler_type == "fastgen"` and `generation_sigmas` is set, the sampler
   uses exactly this list (descending, final entry 0) and
   `num_diffusion_generation_steps` is **derived** as
   `len(generation_sigmas) - 1`.  Validate in `__post_init__`: descending,
   `[-1] == 0`, `len >= 2`, only allowed with `sampler_type="fastgen"`.

2. **Sampler accepts an explicit list.** `fastgen_sampler` gains a
   `t_list: Sequence[float] | None = None` kwarg that bypasses the internal
   grid construction.  The parametric path stays for backward compat with
   old checkpoints and for deliberate step-count overrides.

3. **Export bakes the trained sigmas.** `save_student_checkpoint` gains a
   required `t_list` argument (drop the optional `num_sampling_steps`):

   ```python
   save_student_checkpoint(student_module, teacher, path, t_list=...)
   ```

   It sets `config["generation_sigmas"] = [float(t) for t in t_list]`,
   `config["num_diffusion_generation_steps"] = len(t_list) - 1`, and —
   belt-and-braces for step-count overrides at eval time — overwrites
   `config["sigma_min"]`/`config["sigma_max"]` with the schedule bounds the
   list was built from.  The caller provides the list from the live
   schedule, not from constants.

4. **Callback passes the live schedule's list.**
   `BestStudentCheckpointCallback` computes
   `t_list = student.noise_scheduler.get_t_list(self._student_sample_steps)`
   once, uses it for **both** validation sampling (replacing the
   sigma_min/sigma_max kwargs at `best_student_callback.py:471-478`) and the
   two `save_student_checkpoint` calls.  Validation and the exported
   artifact can then never disagree.

5. **Inference consumes it.** `DiffusionModel.generate` (`models.py:632`)
   passes `t_list=self.config.generation_sigmas` when set.  When a user
   overrides the step count at eval (the 1-step-vs-2-step comparison), they
   set `num_diffusion_generation_steps` and clear `generation_sigmas`; the
   parametric path then reconstructs the grid from the (now corrected)
   `sigma_min`/`sigma_max` — document this in `ARCHITECTURE.md`'s noise-
   schedule section as layer 4.

6. **Out-of-band check.** `evaluator.py`/`predict.py` checkpoint loading:
   when `sampler_type == "fastgen"` and `generation_sigmas` is None, log a
   warning naming the checkpoint as pre-fix and stating the reconstructed
   t_list, so legacy-checkpoint evaluations are auditable.

## Tests

- Round-trip: build a small `AceDiffusionTeacher` (use
  `_build_small_model(...)` from `test_teacher.py`) with a non-default sigma
  range, export via `save_student_checkpoint` with
  `t_list=net.noise_scheduler.get_t_list(2)`, reload with
  `DiffusionModel.from_state`, and assert the sigmas used by
  `fastgen_sampler` equal `get_t_list(2)` exactly (capture via the returned
  `latent_steps` scaling or by passing a recording `randn_like`).
- Equivalence: `fastgen_sampler(t_list=None, sigma_min=a, sigma_max=b,
  num_steps=n)` internal grid == `AceEDMNoiseSchedule(min_t=a,
  max_t=b).get_t_list(n)` for both teacher parameter sets
  ([0.002, 150], [0.005, 200]) and n ∈ {1, 2, 4}.  This pins the parametric
  fallback against upstream FastGen.
- MoE regression: a bundle whose primary expert has a narrower sigma range
  than the bundle schedule → exported checkpoint carries the **bundle**
  sigmas, not the expert's (this is the bug; write it as a failing test
  first per AGENTS.md).
- Backward compat: a state dict without `generation_sigmas` loads, defaults
  to None, and generates via the parametric path
  (mirror `test_diffusion_model_config_backwards_compatible_no_sampler_type`).
- `__post_init__` validation: non-descending list, nonzero last entry, and
  `generation_sigmas` with `sampler_type="heun"` each raise.

## Acceptance criteria

- For a freshly exported MoE student checkpoint, the t_list used by
  `Evaluator` inference is bit-identical to the one
  `BestStudentCheckpointCallback` validated with during training.
- 1-step vs 2-step eval of a 2-step student queries the net at exactly the
  sigmas `{sigmas[998]}` and `{sigmas[998], sigmas[500]}` of the *training*
  schedule, respectively.
- Old (pre-fix) student checkpoints still load and sample; the warning in
  step 6 fires for them.
- New tests pass; `python -m pytest fme/downscaling/ -q` clean.

## Out of scope

- Fixing the intrinsic exposure bias of FastGen's multistep recipe (step-2
  training inputs are renoised *teacher* samples, inference renoises the
  student's own prediction) — method-level, not a sampler/config issue.
- Re-evaluating already-trained students (operational, not code; but note
  existing checkpoints can be repaired post hoc by setting
  `generation_sigmas` in the saved config dict).
- Heun-sampler students (`sampler_type="heun"` is unaffected).

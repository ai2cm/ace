# 08 — Known gaps and hardening backlog

Correctness/robustness items found while building the spike.  None block the
structural specs (01–07); each is independently actionable.  File paths are
current as of 2026-06-10 (pre-refactor names; spec 04 moves some of them).

## 1. DMD2 + MoE teacher discriminator wiring

`fastgen_train.py` §3b wires the DMD2 discriminator from
`teacher.encoder_feature_info()`, which traverses **only the primary
expert's** UNet (`AceDiffusionTeacher._get_songunet`).  For an MoE teacher
the discriminator therefore sees features from one expert while `real`
targets come from full sigma-dispatch sampling.  `run.sh` documents
"DMD2 + MoE requires additional discriminator wiring".  Task: either
implement per-expert feature capture (hook the dispatch module's active
expert) or make `--method dmd2` + MoE teacher a hard startup error instead
of a comment.  Recommend the hard error first; it is one `isinstance` check
in the discriminator-wiring function (justify per AGENTS.md or use a
capability flag on the teacher).

## 2. Student initialization from primary expert only

For MoE teachers the student deepcopy carries only the primary expert's
weights (`AceDiffusionTeacher.__deepcopy__` nulls `_moe_experts`
intentionally, to avoid duplicating all experts into student + frozen-copy).
The student therefore starts from a model that was trained on a *sub-range*
of sigma.  This is a deliberate spike tradeoff, not a bug — but it is
undocumented at the call site where it matters
(`fastgen_train.py` §4) and unvalidated.  Task: log it loudly at startup
for MoE teachers; consider an experiment initializing from the
highest-sigma expert instead (the student's first denoising step operates
at σ_max).

## 3. Per-expert noise distributions are unchecked

Spec 02 derives the training-noise distribution from
`model._primary.config`.  A bundle whose experts were trained with
*different* distributions (plausible: per-sigma-range experts) would be
mis-derived.  Task: in `DenoisingMoEBundledConfig.build()` or the spec-02
auto-config, assert all experts share a distribution family, or define the
union behavior and document it.

## 4. `neg_condition` is a zeros tensor

`AceConditionBuilder.build_fastgen_batch` (`fastgen_loader.py:98`) emits
`"neg_condition": torch.zeros_like(condition)` to satisfy FastGen's CFG
plumbing; `guidance_scale` is None in all spike configs so it should be
inert.  Task: verify FastGen never consumes `neg_condition` when
`guidance_scale is None` (check `fastgen/methods/model.py` and method
forward paths), then either drop the key or add a comment stating why a
dummy is required.  If any method does CFG arithmetic with it
unconditionally, a zeros "negative" condition is silently wrong.

## 5. Validation callback ensemble sizes / rank behavior

`BestStudentCheckpointCallback` computes ensemble CRPS between student
draws and the pre-saved teacher zarr (`best_student_callback.py`,
docstring at top).  Unreviewed corners: how many student samples it draws
per val timestep (cost scales with it), whether scoring runs on all DDP
ranks redundantly or rank-0 only, and whether `best_student_tail.ckpt`
selection (histogram-quantile metric) uses enough samples for a stable
99.9999th percentile estimate at val size.  Task: read the callback
end-to-end, document the answers in ARCHITECTURE.md, and add the missing
rank-0 guard if scoring is duplicated.

## 6. Full-dataset-per-rank training data

`fastgen_train.py` §5 builds data with
`Distributed(force_non_distributed=True)`: every DDP rank iterates the same
shuffled stream, so ranks see overlapping (potentially identical, if seeded
alike) batches; only FastGen's gradient sync makes this "data parallel".
Check whether per-rank shuffling diverges (ACE shuffle seeding under
forkserver) — if all ranks produce identical batch order, effective batch
diversity is 1/world_size of nominal.  Task: verify seeds differ per rank;
if not, seed the generator with the rank; then document.

## 7. `student_sample_steps` vs exported checkpoint steps

`save_student_checkpoint(num_sampling_steps=...)` bakes a step count into
the exported config, and `BestStudentCheckpointCallback` receives
`student_sample_steps` from the FastGen config.  There is no single source
of truth tying "what the student was distilled for" to "what inference will
run".  Task: thread `student_sample_steps` from the (spec 03) config into
both call sites and assert equality at export.

## 8. FastGen upgrade procedure

We pin `123e6a2`.  The seams that can break on upgrade are pinned by tests
(`sample_t` kwarg forwarding, `reset_parameters` schedule rebuild,
`Discriminator_EDM` constructor signature, `Trainer.callbacks._callbacks`
dict injection, `instantiate` pass-through of non-DictConfig objects —
the last two are private-API touches in `fastgen_train.py` §9 and
`AceInfiniteDataLoader` docstring).  Task: collect these into a single
`test_fastgen_compat.py` so an upgrade PR gets one failing file as a
checklist, and note the two private-API usages as upstream-PR candidates
(a public callback-registration hook; a documented external-dataloader
contract).

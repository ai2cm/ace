# 04 — Decompose `fastgen_train.py`; contain the global runtime patches

## Goal

`fastgen_train.py` (606 lines) is a monolithic script mixing arg parsing,
global monkey-patches, teacher loading, discriminator wiring, data pipeline
construction, gradient-accumulation arithmetic, callback registration, and
the train loop invocation.  Split it into reviewable, individually testable
modules, and make every global patch explicit, narrow, and documented.

## Current structure (verified — section numbers are in-file comments)

| Section | What it does | Where it should live |
|---|---|---|
| module top | `_copy_ace_teacher` deepcopy factory (must stay module-level for DDP pickling), `_CheckpointPruner` (keeps newest 20 `.pth`), `AceInfiniteDataLoader`, `_load_data_config` | `runtime.py` / `data.py` |
| `main()` preamble | forkserver start method; **global `torch.load` patch forcing `weights_only=False`**; **patch of `fastgen.callbacks.wandb.to_wandb`** replicating channel 0→RGB | `patches.py` |
| §1 | import spike config by dotted path, apply `- key=value` overrides | replaced by spec 03 |
| §2 | FastGen DDP init (`fastgen.utils.distributed.ddp.init`) | `runtime.py` |
| §3 | teacher load: `CheckpointModelConfig` (with hardcoded `fine_coordinates_path=/climate-default/2026-02-23-...3km.zarr` fallback!) or `DenoisingMoEBundledConfig`; wrap in `AceDiffusionTeacher` | `teacher.py` |
| §3b | DMD2 discriminator auto-wiring from `teacher.encoder_feature_info()` (deepest encoder level) | `teacher.py` |
| §4 | inject `L(_copy_ace_teacher)` as `config.model.net`; clear `pretrained_model_path`, `use_ema`; strip `ema*` callbacks | `build.py` |
| §5 | build `GriddedData` with `Distributed(force_non_distributed=True)`; compute coarse patch extent from `input_shape // downscale_factor`; patching enabled only when domain strictly larger | `data.py` |
| §6 | recompute `grad_accum_rounds` from real per-GPU batch size × world size vs `batch_size_global` | `build.py` |
| §7 | checkpointer save_dir, S3 creds, config snapshot | `runtime.py` |
| §8 | `--dryrun`: one batch + teacher sample, exit | keep on CLI |
| §9 | instantiate model, register `_CheckpointPruner` + `BestStudentCheckpointCallback` (val zarr path, coarse val data, teacher's primary `DiffusionModel`, best/best-tail ckpt paths), `Trainer.run` | `build.py` |

## Specific issues to fix while splitting

1. **Global `torch.load` patch.**  Reason it exists: FastGen loads ACE
   checkpoints with `weights_only=True` but ACE checkpoints contain numpy
   scalars.  Replace the blanket patch with
   `torch.serialization.add_safe_globals([...])` for the specific numpy
   types (preferred, torch ≥2.4), or a `contextlib.contextmanager` wrapping
   only FastGen's checkpoint-load call sites.  If neither is feasible
   (FastGen calls `torch.load` in many places — enumerate them first:
   `grep -rn "torch.load" FastGen/fastgen/ | grep -v test`), keep the patch
   but move it into `patches.py` with the inventory of covered call sites
   in the docstring so the next person can retire it.
2. **`to_wandb` patch** — same treatment: isolate in `patches.py` with a
   regression test (a `[B, 5, H, W]` tensor logs without assertion).
3. **Hardcoded `fine_coordinates_path`** in §3 (a weka path) — must come
   from config (spec 03's `TeacherConfig`), defaulting to None and relying
   on the checkpoint's embedded coords (newer checkpoints carry them; the
   path is only a fallback for the oldest format).
4. **`AceInfiniteDataLoader.sampler_start_idx` is accepted-and-ignored** —
   resumption restarts the data stream.  Document this on the class (it is
   intentional: the stream is shuffled and stateless), don't fix.
5. `main()` currently mixes `Distributed.context()` (ACE) with FastGen DDP
   init — keep the exact ordering (forkserver → parse → FastGen imports →
   DDP init → CUDA work); it is load-bearing for zarr worker hangs (see
   comment at `fastgen_train.py:288`).

## Target layout

```
fme/downscaling/distillation/
    train.py        # CLI: parse YAML (spec 03), orchestrate; <100 lines
    patches.py      # the two runtime patches, each with docstring + test
    teacher.py      # load_teacher(cfg) -> (model, AceDiffusionTeacher, requirements)
                    # wire_discriminator(config, teacher)
    data.py         # AceInfiniteDataLoader, build_train_data, patch-extent logic
    build.py        # build_fastgen_config + trainer/callback assembly
    fastgen_teacher.py / fastgen_loader.py / best_student_callback.py  # unchanged
```

Respect import-layering: none of these may import from `fme.ace` or other
non-`core` siblings (AGENTS.md).  fastgen imports stay deferred or guarded
per spec 01.

## Tests

Most sections become unit-testable without GPU/fastgen-training:

- patch inventory test: numpy-containing checkpoint loads through the
  patched/safe-globals path.
- patch-extent logic: domain ≤ patch in either dim → `None` (the
  drop-partial-patches footgun documented at `fastgen_train.py:474-479`).
- grad-accum arithmetic: non-divisible `batch_size_global` rounds down and
  logs.
- discriminator wiring on `_build_small_model`: `feature_indices`,
  `all_res`, `in_channels` match `encoder_feature_info()`'s deepest level.

## Acceptance criteria

- `train.py` orchestration readable top-to-bottom; no function >80 lines.
- Both monkey-patches live in `patches.py`, applied from exactly one place,
  each with a test and a written retirement condition.
- A `--dryrun` against a small synthetic teacher works on CPU (this nearly
  works today; the dryrun path only needs teacher.sample + one batch).
- Behavior-identical training launch (compare resolved FastGen config
  snapshot `config.yaml` before/after refactor on the same inputs).

# Multi-step FT of paper-final var-masking runs — handoff

Status date: 2026-08-17. Branch: `alexey8-mstepft` (off `exp/alexey8`), pushed,
HEAD `f47edc364`. This doc is written for whoever (human or Claude agent) picks
this up next. For the full technical spec of the configs see [FINETUNE.md](FINETUNE.md);
this doc is the "what we did + current state + what to do next".

## TL;DR

Multi-step fine-tuning of the four paper-final v5 (1°, 6-hourly) var-masking
checkpoints — one per (global-mean-removal × masking) cell. Each FT config **is
that run's exact 1-step pre-training config** with only five deliberate changes
(below). All four run on **ai2/titan (4× B200)** — they must (H100 OOMs). Two are
running; two were just resubmitted to fresh nodes after node-level I/O stalls
(see Known issue). Nothing is blocked on you; the open items are follow-ups.

## The four runs

| Cell (paper label) | Source pre-training run | Source ckpt dataset | FT experiment (titan) |
| --- | --- | --- | --- |
| gmroff-mask0 · *No mask* | `...-gmroff-mask0-seed1-v5` | `01KZEZNEAASGS4JAJKSQB192GF` | [01M094PKN5RV7FB7ZWF2J0YW88](https://beaker.org/ex/01M094PKN5RV7FB7ZWF2J0YW88) |
| gmroff-mask20 · *Mask 20* | `...-gmroff-mask20-seed1-v5` | `01KZEFBKGFJ2V38N8V9HNAVZ27` | [01M094PV2AN054F6TNJTQ0CWJK](https://beaker.org/ex/01M094PV2AN054F6TNJTQ0CWJK) |
| gmron-mask0 · *No mask, GMR* | `...-gmron-mask0-seed2-v5` **(interim)** | `01KYWXQH0XT66EHXVRJ5N9HC4S` | [01M05WN2KPK0S8E012CE2115ZM](https://beaker.org/ex/01M05WN2KPK0S8E012CE2115ZM) |
| gmron-mask20 · *Mask 20, GMR* | `...-gmron-mask20-seed0-v5` | `01KYT8YZZZGKGJFFK6TNJ64SFN` | [01M05WN8N9JC4NGYS3H7XNRN7S](https://beaker.org/ex/01M05WN8N9JC4NGYS3H7XNRN7S) |

wandb project **VarMasking8**, group `ace2-var-masking-mstepft-2026-06-30`. FT
run names = source run name + `-mstepft`.

## Job status (2026-08-17)

- **gmroff-mask0-seed1** — freshly resubmitted (queued), pruned-inference config,
  new experiment `01M094PK…`. See node-stall note below for why.
- **gmroff-mask20-seed1** — freshly resubmitted (queued), pruned-inference config,
  new experiment `01M094PV…`.
- **gmron-mask20-seed0** — RUNNING, healthy (good node, no stalls). Left on its
  older commit (heavier inline inference; harmless, eval-only) to avoid rerolling
  a healthy node.
- **gmron-mask0-seed2** — RUNNING (auto-retry resumed from checkpoint). Trains
  ~7–14h per attempt then dies to preemption / transient NCCL timeout (different
  each time → not a deterministic bug). **Do not manually resubmit** — a fresh
  submit writes a new `/results` and restarts FT at epoch 0; the auto-retry
  resumes from checkpoint. Only intervene if it stops retrying or the *same* NCCL
  timeout recurs at a *consistent* step.

### Known issue: node-level I/O stalls on titan
Some titan nodes freeze for **hours** mid-run — multi-hour gaps between
consecutive log lines, hitting *both* training and inference (not
inference-specific). The process resumes afterward (so it is not a NCCL crash,
which would abort at 30 min; it is an I/O/hardware stall). Two of the original
titan jobs spent ~90% of wall-clock frozen while a third ran clean, so it is
**node-specific**, not a config/code bug. **Symptom:** in `beaker job logs`,
long timestamp gaps between "Step N" or "processing output from window N" lines.
**Fix:** cancel + resubmit that experiment to reroll onto a different node (a
healthy node does an epoch + inference in hours). That is why gmroff-mask0-seed1
and gmroff-mask20-seed1 were resubmitted above.

## What we did, and why

Goal: take the four paper-final var-masking pre-trained checkpoints and continue
training with a multi-step rollout (for rollout stability), keeping everything
else identical to pre-training so the FT is comparable.

Design decisions (each is a commit; see history below):

1. **FT config = exact pre-training config + 4 changes.** The pre-training
   `config.yaml` is pulled from each checkpoint's Beaker dataset and cached under
   `pretrain_source_configs/`. `generate_finetune_configs.py` reads it and changes
   only:
   - `stepper_training.n_forward_steps`: `1` → probability schedule
     `{1:.6, 2:.2, 4:.1, 12:.05, 20:.05}` (the **only** thing borrowed from the
     ERA5 baseline multi-step FT config).
   - `stepper_training.parameter_init.weights_path` — load the pre-trained weights.
   - `max_epochs`: `150` → `20` (FT is short).
   - `stepper.step.config.input_dropout_optimized_steps_only: true` (see #2).
   - **Inline inference pruned** (`INLINE_INFERENCE_DROP`): the weight-0.0
     multi-year diagnostics (`10year`, `10year_insample`, `long_46year`) are
     dropped from inline inference — they cost thousands of windows per
     inference-epoch and dominated FT wall-clock. Kept `aimip_checkpoint`
     (weight 1.0, drives checkpoint selection) + cheap `weather`. Trained weights
     unaffected; run the dropped climate diagnostics in the post-FT eval pass.

   Everything else — 1979–2008 training windows, the retained inference entries,
   FusedAdam, EnsembleLoss (crps .9 / energy .1, no h500 weight),
   `optimize_last_step_only`, GMR, masking level, architecture — is pre-training
   verbatim. (An earlier version derived the whole config from the ERA5 baseline
   FR config; that broke the inference suite — a single 1996-IC entry, no
   aimip/long_46year. Don't do that.)

2. **New fme/core flag `input_dropout_optimized_steps_only`** (default False).
   With `optimize_last_step_only`, the old behavior masked *every* rollout step
   including the non-optimized `no_grad` ones — perturbing the trajectory that
   feeds the optimized step, while inference runs unmasked (train/inference
   mismatch). The flag skips the dropout draw when gradients are off, so masking
   hits only the optimized step. Gated on `torch.is_grad_enabled()`. No-op for the
   mask0 cells. In [../../../fme/core/step/single_module.py](../../../fme/core/step/single_module.py)
   with a test in `fme/core/step/test_step.py`.

3. **GPU footprint per-cluster.** `submit_finetune_jobs.py` sets titan(B200)→4
   GPUs, jupiter(H100)→8, and refuses mixed-cluster submits (a job requests a
   fixed GPU count). `batch_size: 8` is the *global* batch (local = batch//world),
   so 4 vs 8 GPUs trains identically.

4. **Must run on titan/B200.** The 20-step rollout of the embed_dim-512 model
   (channel-mask inputs double the input channels, + GMR) is a genuine ~80 GiB
   shortfall on H100 — it OOMed on jupiter even with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (which is set, overridably,
   in `run-ace-train.sh`). B200's 192 GiB fits it comfortably (local batch 2 ≈
   145 GiB). Escalation if ever needed: gradient checkpointing
   `stepper.step.config.builder.config.checkpointing` (1=encoder/decoder …
   3=per-block; reproducibility-safe, ~20–30% slower).

## How to operate (commands)

```bash
git checkout alexey8-mstepft
cd configs/experiments/2026-06-30-var-masking
conda activate fme   # gantry + validate_config live here

# Regenerate the four run_configs/*-mstepft.yaml (idempotent):
python generate_finetune_configs.py

# Validate one:
python -m fme.ace.validate_config --config_type train \
  run_configs/ace-train-config-4deg-nc-sfno-era5-gmroff-mask0-seed1-v5-mstepft.yaml

# Submit — MUST be titan (H100 OOMs). One cluster at a time:
python submit_finetune_jobs.py --beaker-cluster ai2/titan \
  --beaker-priority high --beaker-workspace ai2/ace
# NOTE: gantry is slow (~20s/job); a 4-job submit can exceed a 2-min shell
# timeout. If it does, only some land — check which, then submit the rest via
# run-ace-train.sh directly (see submit_finetune_jobs.py for the env it sets).

# Check job status (replace IDs with current experiments):
python - <<'PY'
import subprocess, json
for name, ex in {
 "gmroff-mask0-seed1":"01M05WHT63Y6GH23C0BS9QTWF2",
 "gmroff-mask20-seed1":"01M05WMWQ0JYXPQY5B1KVQA0FQ",
 "gmron-mask0-seed2":"01M05WN2KPK0S8E012CE2115ZM",
 "gmron-mask20-seed0":"01M05WN8N9JC4NGYS3H7XNRN7S",
}.items():
    d=json.loads(subprocess.run(["beaker","experiment","get",ex,"--format","json"],
        capture_output=True,text=True).stdout)[0]
    st=d["jobs"][-1]["status"] if d["jobs"] else {}
    state=("RUNNING" if st.get("started") and not st.get("finalized")
           else f"exit={st.get('exitCode')}" if st.get("finalized") else "queued")
    print(f"{name:20s} attempts={len(d['jobs'])} {state}")
PY
```

Evaluate after training with the existing eval tooling in this dir
(`update_beaker_map.py` → `generate_eval_configs.py -v v5` → `submit_eval_jobs.py`;
see the top-level project notes). Plot `-bestinf`.

## Open follow-ups (next actions, none blocking)

1. **Swap gmron-mask0 to seed1.** It currently uses **seed2** as an interim — the
   paper's intended run is `gmron-mask0-seed1`, which had no succeeded checkpoint
   when these were generated. When it finishes: update `SELECTED_SOURCES` in
   `generate_finetune_configs.py`, add its run→dataset entry to
   `wandb_to_beaker_map.json` (via `update_beaker_map.py`), drop its `config.yaml`
   into `pretrain_source_configs/`, regenerate, resubmit that one.
2. **PR `input_dropout_optimized_steps_only` to main.** It's a clean, tested,
   self-contained fme/core commit (`0ddbbacf1`) that's generally useful; currently
   only on this branch. Cherry-pick into its own PR.
3. **Evaluate the FT checkpoints** once training completes (eval tooling above).
4. **main merge was intentionally deferred.** `exp/alexey8` already has its own
   mature distributed-shutdown implementation; main's `#1398`/`#1425` are an
   independent (newer, NCCL-abort-on-listener-thread) implementation of the same
   thing. A full merge is ~35 conflict hunks reconciling the two teardown impls —
   not additive. If picked up, decide which `shutdown.py` wins (likely needs
   Alexey) and validate with the distributed test suite. Branch is a generation
   behind main only for the "rank wedged in a collective at SIGTERM" case.

## File map (this dir)

- `FINETUNE.md` — technical spec of the FT configs (read first for details).
- `generate_finetune_configs.py` — builds the 4 FT run configs from the cached
  pre-training configs + the checkpoint map.
- `submit_finetune_jobs.py` — submits them via `run-ace-train.sh` (per-cluster GPU
  count, refuses mixed clusters).
- `run-ace-train.sh` — shared train launcher; sets `PYTORCH_CUDA_ALLOC_CONF`.
- `pretrain_source_configs/*.yaml` — the 4 exact pre-training configs (source of truth).
- `run_configs/*-mstepft.yaml` — the 4 generated FT configs.
- `wandb_to_beaker_map.json` — run name → checkpoint dataset ID.

## Commit history (branch, newest first)

```
001a58702 Prune heavy inline inference from var-masking multi-step FT
f47edc364 Set expandable_segments to avoid var-masking multi-step FT OOM
5251e5f1b Enable input_dropout_optimized_steps_only in var-masking multi-step FT
0ddbbacf1 Add input_dropout_optimized_steps_only to single-module step   <- PR to main
6bf618d04 Cap var-masking multi-step FT at 20 epochs
2589678e9 Base var-masking multi-step FT on exact pre-training configs
8c55cd484 Make var-masking multi-step FT GPU footprint per-cluster
733189695 Add multi-step fine-tuning configs for paper-final var-masking runs
```
(HANDOFF.md itself is committed on top of these.)

## Notes for a Claude agent

- Repo conventions live in `AGENTS.md` (root): branch naming, commit style (no
  Co-Authored-By), validate in `__post_init__`, `fme/core` may not import
  `fme.ace`, add a failing test first for bugs, use `pre-commit` (ruff/mypy) not
  raw ruff. Run tests with `python -m pytest` inside the `fme` conda env.
- Beaker experiment IDs above are the *current* runs; re-query rather than trust
  them if time has passed (Beaker auto-retries create new jobs within the same
  experiment; a manual resubmit creates a new experiment and loses checkpoint
  progress).
- "exit=1" on a long-running job here is usually preemption or a transient NCCL
  timeout, not a code bug — check runtime duration before assuming failure.

<!--
Phase-2 experiment write-up (WORKFLOW.md). PLANNED — two native training runs
(1-step + 4-step) that, with the existing 2-step i26sidsm, form a 1/2/4 NFE sweep.
Hypothesis + design + config filled before launch; metrics + verdict after.
This file is shared by the 1-step run and the 4-step run; add each run's wandb id
to the table once launched.
-->
# Experiment — f-distill step-count sweep (native 1-step / 2-step / 4-step)

_Hypothesis: sweeping `student_sample_steps` (NFE) trades cost for quality. The current
winner `i26sidsm` is **2-step**; train **native 1-step** and **native 4-step** students
with the same spectral config and see where the spectrum/tail quality vs NFE knee is.
Both are trained **from scratch** (no warm-start — training is short, and a native run at
each step count is the fair comparison; see below). `i26sidsm` (2-step) is the baseline._

## Why native runs (not warm-start / not a 1-step *eval* of the 2-step model)

`student_sample_steps` is **not** just an inference knob — f-distill training depends on
it (`dmd2.py:_generate_noise_and_time:97–116`):

- **1-step:** `t_student` pinned to `sigma_max`, `input_student` = pure noise → trained
  *only* as a one-shot noise→x0 map.
- **N-step:** `t_student` drawn from the discrete N-node `t_list`; interior nodes get
  `input_student = forward_process(real_data, eps, t_student)` — **real data re-noised
  (teacher forcing)**.

Each node's output is independently pushed to the teacher x0 distribution (VSD); a single
net forward per step at train time. Consequences:

1. It's one net conditioned on `t`, so a 2-step model splits capacity across
   `sigma_max`+`σ_mid`, while a **native-1-step model specializes** entirely on
   `sigma_max→x0`. So a 1-step eval of the 2-step model is only a *lower bound* on
   native-1-step — hence we train 1-step natively rather than just eval the 2-step model
   at 1 step.
2. **Exposure bias grows with steps:** interior nodes train on *real*-renoised inputs but
   at inference receive the *student's own* upstream x0-estimate renoised — so 4-step is
   not guaranteed to beat 2-step; measure it.

(See [[fdistill-step-coupling]] in memory.)

## Design

| arm | steps (NFE) | init | role | task |
|---|---|---|---|---|
| **1step** | 1 | from teacher (scratch) | fewer-NFE candidate | #3 |
| `i26sidsm` (existing) | 2 | — | **baseline** | — |
| **4step** | 4 | from teacher (scratch) | more-NFE candidate | #2 |

- **Launch:**
  ```
  conda run -n fme bash configs/experiments/2026-05-18-distillation-with-val/run.sh \
      fdistill --suffix 1step --spectral-weight 1e-2 --student-steps 1
  conda run -n fme bash configs/experiments/2026-05-18-distillation-with-val/run.sh \
      fdistill --suffix 4step --spectral-weight 1e-2 --student-steps 4
  ```
- Everything else matches `i26sidsm` (W=1e-2, `band_gamma=0`, gan=1e-3, single-model
  PRATEsfc teacher); only `--student-steps` differs.
- **Note (1-step + spectral):** `fdistill_kl_spike.py` comments that `STUDENT_STEPS=1`
  historically paired with a higher GAN weight / different LR (`:102–112`). Keep gan=1e-3
  for a clean step-only delta first; revisit if the 1-step run is GAN-unstable.

## Runs

| arm | wandb | beaker | state |
|---|---|---|---|
| 1step | `TBD` | `TBD` | not launched |
| 4step | `TBD` | `TBD` | not launched |

## Decision criteria

- Compare 1 / 2 / 4-step at best-sustained (or spec-13-selected) checkpoints on
  `spec_mae_{lo,mid,hi}`, `crps`, tails — plot quality vs NFE and find the knee.
- If 1-step is close to 2-step → prefer 1-step (half the NFE). If 4-step clearly beats
  2-step and exposure bias doesn't bite → 4-step may be worth it for offline/high-fidelity
  use. NFE is the product cost, so the bar for adding steps is a clear quality gain.

## Result  <!-- filled after runs -->

_Pending._

## Verdict  <!-- HUMAN: fill this in -->

- **Quality-vs-NFE knee (1 / 2 / 4-step):** TODO.
- **Recommended default step count:** TODO.
- **Next:** TODO.

## Caveats

- ⚠️ _Prepend here if a later run invalidates this one._

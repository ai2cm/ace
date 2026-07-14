<!--
Phase-2 experiment write-up (WORKFLOW.md). PLANNED training run + a cheap eval
probe. Hypothesis + design + config filled before launch; metrics + verdict after.
Rename TBD -> wandb id once launched.
-->
# Experiment — 4-step f-distill student, warm-started from the 2-step spectral winner

_Hypothesis: a **4-step** f-distill student (2× the NFE of the current 2-step winner
`i26sidsm`) buys enough quality — especially fine-scale spectrum + tails — to justify
the extra two network evals. **Warm-start** from `i26sidsm`'s `best_student_tail.ckpt`
(identical net architecture) and use `i26sidsm` (2-step) as the **baseline**. Plus a
cheap no-train probe: **1-step eval of the 2-step `i26sidsm`** to measure how much the
2nd step actually contributes._

## How step count couples to training (the mechanism — verified in FastGen)

`student_sample_steps` (`ACE_STUDENT_STEPS`, default **2**; `fdistill_kl_spike.py:36`)
is **not** just an inference knob — it changes the training input distribution
(`dmd2.py:_generate_noise_and_time`, `:97–116`):

- **1-step:** `t_student` is pinned to `sigma_max`; `input_student` is **pure noise**.
  The net is trained *only* as a one-shot noise→x0 map.
- **N-step (N>1):** `t_student` is drawn from the **discrete N-node t_list**
  (`sample_from_t_list`, `get_t_list(N)`); for every node `input_student =
  forward_process(real_data, eps, t_student)` — i.e. **real data re-noised to that σ
  (teacher forcing)**. Interior nodes are trained to denoise partially-noised *real*
  data → x0.

The objective *form* is step-independent (student emits x0 at `t_student` → re-noise to
a random `t` from `sample_t` → VSD between teacher-x0 and fake-score-x0;
`f_distill.py:138–166`), and each node's output is independently pushed toward the
teacher's x0 distribution. But **the input distribution it is trained over is
step-dependent**, so training is **not** independent of step choice.

**Three consequences for this experiment:**

1. **A native-1-step model can genuinely beat a 1-step *eval* of a 2-step model.** It's
   one net conditioned on `t`: a 2-step model must be a competent denoiser at *both*
   `sigma_max` and `σ_mid`, whereas a 1-step model spends all capacity on the single
   `sigma_max→x0` map. So the 1-step probe below is a **lower bound** on native-1-step
   quality, not a fair "is the 2nd step worth it" test.
2. **The 1-step probe is still a valid cheap check** — the 2-step model's top node was
   trained on ~pure-noise@`sigma_max` input (the `200·ε`-dominated forward process ≈
   noise) and its output is a VSD-valid x0 — so it tells you the *2nd step's marginal
   contribution for this specific model*.
3. **Exposure bias grows with steps.** Interior nodes are trained on *real*-data-renoised
   inputs but at inference receive the *student's own* upstream x0-estimate renoised
   (bootstrap). More steps ⇒ more interior nodes exposed to this train/inference
   mismatch — a concrete reason 4-step may **not** beat 2-step and must be measured.

## Warm-start feasibility

The student net architecture == the teacher's (FastGen inits the student as a deepcopy
of the teacher; `fastgen_train.py:677` clears `config.model.pretrained_model_path`).
`i26sidsm`'s student is itself that same architecture, so its weights load cleanly as a
4-step init. **Plumbing needed** (sub-task 1): the entrypoint currently forces the
deepcopy-of-teacher init — add an optional `--init-student-checkpoint` that loads an ACE
student ckpt's weights into the built student net (or set `pretrained_model_path`).

## Design

| arm | steps | init | role |
|---|---|---|---|
| **4step-warmstart** (main) | 4 | `i26sidsm` best_student_tail.ckpt | the candidate |
| `i26sidsm` (existing) | 2 | (deepcopy of teacher) | **baseline** |
| **1-step probe** (no train) | eval `i26sidsm` at 1 step | — | measures the 2nd step's contribution |

- **Launch (main):**
  ```
  conda run -n fme bash configs/experiments/2026-05-18-distillation-with-val/run.sh \
      fdistill --suffix 4step-warmstart \
      --spectral-weight 1e-2 --student-steps 4 \
      --init-student-checkpoint <i26sidsm best_student_tail.ckpt>   # ← needs sub-task 1
  ```
- **1-step probe:** re-run the `i26sidsm` checkpoint through the student sampler with
  `student_sample_steps=1` (a `boundary_aligned`/`get_t_list(1)=[sigma_max,0]` eval) —
  no training. Compare to its 2-step eval.
- Everything else matches `i26sidsm` (W=1e-2, `band_gamma=0`, gan=1e-3, single-model
  PRATEsfc teacher) so the only variable is the step count.

## Decision criteria

- **4-step vs 2-step (`i26sidsm`)** at best-sustained / spec-13-selected checkpoints on
  `spec_mae_{lo,mid,hi}`, `crps`, tails: does 4-step meaningfully improve spectrum/tails?
  If the gain is marginal, **2-step stays the default** (NFE is the product cost).
- **1-step probe:** if 1-step eval of the 2-step model is already close to its 2-step
  eval, the 2nd step adds little → a **native-1-step** run becomes the interesting
  follow-up (fewer NFE). If the 2nd step matters a lot, more steps (4) are worth testing.

## Result  <!-- filled after runs -->

_Pending._

## Verdict  <!-- HUMAN: fill this in -->

- **Is 4-step worth the extra NFE vs 2-step?** TODO.
- **Does the 2nd step contribute much (1-step probe)?** TODO.
- **Next:** TODO (native-1-step run? more steps? stay at 2-step?).

## Caveats

- Warm-start biases the 4-step model toward the 2-step trajectory; if results are
  ambiguous, a from-teacher-init 4-step control disambiguates.
- ⚠️ _Prepend here if a later run invalidates this one._

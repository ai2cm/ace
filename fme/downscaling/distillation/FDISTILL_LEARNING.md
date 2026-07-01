# How FastGen f-distill Works (learning notes)

A conceptual reference for the f-distill distillation used in this repo — how the
few-step student is trained, what each network does, and the identities behind it.
Companion to `MOE_DISTILLATION_STATUS.md` (which tracks the actual runs/experiments
and project-specific findings). Code references are `file:line` into `FastGen/` and
`fme/downscaling/distillation/`.

> One-line summary: f-distill trains a **few-step generator** so that its **clean
> output distribution** matches a frozen many-step teacher's, by re-noising the
> student's outputs to all noise levels and matching the teacher's score against a
> learned estimate of the student's own score (Variational Score Distillation),
> reweighted to a **forward-KL** for mode/tail coverage.

---

## 1. The networks (and what survives to inference)

| Network | What it is | Trained? | Kept at inference? |
|---|---|---|---|
| **student** (`net`) | few-step generator, σ_max→x0 (+ intermediate nodes) | yes (the goal) | **yes — the only one** |
| **teacher** | frozen, full many-step diffusion model; the fixed target score; also the critic's feature source | no (frozen) | no |
| **fake_score** | a full diffusion model continually fit to the *student's current* output samples | yes (its own step) | no |
| **discriminator** | small projected GAN on teacher **encoder** features | yes (its own step) | no |

Only the student is exported. The teacher, fake_score, and discriminator are
training scaffolding.

---

## 2. How student training differs from teacher training

**Teacher (standard EDM/diffusion).** Sample clean `x0` from data, draw σ, perturb
`x_t = x0 + σ·ε`, regress the net to predict `x0` from `(x_t, σ)` — per-sample,
per-σ **MSE against a known clean target** (denoising score matching). The teacher
learns the score at every noise level by denoising real data.

**Student (f-distill / DMD-family).** The student is a **generator**, not a
denoiser-of-given-data. There is **no per-sample clean regression target**.
Instead the loss is **distribution matching**: make the student's clean-output
distribution equal the teacher's. The mechanism re-noises the student's *own*
output to a random level and compares scores (Sections 5–7). The random noising is
how the divergence is sampled across noise levels — it is applied to the *student's
output*, not to ground-truth data.

---

## 3. Student sampling (inference): predict-x0-then-renoise

`generator_fn` (`model.py:391`) → `_student_sample_loop`, driven by an explicit
`t_list = [σ_max, …, 0]`:

1. Start at pure noise scaled to σ_max: `latents = noise · σ_max`.
2. **Predict clean x0 directly** at the current σ (a big denoising jump).
3. **Re-noise** that x0 estimate down to the *next* σ in `t_list`; predict x0 again.
4. Final node is σ=0 → the clean output.

- 1 step: σ_max → x0.
- 2 steps: σ_max → x0 → renoise to σ_mid → x0.

This is a **restart / consistency-style** sampler, NOT the teacher's many-small-step
Heun ODE integration. The student compresses the teacher's whole trajectory into
1–2 big jumps. The `t_list` nodes are the **specific** σ values the student is built
around (and the per-expert bundle places nodes at the σ boundaries, e.g.
2000→200→0).

---

## 4. The training loop (one generator step)

`_student_update_step` (`f_distill.py:113-182`):

```
gen_data       = net(input_student, t_student)              # 1. student generates clean x0   (f_distill.py:138)
perturbed_data = forward_process(gen_data, eps, t)          # 2. LOOP-BACK: re-noise to random t (:139)
fake_score_x0  = fake_score(perturbed_data, t)   # no_grad  # 3a. student's own score          (:142)
teacher_x0, fake_feat = teacher(perturbed_data, t, feature_indices=…)  # 3b. teacher score + critic feats (:146)
fake_logits    = discriminator(fake_feat)                   # 3c. GAN logits                    (:153)
h              = f_div_weighting(fake_logits, t)            # 4. f-div reweight (Section 8)     (:163)
f_distill_loss = VSD(gen_data, teacher_x0, fake_score_x0, h)# 5. score-difference loss          (:166)
loss           = f_distill_loss + 1e-3 * gan_loss_gen       # 6.                                (:169)
```

The "loop-back" (step 2) is the crux: the **generator's own clean output is
re-noised and fed back** into the teacher and fake_score nets. `input_student`
(`dmd2.py:96-116`) is pure noise·σ_max for a 1-step student, or *real data noised to
a node* for multi-step (teacher-forced at nodes — see Section 9).

**The other two networks update on their own steps** (`dmd2.py:319`,
`_fake_score_discriminator_update_step`): the generator step holds fake_score and
the discriminator fixed; they alternate. This is the two-time-scale DMD structure.

---

## 5. Two different `t`'s (a common confusion)

- **Generation nodes** (`t_student`, and inference `t_list`): **specific** discrete
  σ the student is built around (σ_max, σ_mid, 0).
- **Score-matching `t`** (`t = sample_t(...)`, `dmd2.py:118`): a **continuous
  random draw across the whole range** (loguniform over the expert's [σ_min,σ_max]).
  This is the `t` used to re-noise the output for the loss (`forward_process(gen_data,
  eps, t)`).

So: the student *generates* at specific nodes, but its output is *scored* at
continuous random levels. VSD needs the divergence sampled across the full noise
spectrum — otherwise the distributions would only be matched at the few nodes.

---

## 6. What "distribution matching" actually means

There is **one** target: the clean output distribution `p_teacher(x0 | c)`
(conditioned on the coarse input `c`). The goal is `p_student(x0|c) = p_teacher(x0|c)`.

"Across all σ" is **not** a distribution over σ — it's the *measurement basis*. For
each σ define the noised marginal `p_σ(x|c) = p(x0|c) * N(0, σ²)`. The family of
these blurred views over all σ uniquely determines the clean `p`. So **matching the
scores of the noised views at all σ ⟺ the clean distributions are equal.** The
all-σ comparison is the yardstick; the matched object is the single clean output
distribution.

The "mapping across all noise levels" is held by the **teacher** (and fake_score) —
used as references. The **student** only learns its few-step clean-x0 map.

---

## 7. The score, Tweedie, and the reverse ODE

**Score** = `∇ₓ log p_σ(x|c)`: the gradient of the log-density of the *noised* data
w.r.t. the noisy sample `x` (a vector field, same shape as x; points in the
denoising direction). Conditional on the coarse input `c`.

**Tweedie's formula** (an *algebraic identity*, not a differential equation; returns
a *point estimate*, not a density):

```
E[x0 | x, c] = x + σ² · ∇ₓ log p_σ(x|c)     ⟺     ∇ₓ log p_σ(x|c) = (D(x,σ,c) − x) / σ²
```

So an **x0-prediction net IS a reparameterized score model** — the denoiser `D` and
the score are the same object up to `σ²`. This is why the code compares **x0
predictions** (`fake_score_x0 − teacher_x0`) instead of scores directly: in x0-space
that difference equals `σ²·(score_student − score_teacher)`, with `σ²` folded into the
weight `w`. "`teacher_x0`" *is* the teacher's score, in x0 form.

**Don't confuse Tweedie with the reverse flow.** The *differential equation* that
transports a corrupted density back to the clean one is the **probability-flow ODE /
reverse SDE**:

```
dx/dσ = (x − D(x,σ,c)) / σ = −σ · ∇ₓ log p_σ(x|c)
```

Integrating it σ_max→0 carries noise (and the whole noised marginal) back to
`p_data(·|c)` — this is what the teacher's many-step sampler does. Tweedie is the
per-point building block (one step's target / the score); the ODE chains it across σ.

---

## 8. VSD, the `h` term, and forward- vs reverse-KL

**VSD = Variational Score Distillation** (`variational_score_distillation_loss`,
`common_loss.py:63-103`) — the core of DMD/DMD2. It updates the student by the
**difference of two scores** on the re-noised output:

```
gradient on gen_data  ∝  (fake_score_x0 − teacher_x0) · w
```

- `teacher_x0` = frozen teacher's score (target distribution).
- `fake_score_x0` = learned estimate of the **student's own** current score.
- **"Variational"** = the student's own score is intractable, so it's *approximated
  by a learned net* (`fake_score`, Section 10).

Implementation trick (`common_loss.py:99-102`): inside `no_grad`, build
`pseudo_target = gen_data − (fake_score_x0 − teacher_x0)·w` (detached), then
`loss = ½·MSE(gen_data, pseudo_target)`. So `d loss/d gen_data = (fake_score_x0 −
teacher_x0)·w` exactly — a **"fake gradient" injected through gen_data only**; the
score nets are not backpropped (they're frozen references for this step).

**The `h` term** (`f_distill.py:59-69`, `:20`):
- `r` = density-ratio estimate `p_teacher/p_student`, read from the discriminator
  logit (`ratio = exp(fake_logits.mean(dim=1))` → **one scalar per sample**, clamped
  to `[ratio_lower, ratio_upper]`, EMA-normalized). Per-sample, channels already
  collapsed.
- `h = f_div_weighting_function(r)` scales the per-sample VSD gradient. Table:
  `rkl → 1` (vanilla DMD), **`kl` (forward-KL) → r**, `js → 1−1/(1+r)`, etc.

**Reverse vs forward KL:**
- **Reverse KL** `KL(p_student ‖ p_teacher) = E_{x∼student}[log(p_student/p_teacher)]`.
  **Mode-seeking**: penalizes mass where the teacher has none, but NOT missing
  teacher modes → collapses to a subset (sharp, drops tails/diversity). `h=1`.
- **Forward KL** `KL(p_teacher ‖ p_student) = E_{x∼teacher}[log(p_teacher/p_student)]`.
  **Mass-covering**: `p_student→0` where `p_teacher` is high blows up the log →
  forced to cover all modes/tails. `h=r`. **Chosen here for tail preservation.**

**Why `h=r` converts reverse→forward:** the base VSD gradient is an expectation over
**student** samples (reverse-KL-shaped). Multiplying each student-drawn sample by the
importance weight `r = p_teacher/p_student` is importance sampling — it turns
"expectation over student" into "expectation over teacher" = the forward-KL gradient.
So `h=r` up-weights exactly the **teacher-likely, student-unlikely** samples (missed
modes / tails). This is why `ratio_upper` matters: those high-`r` tail samples
dominate and can blow up; clipping stabilizes, and a higher `ratio_upper` (e.g. 100)
keeps genuine tail samples contributing instead of being clipped — the tail lever.

---

## 9. fake_score: how it tracks a few-step student

`fake_score` is **not** trained on the student's denoising — it's trained on the
student's **output samples**, as a normal full diffusion model.

`_fake_score_discriminator_update_step` (`dmd2.py:319-358`):
1. `gen_data = gen_data_from_net(...)` (no_grad) — draw clean samples from the
   student.
2. `x_t_sg = forward_process(gen_data, eps, t)` — noise them to a **continuous random
   `t` (all σ)**.
3. `fake_score(x_t_sg, t)` → `denoising_score_matching_loss(...)` — ordinary DSM.

So the student only supplies **clean output samples** `gen_data ~ p_student(x0)`; the
all-σ noising is added by the fake_score trainer. The student's few-step / specific-
node nature is irrelevant — you train a diffusion model on any sample set the same
way. That's why fake_score can represent `∇log p_student,σ` across **all** σ. It's
refreshed each step as the student's outputs drift (typically initialized from the
teacher).

---

## 10. The discriminator (where it taps)

The GAN is a **projected discriminator on the teacher's ENCODER features**
(`_capture_encoder_features` hooks `unet.enc[block_key]` only — `fastgen_teacher.py:347`;
built from the bottleneck level in `fastgen_train.py:535-543`). Consequences:
- **Coarse** — taps the deepest/bottleneck encoder level (a coarse/global critic).
- **Channel-entangled** — the encoder mixes all output variables from layer 1; there
  is no per-variable signal.
- **Not the decoder / not the output** — it never sees the reconstructed fields.
- **Small** — GAN weight is `1e-3`; it's a stabilizer, the VSD term is the main driver.

(Project-specific levers built on this — tap depth, encoder-vs-decoder, per-variable
critics, spectral loss — live in `MOE_DISTILLATION_STATUS.md`.)

---

## 11. Gradient-flow cheat sheet

- **VSD term** → flows **only** into `gen_data` → student (via the pseudo-target MSE
  trick). Teacher and fake_score are frozen references; not backpropped.
- **GAN term** → `gan_loss_gen → discriminator → fake_feat → teacher encoder
  (frozen, grad passes through) → perturbed_data → gen_data → student`.
- **fake_score** and **discriminator** weights update only in their own step, on the
  student's (detached) generated samples.
- At inference: **student only**.

---

## 12. Project implications (pointers)

These mechanics drive the findings in `MOE_DISTILLATION_STATUS.md`:
- **The student is uniformly too smooth — a real high-k power deficit across all
  broadband fields** (confirmed on the raw `val/psd_*` curves; not a metric
  artifact). This one deficit drives *both* the under-powered high-k spectra and the
  tail under-prediction — they share a root, they are not separate.
- A **spectral loss** attaches at the `gen_data` node (clean output, every step) and
  should penalize the *deficit* (student PSD below teacher). Caveat: a pure mean-PSD
  match can be met with *incoherent* high-k noise → an adversarial fine-scale critic
  is better for coherent extremes; the spectral term is the cheaper scaffold.
- The **discriminator tap location** (coarse encoder bottleneck) is why the GAN can't
  add fine-scale power — blind to high-k for every variable — motivating the tap-depth
  A/B and the per-variable / decoder-tap critic ideas (now the *primary* lever).
- `ratio_upper` is **per-sample** (channels collapsed) — so it can't give per-channel
  tail emphasis; that needs per-variable critics or direct per-channel weights.

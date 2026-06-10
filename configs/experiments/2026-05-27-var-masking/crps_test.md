# Variable Masking Absorption Experiment

## Hypothesis

In a noise-conditioned SFNO trained with CRPS loss, input variable masking may be
"absorbed" by the noise conditioning pathway rather than driving the model to learn
mask-robust representations.

CRPS rewards calibrated ensemble spread. If ensemble members share the same mask
(non-iid) and the model has noise conditioning in its layernorms, the noise pathway
alone can generate the spread needed to satisfy CRPS. The model then has no gradient
pressure to learn how to predict well when a specific input is missing — the masking
signal is absorbed into noise.

## Experimental Design

A 2×2 ablation over noise conditioning and mask sharing, applied to nc-sfno models
with active masking (mask5, mask10, mask40):

|                          | Noise conditioning | No noise conditioning (`-nonoise`) |
|--------------------------|--------------------|------------------------------------|
| Shared mask (`non-iid`)  | A                  | B                                  |
| Per-member mask (`-iid`) | C                  | D                                  |

- **Noise conditioning**: `noise_embed_dim > 0` in the layernorms (default nc-sfno).
- **No noise conditioning**: `noise_embed_dim = 0`, disabling conditional layernorm scaling.
- **Shared mask (non-iid)**: all ensemble members receive the same input mask each step.
- **Per-member mask (iid)**: each ensemble member draws an independent mask each step.

sfno (no noise conditioning by construction) serves as an additional baseline and is
not part of the 2×2.

## Comparisons and What They Test

**A vs C — noise conditioning, non-iid vs iid (primary test)**
Does iid masking improve robustness when noise conditioning is present? If A < C,
noise conditioning is absorbing the masking signal when masks are shared: the model
satisfies CRPS via noise spread rather than learning mask-robust representations.

**B vs D — no noise conditioning, non-iid vs iid (control)**
Does iid masking help even without noise conditioning? If B ≈ D, the harm seen in
A vs C is specifically caused by the interaction with noise conditioning, not by mask
sharing per se.

**A vs B — non-iid, noise vs no noise (secondary diagnostic)**
Does removing noise conditioning improve a non-iid model? If B > A, noise conditioning
is directly hurting masking learning when masks are shared.

**C vs D — iid, noise vs no noise**
Does noise conditioning matter when masks are already independent? If C ≈ D, iid masking
neutralises the absorption mechanism — the CRPS spread comes from mask diversity, leaving
nothing for noise conditioning to absorb.

## Key Prediction Under the Absorption Hypothesis

Absorption predicts an interaction: the benefit of iid masking is larger when noise
conditioning is present than when it is absent:

```
(A − C) > (B − D)
```

If instead the effect sizes are similar, iid masking helps simply because diverse masked
inputs improve training, independent of noise conditioning.

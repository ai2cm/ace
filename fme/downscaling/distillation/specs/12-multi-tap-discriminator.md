# 12 — Multi-head (multi-scale) GAN discriminator for f-distill

## Goal

Implement and test FastGen's **multi-head discriminator** — several encoder
taps fed to the critic at once (`feature_indices` as a *set* of >1 element),
one head per tap, logits concatenated and averaged into the density ratio.
This is the "multiple GAN components" pattern the FastGen author pointed to,
and the one regime the ACE tap sweep has **never exercised**: every prior run
tapped a *single* encoder level.

Scope of this spec: the **lo-noise expert only** (expert 0), **spectral loss
OFF**, validated against the **residual-bug-fixed** baseline. Per-variable
critics, decoder/output-space taps, and the Hi expert are explicitly out of
scope (see below).

## Why this exists (history)

The single-tap depth sweep is done and is **non-monotone**: 64² (offset 2) is
the sweet spot; 32²/16² collapse (blind to hi-k), and **256²/512² collapse
when tapped *alone*** — the finer single critic wins the GAN and injects
incoherent hi-k (`gan_loss_gen`↑ 1.1–1.4, `gan_loss_disc`↓). Raising the GAN
weight on 64² also collapses. Winds hi-k and the PRMSL deep-low tail were
**not** fixed by any single tap. (Full record: `MOE_DISTILLATION_STATUS.md`
§ "Discriminator-tap A/B" and the 2026-07-02 non-monotone check-in.)

The untested hypothesis — and the reason to do this:

> A fine head (128²/256²) that **collapses alone** may contribute coherent
> hi-k **without** the disc winning when it is *anchored* by a stable 64² (and
> optionally bottleneck) head in one multi-head critic, because the ratio is
> the **mean over heads** (`fake_logits.mean(dim=1)`, `f_distill.py:67`) — the
> stable heads regularize the fine one. This is the classic projected-GAN
> multi-scale critic, and it directly targets what single taps could not fix
> (winds hi-k, deep-low tail) via simultaneous multi-scale coverage.

### ⚠️ Metric-validity caveat (must read before comparing to old runs)

The old tap runs (tap1/tap2/tap4/tap6, offsets 0–6) were validated with the
**residual-vs-full-field bug**: the student's validation output was the
*residual only* while the teacher val zarr was the *full field* (base added).
Fixed in `b2a47628b` (validation-only). Consequences for this spec:

- **Bunk in the old runs:** absolute PRMSL lo/mid spectra, PRMSL deep-low
  tail, PRMSL histogram/CRPS, PRMSL-driven checkpoint selection. Do **not**
  quote these.
- **Still valid:** all hi-band spectra (base-free), precip, training dynamics
  / GAN-collapse behavior, and **relative hi-band run-to-run comparisons** (the
  base is common-mode + low-k, so it cancels) — hence the "64² sweet spot" and
  "2-step > 1-step" findings stand on the hi-band.
- **Requirement:** every run in this spec launches from a commit **at or after
  the fix**, and compares to the fixed reset baselines below — never to the
  pre-fix tap numbers on any low-k / tail / PRMSL-distribution metric.

**Fixed reset baselines (2026-07-02, commit `184fa298b`), 2-step expert 0:**
- `baseline-fixed` (offset 0, bottleneck critic): `01KWJAFKZ96YBR73F0TETBKC0Q`
- `tap2-fixed` (offset 2, single 64² critic): `01KWJAFQW38E8AQ70YK7JYHCAK`

`tap2-fixed` is the incumbent-best single tap and the primary control — any
multi-tap win must beat it, and must attribute the delta to *multi-head*
(not just "has a 64² head").

## Verified current state (code)

- **The plumbing is multi-index-ready on the ACE side.**
  `AceDiffusionTeacher.forward(..., feature_indices=set)` +
  `_capture_encoder_features` (`fastgen_teacher.py:337`) already loop over a
  *sorted set*, hook the last block at each requested encoder level, and
  return `feat_list` in finest→coarsest order. No change needed there.
- **`encoder_feature_info()`** (`fastgen_teacher.py:306`) enumerates levels
  finest→coarsest as `(block_key, out_channels, resolution)`.
- **The auto-wiring hard-codes a single tap.**
  `fastgen_train.py:645-666` computes one `feature_idx = deepest_idx -
  disc_feature_depth` and builds `Discriminator_EDM(feature_indices={idx},
  all_res=..., in_channels=disc_channels)`. This is the only place that must
  change to enable a set.
- **`Discriminator_EDM` uses one `in_channels` for *all* heads**
  (`FastGen/fastgen/networks/discriminators.py:76`). So a multi-tap set works
  out-of-the-box **only if all tapped levels share a channel count**; mixed
  channels need an ACE subclass (FastGen stays unpatched — ARCHITECTURE.md
  rule).

### Expert-0 encoder level map (finest → coarsest)

| `all_res` idx | `disc_feature_depth` (offset) | resolution | channels |
|---|---|---|---|
| 0 | 5 | 512² | 128 |
| 1 | 4 | 256² | 256 |
| 2 | 3 | 128² | 256 |
| 3 | **2** | **64²** | **256** ← proven sweet spot |
| 4 | 1 | 32² | 128 |
| 5 | 0 | 16² | 128 (bottleneck) |

`feature_idx = 5 − offset`. Uniform-channel contiguous groups: offsets
`{2,3,4}` (all 256ch) and `{0,1}` (all 128ch).

## Implementation

### Task A — auto-wiring accepts a set of offsets (required)

- Extend the `--disc-feature-depth` knob to also accept a **comma-separated
  set** (e.g. `ACE_DISC_FEATURE_DEPTH=2,3,4`), parsed to a set of offsets.
  Keep single-int backward compatible (a lone int → single-element set, so
  existing runs and the `run.sh` flag are unchanged).
- Map each offset → `feature_idx = deepest_idx − offset`, clamp to `[0,
  deepest_idx]`, dedupe.
- Look up `(res, channels)` per idx from `encoder_feature_info()`.
- **If all tapped channels are equal:** build stock
  `Discriminator_EDM(feature_indices={...}, all_res=..., in_channels=<common>)`.
- **If channels differ:** build `AceMultiScaleDiscriminatorEDM` (Task B).
- Log every tapped `(idx, resolution, in_channels)` at launch (extend the
  existing `DMD2 discriminator: ...` line so the resolved multi-tap is
  auditable, matching how the single-tap line is verified in the run reports).

### Task B — `AceMultiScaleDiscriminatorEDM` (per-head channels; only if a mixed-channel set is tested)

- New subclass in the ACE distillation package (NOT in `FastGen/`), subclassing
  `Discriminator_EDM`, accepting `in_channels: list[int]` aligned to
  `sorted(feature_indices)`. Build each head's conv stack with its own channel
  count; keep the rest of `Discriminator_EDM.forward` (per-head logit, concat)
  intact — override only `__init__` head construction.
- Unit test: construct with a mixed `all_res`/`in_channels`, feed a list of
  feature maps with matching (res, channels), assert output shape
  `(B, num_heads)` and that head count == len(feature_indices). Mirror the
  existing discriminator-wiring test setup.

### Constraints

- Spectral loss **OFF** for all runs in this spec (`--spectral-weight 0` /
  unset) — isolate the multi-head GAN mechanism, comparable to the (fixed)
  single-tap runs which had no spectral term. Combining with spectral is a
  follow-up, not this spec.
- Lo-noise **expert 0 only** (`--expert 0`), 2-step, GAN weight at the baseline
  `1e-3` (single-64² could not tolerate more; test whether *multi-head* can as
  a secondary question, not the primary).
- All runs from a commit **≥ `b2a47628b`** (residual fix) — ideally current
  HEAD so they also carry the depth-based PRMSL tail (`80db7e7b1`) and the
  step-slidable PSD curves (`dc7876eeb`).

## Experiment matrix

Phase 1 is stock (Task A only, uniform channels — zero discriminator code).
Phase 2 needs Task B.

| # | Phase | offsets | idx | taps (res / ch) | code | question |
|---|---|---|---|---|---|---|
| E1 | 1 | `{2,3}` | 3,2 | 64²+128² (both 256ch) | A only | cheapest multi-tap: does anchor + one finer add hi-k w/o collapse? |
| E2 | 1 | `{2,3,4}` | 3,2,1 | 64²+128²+256² (all 256ch) | A only | **flagship** — does the anchor let 256² (which collapsed alone) *hold*? primary winds-hi-k test |
| E3 | 1 | `{0,1}` | 5,4 | 16²+32² (both 128ch) | A only | coarse control — confirms multi-head plumbing + isolates that it's the *fine* heads that matter |
| E4 | 2 | `{0,2,4}` | 5,3,1 | 16²+64²+256² (128/256/256) | A+B | full fine/mid/coarse pyramid — coarse head guards large-scale/deep-low, fine head adds texture |
| E5 | 2 | winner + GAN↑ | — | winner of E1–E4, `gan_weight 3e-3` | A(+B) | does multi-head tolerate the higher weight single-64² could not? |

Start with **E2** (flagship, no code) if only one run is possible; it is the
minimal change from `tap2-fixed` that tests the collapse-regularization
hypothesis. E1/E3 bracket it as cheap controls. Defer E4/E5 until Phase 1
shows a fine head contributing without collapse.

## Signals (comparable to the fixed reset baselines)

Primary (attribute deltas to multi-head vs `tap2-fixed`):
- `spec_mae_hi_PRMSL` — must stay held (~0.05, not collapse to ~0.8).
- `spec_mae_hi_{eastward,northward}_wind` — the gap single taps left (~0.58–0.75); a win here is the main prize.
- Raw `val/psd_*` (step-slidable) — **coherence check**: student hi-k tail must
  *approach* teacher, not **overshoot** (overshoot = incoherent-texture
  injection, the fine-tap failure mode).
- `tail_99.99_{PRMSL,eastward_wind,northward_wind,PRATEsfc}` — now measured on
  the fixed code (depth-based PRMSL tail); watch the PRMSL deep-low tail.

GAN health (collapse guard): `gan_loss_gen` / `gan_loss_disc` balance,
`fake_score_loss` drift. Collapse signature = `gan_loss_gen`→1.1–1.4 +
`gan_loss_disc`↓.

Now-trustworthy (post-fix): absolute PRMSL lo/mid spectra, PRMSL histogram/CRPS
— but still treat cross-run PRMSL low-k as secondary to the base-free hi-band.

## Out of scope (explicit)

- **Per-variable critics** (channel-split heads) — a separate, larger lever;
  the multi-tap here is multi-*resolution*, still channel-entangled.
- **Decoder / output-space taps** — FastGen's reference taps encoder only and
  short-circuits before the decoder (`EDM/network.py:543`); encoder skip
  connections carry encoder content into the decoder, so encoder multi-tap is
  tried first. Decoder taps require extending `_capture_encoder_features` to
  hook `unet.dec[...]` and are deferred to a follow-up spec.
- **Hi-noise expert** — everything interesting is in the lo expert; the Hi
  expert is not exercised here.
- **Combining with the spectral loss** — deferred; spectral OFF isolates the
  GAN mechanism first.

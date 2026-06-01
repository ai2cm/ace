# Plan: NaN Gradient Diagnostic ValueError Checks

## Context

Training of `NoiseConditionedSwinTransformer` crashes with `ValueError: Loss is NaN-valued` after ~900 steps. Gradient clipping (`max_grad_norm: 1.0`) is in place but the NaN still occurs. The existing check only catches NaN at the *loss* level — it doesn't tell us whether:
- (A) The backward pass produced NaN/inf gradients (forward pass overflow → bad backward), or
- (B) A prior optimizer step wrote NaN parameters (NaN gradients slipped past clipping), which then caused the next forward pass to produce NaN outputs.

We need checks at three points to pinpoint the failure mode.

---

## Checks to Add (all in `fme/core/optimization.py`)

### Check 1 — Pre-clip gradient norm (`_step_weights`)

**Where:** In `_step_weights`, after unscaling but before calling `clip_grad_norm_`.

**How:** Call `clip_grad_norm_(params, float("inf"))` — this returns the global gradient norm without modifying any gradients (clamp factor is 1.0 for any finite norm; we raise before reaching the real clip call if non-finite).

```python
pre_clip_norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
if not torch.isfinite(pre_clip_norm):
    raise ValueError(
        f"Gradient norm is {pre_clip_norm.item()} before clipping — "
        "the backward pass produced NaN/inf gradients. "
        "Gradient clipping cannot prevent weight corruption. "
        "Check for numerical instability in the forward or backward pass "
        "(e.g. CLN scale blowup, attention logit overflow)."
    )
```

**What this tells us:** If this fires, the backward pass itself is the problem — forward pass overflow produced NaN activations that propagated through autograd. The fix is upstream of clipping.

### Check 2 — Post-optimizer-step parameter NaN (`_step_weights`)

**Where:** In `_step_weights`, immediately after `self.optimizer.step()` / `self.gscaler.step(self.optimizer)`.

**How:**
```python
nan_param_count = sum(
    1 for p in params if not torch.isfinite(p).all()
)
if nan_param_count:
    raise ValueError(
        f"NaN/inf detected in {nan_param_count} parameter tensor(s) "
        "after optimizer step. Pre-clip gradient norm was finite — "
        "the optimizer update itself introduced NaN "
        "(possible cause: Adam second-moment underflow with very large gradients)."
    )
```

We only reach Check 2 if Check 1 passed (norm was finite), so the message can state that.

**What this tells us:** Gradients were finite but the optimizer step produced NaN weights — points to a learning-rate / Adam moment interaction.

### Check 3 — Post-backward gradient NaN (`_backward`)

**Where:** In `_backward`, after the `.backward()` call, gated on `self._max_grad_norm is not None` to avoid overhead when clipping is disabled.

**How:**
```python
if self._max_grad_norm is not None:
    nan_grad_count = sum(
        1 for group in self.optimizer.param_groups
        for p in group["params"]
        if p.grad is not None and not torch.isfinite(p.grad).all()
    )
    if nan_grad_count:
        raise ValueError(
            f"NaN/inf detected in gradients of {nan_grad_count} parameter tensor(s) "
            "immediately after backward pass. The forward pass or loss computation "
            "produced NaN values that propagated through autograd."
        )
```

**What this tells us:** Pinpoints the exact backward call where NaN gradients first appear, narrowing it to a specific loss term or forward step within gradient accumulation.

---

## Files to Modify

| File | Change |
|------|--------|
| `fme/core/optimization.py` | Add Checks 1, 2, 3 as described above |

---

## Ordering / Interaction

- Check 3 fires first (inside each `_backward` call during gradient accumulation).
- Check 1 fires next (in `_step_weights` before clipping).
- Check 2 fires last (in `_step_weights` after optimizer step).

If Check 3 fires → backward pass introduced NaN → look at CLN / attention in forward pass.  
If Check 1 fires but not Check 3 → gradient accumulation summed NaN from multiple terms.  
If Check 2 fires but not Check 1 → optimizer step introduced NaN with finite gradients.

---

## Verification

Run `FME_FORCE_CPU=1 python -m pytest fme/core/test_optimization.py -v` — existing tests must pass. Then relaunch training and observe which check fires first.

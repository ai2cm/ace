# Plan: Add Scaled Cosine Attention & Res-Post-Norm to ACE Swin Transformer

## Context

The ACE Swin Transformer (`fme/core/models/swin_transformer/`) currently uses standard
dot-product attention with pre-norm LayerNorm (applied before each sub-block). Swin
Transformer V2 introduced two training-stability improvements that are adopted here as
the new default architecture (no opt-in flags, no old paths retained):

- **Scaled cosine attention**: replaces dot-product with bounded cosine similarity +
  learnable per-head temperature, preventing attention collapse when input magnitudes vary
  across variables (relevant for multi-variable atmospheric fields).
- **Res-post-norm**: LayerNorm is applied to the residual branch *output* before adding
  it back to the skip connection, preventing activation amplitude blow-up with depth.

---

## Files to Modify

| File | Change |
|------|--------|
| `fme/core/models/swin_transformer/swin_layers.py` | Core attention + block changes |
| `fme/core/models/swin_transformer/test_swin_transformer.py` | Add tests; update stale `test_adaln_regression` docstring |

`swin_transformer.py` and `fme/ace/registry/swin_transformer.py` require **no changes**
because the features are unconditional — no flags propagate through the stack.

---

## Step 1 — Scaled Cosine Attention in `WindowAttention2D`

**File:** `swin_layers.py`, class `WindowAttention2D`

### `__init__` changes
- Remove `self.scale = head_dim ** -0.5`.
- Add `self.tau = nn.Parameter(torch.ones(1, num_heads, 1, 1))` unconditionally.

### `forward` changes

Replace the current dot-product block:
```python
q = q * self.scale
attn = q @ k.transpose(-2, -1)
```

With (unconditionally):
```python
norm_q = torch.norm(q, dim=-1, keepdim=True)
norm_k = torch.norm(k, dim=-1, keepdim=True).transpose(-2, -1)
attn = (q @ k.transpose(-2, -1)) / (norm_q * norm_k).clamp(min=1e-6)
attn = attn / self.tau.clamp(min=0.01)
```

Masking and softmax are unchanged. The CPB bias calls are updated in the subsection below.

### CPB bias — sigmoid bounding

Swin V2 bounds the positional bias so it cannot overwhelm the cosine attention values (which are in `[-1, 1]`). Wrap **both** `cpb_mlp` call sites in `forward` with `16.0 * torch.sigmoid(...)`:

**Regular path (no `lat_mean`)** — replace:
```python
bias = self.cpb_mlp(self.relative_coords_log)  # (N*N, num_heads)
```
With:
```python
bias = 16.0 * torch.sigmoid(self.cpb_mlp(self.relative_coords_log))  # (N*N, num_heads)
```

**`lat_mean` path** — replace:
```python
bias = self.cpb_mlp(coords_log)  # (nW, N*N, num_heads)
```
With:
```python
bias = 16.0 * torch.sigmoid(self.cpb_mlp(coords_log))  # (nW, N*N, num_heads)
```

The remaining reshape/add lines in each branch are unchanged. The existing zero-init of the last `cpb_mlp` layer is kept: at initialisation, the MLP output is 0 everywhere, so `16 * sigmoid(0) = 8.0` — a constant offset that softmax cancels out, giving a flat positional prior at init.

---

## Step 2 — Res-Post-Norm in `SwinTransformerBlock`

**File:** `swin_layers.py`, class `SwinTransformerBlock`

### `__init__` changes
None — `norm1` and `norm2` initialisation is unchanged. No new parameters.

### `forward` changes — AdaLN mode

> **Implementation note:** The current AdaLN branch reuses `x` as the working
> variable throughout the attention path (norm → shift/scale → window → attn →
> unwindow → column_mixer, all stored back into `x`).  Post-norm requires the
> raw input to remain available as the shortcut *and* the attention output to be
> available separately for norming.  Introduce `h` to hold the attention output,
> keeping `shortcut = x` unchanged.  Concretely, after unwindowing and the
> ColumnMixer folded residual, `h = attn_out + self.column_mixer(attn_out)`;
> norm1 is then applied to this combined output.  The pseudocode below uses `h`
> consistently for this purpose.
>
> **Gate placement:** Both branches use `shortcut + gate * drop_path(normed_residual)`
> (gate *outside* `drop_path`).  The current code has `gate_mlp` inside `drop_path`
> for Branch 2; this change makes both branches consistent.  The two orderings are
> equivalent since `drop_path` zeroes entire samples regardless.

Current pre-norm pattern:
```
norm1(x) → [AdaLN shift/scale] → attn → column_mixer → shortcut + gate*drop(result)
norm2(x) → [AdaLN shift/scale] → mlp → x + gate*drop(result)
```

New unconditional post-norm pattern:
```
attn(x) → column_mixer → norm1(result) → [AdaLN shift/scale] → shortcut + gate*drop(result)
mlp(x)  → norm2(result) → [AdaLN shift/scale] → x + gate*drop(result)
```

With AdaLN conditioning:
```python
# Branch 1 (attn)
shortcut = x
# ... windowing, attention on raw x (not normed), unwindowing, column_mixer → h ...
h_norm = self.norm1(h) * (1 + scale_msa) + shift_msa
x = shortcut + gate_msa * self.drop_path(h_norm)

# Branch 2 (mlp)
shortcut = x
h_norm = self.norm2(self.mlp(x)) * (1 + scale_mlp) + shift_mlp
x = shortcut + gate_mlp * self.drop_path(h_norm)
```

Without conditioning:
```python
# Branch 1
shortcut = x
# ... windowing, attention on raw x, unwindowing, column_mixer → h ...
x = shortcut + self.drop_path(self.norm1(h))

# Branch 2
shortcut = x
x = shortcut + self.drop_path(self.norm2(self.mlp(x)))
```

### `forward` changes — CLN mode

Current: `norm1(x, context)` applied before windowing.

Post-norm: attention runs on raw `x`, then `norm1(attn_out, context)` is applied to the
residual branch output before adding to shortcut. Same for MLP with `norm2`.

```python
# Branch 1 (attn)
shortcut = x
# ... windowing attn on raw x, unwindowing, column_mixer ...
h_norm = self.norm1(h.permute(0,3,1,2), context).permute(0,2,3,1)
x = shortcut + self.drop_path(h_norm)

# Branch 2 (mlp)
shortcut = x
y_norm = self.norm2(self.mlp(x).permute(0,3,1,2), context).permute(0,2,3,1)
x = shortcut + self.drop_path(y_norm)
```

---

## Step 3 — Tests

**File:** `fme/core/models/swin_transformer/test_swin_transformer.py`

`_build_net()` needs no signature changes. Add:

1. `test_cosine_attention_forward()` — forward pass completes without NaN; verify
   `tau` parameter exists on each `WindowAttention2D` in the net.
2. `test_v2_with_conditioning()` — forward + backward with AdaLN scalar/label
   conditioning; verify all parameters including `tau` receive gradients.

> **Note:** A plain forward + backward without conditioning is already covered by the
> existing `test_backward`; do not add a duplicate.  Update the docstring of
> `test_adaln_regression`: remove the sentence "byte-for-byte identical to the
> pre-CLN deterministic path" (stale after V2 changes) and replace the whole
> docstring with `"Forward pass produces the correct output shape in AdaLN mode."`

Tests must not reference any `cosine_attention` or `res_post_norm` flags.

---

## Verification

```bash
# Run the swin transformer unit tests
python -m pytest fme/core/models/swin_transformer/test_swin_transformer.py -v

# Run registry tests
python -m pytest fme/ace/registry/test_swin_transformer.py -v

# Run pre-commit (ruff + mypy)
pre-commit run --all-files
```

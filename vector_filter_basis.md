# DISCO Convolution with Vector-Typed Hidden Features

## Overview

This document specifies an approach for handling directional (vector-valued) data in DISCO convolution on the sphere. The design uses **radial filter basis functions** `ψ_k(r)` combined with **geometric angular factors** `cos(γ)` and `sin(γ)` (the frame-rotation angle between input and output points) to create filters that are rotationally invariant but not isotropic. The angular dependence enables gradient-like, divergence-like, and curl-like operations on the sphere while correctly transforming vectors between different local meridian frames.

The key insight is that a single set of radial basis functions ψ_k(r) generates three filter tensors — `ψ_k(r)` (scalar), `ψ_k(r)·cos(γ)`, and `ψ_k(r)·sin(γ)` — which are always applied in matched pairs. The cos/sin pairing is determined by geometry, not learned weights, so the network cannot break rotational symmetry. For example, a d/dx filter (for the u component) automatically has a corresponding d/dy filter (for the v component) from the same radial basis.

The frame-rotation angle between two points depends only on `(lat_out, lat_in, lon_in - lon_out)`, making it invariant to translations along longitude. This means all three filter tensors can use the FFT-based cross-convolution optimization.

The meridian frame's polar singularity does not pose a problem: the physics is mostly rotationally invariant, so it is acceptable for the reference orientation to be discontinuous at the poles. What matters is that vector-valued inputs are correctly handled — that is, the convolution accounts for the fact that neighboring points have different definitions of "north" and "east."

## Hidden Representation

The network maintains two types of hidden feature channels at each grid point:

- **Scalar channels** `x_scalar`: shape `(B, N_s, H, W)`. Frame-independent quantities (temperature, pressure, geopotential, etc.).
- **Vector channels** `x_vector`: shape `(B, N_v, H, W, 2)`. Each channel is a tangent vector `(u, v)` in the local meridian frame, where `u` is the eastward component and `v` is the northward component. The last dimension indexes `(u, v)`.

The number of scalar channels `N_s` and vector channels `N_v` are independent. Initially, vector channels hold the physical eastward and northward components of the input data (e.g., wind). Through the network, new vector channels are created by the convolution's scalar-to-vector pathway and weighted by interactions with scalar channels.

## The Meridian Frame and Its Discontinuity

The meridian frame is defined everywhere except at the geographic poles, where all meridians converge and "north" is undefined. At the poles, the frame has a rotational discontinuity: approaching the pole from different longitudes gives different limiting orientations.

This discontinuity is acceptable because:

1. **The physics is approximately rotationally invariant.** The dominant dynamics (advection, pressure gradients, wave propagation) do not have a preferred horizontal direction. The Coriolis force breaks this symmetry, but it varies smoothly with latitude, not with the frame convention.
2. **The discontinuity is at a measure-zero set of points.** On a discrete grid, the poles are either absent (Gaussian grid) or have a single row of points. The network can learn to handle these points through the local structure of the filter.
3. **Vector channels are correctly rotated within the filter support.** The frame rotation baked into the filter tensor ensures that each input vector is seen in the output point's frame, regardless of how different their meridian frames are.

## Frame Rotation in the Convolution

### The Problem

When a filter centered at an output point looks at a neighboring input point, the input's `(u, v)` components are measured in the input point's meridian frame. If the two points have different "north" directions (which they always do, unless they share a meridian), the raw `u` and `v` values cannot be directly combined — they would be mixing vectors expressed in different coordinate systems.

### The Solution

Before the filter integrates over input points, each input vector is rotated from the input point's meridian frame into the output point's meridian frame. This rotation is characterized by a single angle `γ(lat_out, lat_in, Δlon)` — the angle between the two points' north directions, as measured via parallel transport along the connecting geodesic.

The rotation acts as:

```
[u_rotated]   [cos γ  −sin γ] [u_in]
[v_rotated] = [sin γ   cos γ] [v_in]
```

where `(u_in, v_in)` are the input vector components in the input's meridian frame, and `(u_rotated, v_rotated)` are the same vector expressed in the output's meridian frame. This rotation is not applied at runtime — it is baked into the precomputed filter tensors `psi_cos` and `psi_sin`.

### Longitude Invariance

The rotation angle `γ` depends on `(lat_out, lat_in, Δlon)` where `Δlon = lon_in − lon_out`. It does **not** depend on absolute longitude. This is because the relative geometry of any two points — their geodesic distance, bearing angles, and frame rotation — is invariant under rotations about the polar axis.

This means `ψ_k(r) · cos(γ)` and `ψ_k(r) · sin(γ)` are both functions of the longitude difference, just like the scalar filter `ψ_k(r)`. The FFT-based cross-convolution optimization applies to all of them.

## Computing the Frame Rotation Angle

### Setup

The DISCO precomputation uses a YZY Euler rotation (rotation about the y-axis by `α = −θ_out`, where `θ_out` is the output colatitude) to bring the output point to the north pole. For each input point at colatitude `γ` and longitude `λ`, the rotated Cartesian position is:

```
x' = cos(α) cos(λ) sin(γ) + sin(α) cos(γ)
y' = sin(λ) sin(γ)
z' = −sin(α) cos(λ) sin(γ) + cos(α) cos(γ)
```

The code already computes `θ = arccos(z')` and `φ = atan2(y', x')` from this.

### Computing γ

The frame rotation angle `γ = φ − β`, where `β` is the angle that the input point's geographic north makes with the local "toward-pole" direction in the Euler-rotated frame. The components of `β` are:

```
cos(β) = ê_N' · (−θ̂)
sin(β) = ê_N' · φ̂
```

where:

- `ê_N'` is the input point's geographic north direction, Euler-rotated to the new frame:
  ```
  ê_N'_x = −cos(α) cos(γ) cos(λ) + sin(α) sin(γ)
  ê_N'_y = −cos(γ) sin(λ)
  ê_N'_z =  sin(α) cos(γ) cos(λ) + cos(α) sin(γ)
  ```
- `−θ̂` and `φ̂` are the local basis vectors at the rotated position `(θ, φ)`:
  ```
  −θ̂ = (−cos θ cos φ,  −cos θ sin φ,  sin θ)
  φ̂  = (−sin φ,          cos φ,         0    )
  ```

Then:

```
cos(γ) = cos(φ − β) = cos φ · cos β + sin φ · sin β
sin(γ) = sin(φ − β) = sin φ · cos β − cos φ · sin β
```

These quantities can be computed in the precomputation loop alongside `θ` and `φ`, adding only a few vector dot products per support point.

### Geometric Meaning

The angle `γ` is the total rotation from the input's meridian frame to the output's meridian frame, accounting for:

1. The input's geographic north direction relative to the Euler-frame local basis (the angle `β`)
2. The azimuthal position of the input point relative to the output point's meridian (the angle `φ`)

At the output point's location (where `θ → 0`), the Euler frame's `φ = 0` direction corresponds to the output point's geographic south, and `φ = π/2` corresponds to geographic east. The parallel transport of the input's north vector to the output point arrives at Cartesian angle `π + γ` from the x-axis in the output's tangent plane (since north is the `−x` direction).

## Convolution Operation

### Filter Tensors

Three banded FFT tensors are precomputed from the same radial basis functions `ψ_k(r)`:

```
psi_scalar_fft: shape (K, nlat_out, max_bw, nfreq)  — FFT of ψ_k(r)
psi_cos_fft:    shape (K, nlat_out, max_bw, nfreq)  — FFT of ψ_k(r) · cos(γ)
psi_sin_fft:    shape (K, nlat_out, max_bw, nfreq)  — FFT of ψ_k(r) · sin(γ)
```

The scalar filter `psi_scalar` is isotropic (depends only on geodesic distance). The oriented filters `psi_cos` and `psi_sin` have angular dependence — they are the same radial envelope modulated by the frame rotation angle. All three share the same sparsity pattern and banding structure.

### Contraction Intermediates

Using `contraction(psi, x)` to denote the standard FFT-based DISCO contraction (`_disco_s2_contraction_fft`), the convolution produces these intermediate features:

**From scalar inputs** (shape `(B, N_s, K, H, W)` each):
- `f_s = contraction(psi_scalar, x_scalar)` — isotropic filtering
- `f_cs = contraction(psi_cos, x_scalar)` — gradient-like, u-aligned
- `f_ss = contraction(psi_sin, x_scalar)` — gradient-like, v-aligned

**From vector inputs** — define shorthand `u = x_vector[..., 0]`, `v = x_vector[..., 1]` (each `(B, N_v, H, W)`):
- `A = contraction(psi_cos, u)` — shape `(B, N_v, K, H, W)`
- `B = contraction(psi_sin, u)`
- `C = contraction(psi_cos, v)`
- `D = contraction(psi_sin, v)`

From these four, we form rotationally invariant combinations:
- **Frame-rotated vector:** `conv_u = A − D`, `conv_v = B + C`
- **Divergence-like scalar:** `div = A + D`
- **Curl-like scalar:** `curl = C − B`

Total contraction calls: 1 (psi_scalar on scalars) + 2 (psi_cos and psi_sin on scalars) + 2 (psi_cos and psi_sin on each of u, v, but these can be batched as 2 calls over 2·N_v channels) = **5 contraction calls**.

### Weight Contraction

The weight tensor has separate blocks for each type interaction. Within each block, u and v pathways share the same weight values, enforcing rotational invariance.

**Scalar output from scalar input** — `W_ss`: shape `(N_s_out, N_s_in, K)`:
```
s_out += einsum("ock,bckxy->boxy", W_ss, f_s)
```

**Scalar output from vector input** — `W_vs`: shape `(N_s_out, N_v_in, K, 2)`:
```
s_out += einsum("ock,bckxy->boxy", W_vs[..., 0], div)    # divergence-like
s_out += einsum("ock,bckxy->boxy", W_vs[..., 1], curl)   # curl-like
```

**Vector output from scalar input** — `W_sv`: shape `(N_v_out, N_s_in, K, 2)`:
```
u_out += einsum("ock,bckxy->boxy", W_sv[..., 0], f_cs)               # gradient
u_out += einsum("ock,bckxy->boxy", W_sv[..., 1], -f_ss)              # perp gradient
v_out += einsum("ock,bckxy->boxy", W_sv[..., 0], f_ss)               # gradient
v_out += einsum("ock,bckxy->boxy", W_sv[..., 1], f_cs)               # perp gradient
```

Note: the same `W_sv` weight values produce both u and v, with the cos/sin factors swapped. The "gradient" component (index 0) produces a vector aligned with the spatial gradient of the scalar field. The "perpendicular gradient" component (index 1) produces a vector 90° rotated from the gradient — physically meaningful for e.g. geostrophic wind from pressure.

**Vector output from vector input** — `W_vv`: shape `(N_v_out, N_v_in, K, 2)`:
```
u_out += einsum("ock,bckxy->boxy", W_vv[..., 0], conv_u)             # stretch
u_out += einsum("ock,bckxy->boxy", W_vv[..., 1], -conv_v)            # rotate 90°
v_out += einsum("ock,bckxy->boxy", W_vv[..., 0], conv_v)             # stretch
v_out += einsum("ock,bckxy->boxy", W_vv[..., 1], conv_u)             # rotate 90°
```

The "stretch" component (index 0) preserves vector direction (scaling, advection-like). The "rotate" component (index 1) rotates vectors 90° (Coriolis-like).

### Why This Is Rotationally Invariant

Each weight block uses the same scalar weight for both u and v pathways. The directional structure comes entirely from the geometric factors cos(γ) and sin(γ), which are precomputed from the grid geometry. Because these factors always appear in matched cos/sin pairs, the resulting operations are invariant under rotations about the polar axis:

- Gradient and perpendicular gradient of a scalar are geometrically locked — rotating the input rotates the output vector consistently.
- Divergence and curl of a vector field are true scalars — they don't depend on the frame.
- Frame-rotated vector convolution and 90°-rotation both preserve vector transformation rules.

## Nonlinearities

Between convolution layers, **nonlinearities are applied only to scalar channels**. Standard activations (ReLU, GELU, etc.) can be applied freely to scalars without breaking any symmetry.

Vector channels do **not** receive direct nonlinearities. Applying independent nonlinearities to u and v (e.g., `ReLU(u)`, `ReLU(v)`) would break rotational structure because the result would depend on the frame orientation. Instead, vectors gain nonlinear behavior indirectly: scalar channels pass through nonlinearities, and the next convolution layer's scalar-to-vector pathway (via `psi_cos`/`psi_sin`) produces new vector features that depend nonlinearly on the original inputs.

## Performance Characteristics

### Compute Cost

For a layer with `N_s` scalar input channels and `N_v` vector input pairs:

| Operation | Contraction calls | Description |
|---|---|---|
| Scalar→scalar | 1 call over `N_s` channels | `psi_scalar` on scalars |
| Scalar→vector | 2 calls over `N_s` channels each | `psi_cos`, `psi_sin` on scalars |
| Vector→(vector+scalar) | 2 calls over `2·N_v` channels each | `psi_cos`, `psi_sin` on u and v |

Total: 5 contraction calls (3 for scalar inputs, 2 for vector inputs). The cost per call scales with `(channels × K × nlat × nlon × log(nlon))`.

### Memory Cost

Three banded FFT tensors (`psi_scalar_fft`, `psi_cos_fft`, `psi_sin_fft`) of the same shape are stored per layer. Total filter storage is 3× the scalar-only case. The extra memory is typically small relative to activations and weights.

### Comparison with Cube-Gauged Directional Filters

The cube-gauged approach (documented in `cube_filter_basis.md`) adds angular modes `sin(mφ)`, `cos(mφ)` to the filter basis, giving `1 + 2M` basis functions per radial scale. The post-convolution rotation to the cube frame is cheap (pointwise), and the cube construction provides a globally smooth (up to Z₄ vertex singularities) directional filter.

This approach is different in character:

- **Angular dependence from geometry, not learned modes.** The cos(γ)/sin(γ) angular factors are determined by the grid geometry, not learned. This means the angular structure is fixed (dipole-like), while the cube-gauged approach can learn arbitrary angular patterns.
- **No cube geometry.** No cube partition, blending zones, transition maps, or equivariant nonlinearity constraints.
- **Frame rotation is per-input-point.** The rotation matrix varies across the filter support (different input points have different frame mismatches), so it must be baked into the filter tensor. In contrast, the cube-frame rotation is at the output point only (constant across the sum, negligible cost).
- **Directional sensitivity through vector channels.** Gradient-like operations on scalar fields are possible (via the scalar-to-vector pathway), but the angular resolution is limited to the first angular mode (dipole). This is sufficient if the physically important directional information is carried by vector quantities (wind, currents, etc.), which is typically the case in atmospheric modeling.

## Design Parameters

- **K (radial basis size):** Number of radial filter basis functions per convolution layer. Same as the current `kernel_shape` for the filter basis. Controls the radial resolution of the filter.
- **N_s, N_v (channel counts):** Number of scalar channels and vector channels in the hidden representation. These are independent. A reasonable starting point is to match the current total channel count, reserving some for vector pairs.

## Module Interface

The new module `VectorDiscoConvS2` has this interface:

```python
class VectorDiscoConvS2(nn.Module):
    def __init__(
        self,
        in_channels_scalar: int,
        in_channels_vector: int,
        out_channels_scalar: int,
        out_channels_vector: int,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        kernel_shape: int | tuple[int, ...],
        basis_type: str = "piecewise linear",
        basis_norm_mode: str = "mean",
        grid_in: str = "equiangular",
        grid_out: str = "equiangular",
        bias: bool = True,
        theta_cutoff: float | None = None,
    ): ...

    def forward(
        self,
        x_scalar: torch.Tensor,   # (B, N_s_in, H, W)
        x_vector: torch.Tensor,   # (B, N_v_in, H, W, 2)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns (y_scalar, y_vector)
        # y_scalar: (B, N_s_out, H, W)
        # y_vector: (B, N_v_out, H, W, 2)
        ...
```

## Summary of Implementation Steps

1. **New precomputation function.** Create `_precompute_vector_convolution_tensor_s2` (in a new file `fme/core/disco/_vector_convolution.py`) that extends the existing precomputation logic. For each support point, compute `cos(γ)` and `sin(γ)` alongside `(θ, φ)`. Return three sets of sparse values: `vals_scalar = ψ_k(r)`, `vals_cos = ψ_k(r)·cos(γ)`, `vals_sin = ψ_k(r)·sin(γ)`. Call the existing `_precompute_psi_banded` three times (with shared sparsity pattern) to produce the three banded FFT tensors. Reuse the existing `FilterBasis`, `_normalize_convolution_tensor_s2`, and `_precompute_psi_banded` utilities.

2. **New convolution module.** Create `VectorDiscoConvS2` in the same new file. The `__init__` method:
   - Calls the new precomputation to get `psi_scalar_fft`, `psi_cos_fft`, `psi_sin_fft`.
   - Creates the four weight tensors: `W_ss`, `W_sv`, `W_vs`, `W_vv` with shapes as described above.
   - Creates bias parameters for scalar and vector outputs.

3. **Forward pass.** The `forward` method:
   - Runs 5 contraction calls (1 scalar-on-scalar, 2 cos/sin-on-scalar, 2 cos/sin-on-vector-uv).
   - Assembles intermediate features into the rotationally invariant combinations (conv_u, conv_v, div, curl, f_cs, f_ss).
   - Applies the four weight blocks to produce scalar and vector outputs.
   - Returns `(y_scalar, y_vector)`.

4. **Export.** Add `VectorDiscoConvS2` to `fme/core/disco/__init__.py`.

## Testing Plan

1. **Uniform vector field is preserved.** Create a constant vector field (e.g., everywhere pointing east). After vector→vector convolution, the output should still be uniform and pointing east — frame rotations cancel for a constant field.

2. **Longitude shift equivariance.** Shift scalar and vector inputs by N longitude points. Verify outputs shift by N points (both scalar and vector channels). Tests that the FFT structure preserves translational symmetry.

3. **Arbitrary rotation equivariance.** Rotate the entire input field by moving the north pole elsewhere (a full SO(3) rotation, not just a longitude shift). Verify the output field rotates consistently — scalar outputs match under the rotation, vector outputs transform correctly into the new frame. This is the strongest test of rotational invariance, exercising the frame rotation geometry at every latitude.

4. **Scalar gradient produces correct vector.** Apply scalar→vector to a scalar field with a known gradient (e.g., linear ramp in longitude → expect u output, zero v). Verify the output vector direction matches the gradient direction.

5. **Divergence recovery.** Create a radially diverging vector field from a point. Apply vector→scalar with divergence-like weights. Verify the output scalar is positive at the source.

6. **Curl recovery.** Create a rotating vector field. Apply vector→scalar with curl-like weights. Verify the result reflects the rotation.

7. **Scalar-only consistency.** When `N_v_in = N_v_out = 0`, verify the module produces identical results to the existing `DiscreteContinuousConvS2`.

8. **Regression baselines.** Save reference outputs from known inputs/weights, verify they don't change across refactors (using `validate_tensor_dict`).

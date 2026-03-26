# DISCO Convolution with Vector-Typed Hidden Features

## Overview

This document specifies an approach for handling directional (vector-valued) data in DISCO convolution on the sphere. Rather than constructing a continuous directional filter basis (as in the cube-gauged approach), this design uses **isotropic radial filters** combined with a **frame-rotation matrix** that correctly transforms vector inputs from each input point's local meridian frame into the output point's meridian frame.

The key insight is that the meridian frame's polar singularity does not pose a problem: the physics is mostly rotationally invariant, so it is acceptable for the reference orientation to be discontinuous at the poles. What matters is that vector-valued inputs are correctly handled — that is, the convolution accounts for the fact that neighboring points have different definitions of "north" and "east."

The frame-rotation angle between two points depends only on `(lat_out, lat_in, lon_in - lon_out)`, making it invariant to translations along longitude. This means the rotation can be baked into the filter tensor without breaking the FFT-based cross-convolution optimization.

## Hidden Representation

The network maintains three types of hidden feature channels at each grid point:

- **Scalar channels** `s(lat, lon)`: Frame-independent quantities (temperature, pressure, geopotential, etc.). No special treatment needed in convolution.
- **u channels** `u(lat, lon)`: East component of a vector in the local meridian frame (eastward wind, etc.).
- **v channels** `v(lat, lon)`: North component of a vector in the local meridian frame (northward wind, etc.).

Each `(u, v)` pair represents a tangent vector at the grid point, expressed in the local geographic frame where "east" = direction of increasing longitude and "north" = direction of decreasing colatitude (toward the geographic north pole).

Initially, `u` and `v` channels are the physical eastward and northward components of the input data. Through the network, new vector channels can be created via pointwise interactions (e.g., scalar times vector), and the convolution propagates them while correctly handling the frame geometry.

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

where `(u_in, v_in)` are the input vector components in the input's meridian frame, and `(u_rotated, v_rotated)` are the same vector expressed in the output's meridian frame.

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

These quantities can be computed in the existing precomputation loop alongside `θ` and `φ`, adding only a few vector dot products per support point.

### Geometric Meaning

The angle `γ` is the total rotation from the input's meridian frame to the output's meridian frame, accounting for:

1. The input's geographic north direction relative to the Euler-frame local basis (the angle `β`)
2. The azimuthal position of the input point relative to the output point's meridian (the angle `φ`)

At the output point's location (where `θ → 0`), the Euler frame's `φ = 0` direction corresponds to the output point's geographic south, and `φ = π/2` corresponds to geographic east. The parallel transport of the input's north vector to the output point arrives at Cartesian angle `π + γ` from the x-axis in the output's tangent plane (since north is the `−x` direction).

## Convolution Operation

### Filter Tensor for Scalar Channels

Unchanged from the current implementation. The filter basis functions `ψ_k(r)` depend only on geodesic distance. The precomputed banded FFT tensor is:

```
psi_scalar_fft: shape (K, nlat_out, max_bw, nfreq)
```

The contraction gives:

```
conv_scalar[b, c, k, lat, lon] = Σ_in ψ_k(r) · s_c(lat_in, lon_in)
```

### Filter Tensor for Vector Channels

For vector channels, the filter tensor incorporates the frame rotation. Two banded FFT tensors are precomputed:

```
psi_cos_fft: shape (K, nlat_out, max_bw, nfreq)  — FFT of ψ_k(r) · cos(γ)
psi_sin_fft: shape (K, nlat_out, max_bw, nfreq)  — FFT of ψ_k(r) · sin(γ)
```

These are built from the same radial filter basis `ψ_k(r)`, multiplied by `cos(γ)` and `sin(γ)` at each support point before banding and FFT.

### Vector Convolution

For each vector input pair `(u_c, v_c)` with radial basis `k`, the frame-rotated convolution gives the vector in the output point's frame:

```
conv_u[b, c, k, lat, lon] = contraction(psi_cos, u_c) − contraction(psi_sin, v_c)
conv_v[b, c, k, lat, lon] = contraction(psi_sin, u_c) + contraction(psi_cos, v_c)
```

where `contraction(psi, x)` denotes the standard FFT-based DISCO contraction (`_disco_s2_contraction_fft`).

This requires two contraction calls for vector channels (one with `psi_cos`, one with `psi_sin`), each processing all `u` and `v` channels together. The reassembly into `(conv_u, conv_v)` is a cheap pointwise operation.

### Weight Contraction

After convolution, the intermediate features are:

- From scalar inputs: `conv_scalar` with shape `(B, N_scalar, K, H, W)`
- From vector inputs: `conv_u` and `conv_v`, each with shape `(B, N_vec, K, H, W)`

All intermediate features are in the output point's meridian frame, so they can be freely mixed by the learned weight tensor. The weight produces output scalar, u, and v channels:

- **Scalar output from scalar input:** Standard — weight contracts over `(c_in, k)`.
- **Scalar output from vector input:** Weight contracts over `(c_in, k)` separately for `conv_u` and `conv_v`, producing a scalar from vector components (analogous to divergence or a directional projection).
- **Vector output from vector input:** Weight contracts over `(c_in, k)` for `conv_u` and `conv_v` to produce output `u` and `v` components. The weight can independently scale and mix the u/v components.
- **Vector output from scalar input:** The weight assigns a scalar intermediate feature to a vector output channel. This implicitly creates a preferred direction in the meridian frame, and is therefore not rotationally equivariant. But if `u_out` and `v_out` channels are created from the same scalar feature with appropriate weights, the network can learn latitude-dependent directional structures (e.g., patterns aligned with Coriolis deflection).

In practice, the simplest implementation treats all intermediate features as a flat channel dimension of size `N_scalar + 2 · N_vec` (with `conv_u` and `conv_v` occupying separate channel slots), and uses a single weight tensor of shape `(C_out, C_in_effective, K)` that mixes everything. The type labels (scalar, u, v) are bookkeeping for the next layer's convolution, not a constraint on the weight.

## Nonlinearities and Vector-Scalar Interactions

Between convolution layers, pointwise nonlinearities must respect the typed channel structure. All operations below are pointwise (per grid point) and therefore frame-consistent, since all channels at a given grid point share the same meridian frame.

### Safe Nonlinearities

- **Scalar channels:** Any standard nonlinearity (ReLU, GELU, etc.) applied independently.
- **Vector norm → scalar:** `n = sqrt(u² + v²)` is frame-invariant and produces a true scalar.
- **Norm-gated vectors:** `σ(n) · (u, v) / n` applies a nonlinearity to the vector magnitude while preserving direction.
- **Scalar × vector → vector:** `s · (u, v)` multiplies a scalar channel with a vector pair to produce a new vector pair. This is how the network creates new vector features from scalar-vector interactions.
- **Vector dot product → scalar:** `u₁·u₂ + v₁·v₂` produces a frame-invariant scalar from two vector pairs.
- **2D cross product → scalar:** `u₁·v₂ − u₂·v₁` produces a pseudo-scalar (changes sign under reflection but is frame-rotation invariant).

### Unsafe Nonlinearities

- **Independent nonlinearities on u and v:** Applying `ReLU(u)`, `ReLU(v)` independently breaks rotational structure — the result depends on the frame orientation.
- **Adding scalar to vector component:** `u + s` breaks vector transformation rules.

## Performance Characteristics

### Compute Cost

For a layer with `N_s` scalar input channels and `N_v` vector input pairs:

| Operation | Filter calls | Relative to scalar-only |
|---|---|---|
| Scalar convolution | 1 call with `psi_scalar` over `N_s` channels | baseline |
| Vector convolution | 2 calls (`psi_cos`, `psi_sin`) over `2·N_v` channels each | 4× per vector pair vs 1× per scalar |

Each "filter call" is one invocation of `_disco_s2_contraction_fft`. The cost per call scales with `(channels × K × nlat × nlon × log(nlon))`.

The total cost is `(N_s + 4·N_v) · K` filter applications, compared to `(N_s + 2·N_v) · K` if vector channels were treated as independent scalars (ignoring frame rotation). The overhead is a factor of 2× for the vector channels only.

### Memory Cost

Two additional banded FFT tensors (`psi_cos_fft`, `psi_sin_fft`) of the same shape as `psi_scalar_fft` are stored per layer. Total filter storage is 3× the scalar-only case. The extra memory is typically small relative to activations and weights.

### Comparison with Cube-Gauged Directional Filters

The cube-gauged approach (documented in `cube_filter_basis.md`) adds angular modes `sin(mφ)`, `cos(mφ)` to the filter basis, giving `1 + 2M` basis functions per radial scale. The post-convolution rotation to the cube frame is cheap (pointwise), and the cube construction provides a globally smooth (up to Z₄ vertex singularities) directional filter.

This approach is different in character:

- **No angular filter modes.** The filters remain isotropic (radial only). Directional information enters through the vector channels, not through the filter shape.
- **No cube geometry.** No cube partition, blending zones, transition maps, or equivariant nonlinearity constraints.
- **Frame rotation is per-input-point.** The rotation matrix varies across the filter support (different input points have different frame mismatches), so it must be baked into the filter tensor, doubling the vector channel cost. In contrast, the cube-frame rotation is at the output point only (constant across the sum, negligible cost).
- **Simpler but less expressive filters.** Isotropic filters cannot detect angular structure in scalar fields. Directional sensitivity comes only from vector inputs. This is sufficient if the physically important directional information is carried by vector quantities (wind, currents, etc.), which is typically the case in atmospheric modeling.

## Design Parameters

- **K (radial basis size):** Number of radial filter basis functions per convolution layer. Same as the current `kernel_shape` for `IsotropicMorletFilterBasis`. Controls the radial resolution of the filter.
- **N_s, N_v (channel counts):** Number of scalar and vector channel pairs in the hidden representation. The network architecture determines these. A reasonable starting point is to match the current total channel count, reserving some for vector pairs.
- **Nonlinearity design:** The choice of how to combine scalar and vector channels between layers (norm-gating, scalar-vector products, etc.) is an architectural decision. Simple norm-gating plus scalar×vector products provide the essential interactions.

## Summary of Implementation Steps

1. **Extend the precomputation loop.** In `_precompute_convolution_tensor_s2`, after computing `(θ, φ)` for each support point, additionally compute `cos(γ)` and `sin(γ)` from the input point's Euler-rotated north direction (as described in "Computing the Frame Rotation Angle"). Store three sets of sparse filter values: `ψ_k(r)` (scalar), `ψ_k(r)·cos(γ)` (vector cosine), `ψ_k(r)·sin(γ)` (vector sine).

2. **Build banded FFT tensors.** Run `_precompute_psi_banded` three times to produce `psi_scalar_fft`, `psi_cos_fft`, and `psi_sin_fft`. All three share the same sparsity pattern (same support points), so the banding and gather indices are identical.

3. **Modify the forward pass.** The convolution layer:
   - Applies `psi_scalar_fft` to scalar input channels → scalar intermediate features.
   - Applies `psi_cos_fft` and `psi_sin_fft` to vector input channels (u and v concatenated) → four intermediate tensors.
   - Reassembles the four vector intermediates into `(conv_u, conv_v)` via the rotation formula.
   - Concatenates scalar and vector intermediates and contracts with the weight tensor.

4. **Type-aware channel management.** Track which output channels are scalar vs. vector (u, v). This metadata is used by the next convolution layer to determine which channels need frame rotation and by the nonlinearity to apply type-appropriate operations.

5. **Implement equivariant nonlinearities.** Between convolution layers, apply norm-gated activations to vector pairs and standard activations to scalar channels. Optionally include scalar×vector product layers to enable cross-type feature creation.

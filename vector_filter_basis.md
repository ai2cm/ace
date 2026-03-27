# DISCO Convolution with Vector-Typed Hidden Features

## Overview

This document specifies an approach for handling directional (vector-valued) data in DISCO convolution on the sphere. The design distinguishes between two types of filters:

- **Scalar filters** — isotropic radial basis functions `ψ_k(r)` that depend only on geodesic distance.
- **Vector filters** — oriented filter pairs `(ψ_k(r)·cos(γ), ψ_k(r)·sin(γ))` formed by modulating a radial basis with the geometric frame-rotation angle `γ`. The two components are perpendicular by construction (they are the cosine and sine of the same angle), so rotating the first component by 90° gives the second.

The type of the output is determined by the types of the input and filter:

| Input | Filter | Output | Physical example |
|---|---|---|---|
| scalar | scalar | scalar | smoothing, diffusion |
| scalar | vector | vector | pressure gradient |
| vector | scalar | vector | advection of momentum |
| vector | vector | scalar | divergence, curl |

Each interaction involving a vector (either as input, filter, or output) carries two weight components — one for the operation and one for its 90° rotation. These two components correspond to a single weight on the two-component vector filter. The weight is shared between the u and v pathways, so the network cannot break rotational symmetry. This structure is equivalent to complex-valued weights: the "real" part gives gradient/divergence/stretch, and the "imaginary" part gives perpendicular-gradient/curl/rotation.

The scalar and vector filter bases can use different numbers of radial basis functions (`K_s` and `K_v`), allowing independent radial resolution for isotropic and oriented operations. They are paired in a `VectorFilterBasis`.

The frame-rotation angle between two points depends only on `(lat_out, lat_in, lon_in - lon_out)`, making it invariant to translations along longitude. This means all filter tensors can use the FFT-based cross-convolution optimization.

## Hidden Representation

The network maintains two types of hidden feature channels at each grid point:

- **Scalar channels** `x_scalar`: shape `(B, N_s, H, W)`. Frame-independent quantities (temperature, pressure, geopotential, etc.).
- **Vector channels** `x_vector`: shape `(B, N_v, H, W, 2)`. Each channel is a tangent vector `(u, v)` in the local meridian frame, where `u` is the eastward component and `v` is the northward component. The last dimension indexes `(u, v)`.

The number of scalar channels `N_s` and vector channels `N_v` are independent. Initially, vector channels hold the physical eastward and northward components of the input data (e.g., wind). Through the network, new vector channels are created by the convolution's scalar-to-vector pathway (vector filters applied to scalars).

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

where `(u_in, v_in)` are the input vector components in the input's meridian frame, and `(u_rotated, v_rotated)` are the same vector expressed in the output's meridian frame. This rotation is not applied at runtime — it is baked into the precomputed filter tensors.

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

From this, `θ = arccos(z')` and `φ = atan2(y', x')` give the angular position in the Euler-rotated frame.

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

### Filter Basis: VectorFilterBasis

A `VectorFilterBasis` pairs two radial `FilterBasis` instances:

- **Scalar basis** (K_s radial functions): used for the scalar→scalar path (isotropic filtering) and the vector→vector path (isotropic filtering with frame rotation).
- **Vector basis** (K_v radial functions): used for the scalar→vector path (gradient-like operations) and the vector→scalar path (divergence/curl-like operations). Each radial function generates a perpendicular filter pair via cos(γ) and sin(γ).

K_s and K_v can differ, allowing independent radial resolution for isotropic and oriented operations. When the same basis is used for both, K_s = K_v and the behavior matches the simpler single-basis case.

### Filter Tensors

Five banded FFT tensors are precomputed (from two bases):

**From the scalar basis** (K_s radial functions each):
```
psi_scalar_fft:  (K_s, nlat_out, bw_s, nfreq)  — FFT of ψ_k^s(r)
psi_s_cos_fft:   (K_s, nlat_out, bw_s, nfreq)  — FFT of ψ_k^s(r)·cos(γ)
psi_s_sin_fft:   (K_s, nlat_out, bw_s, nfreq)  — FFT of ψ_k^s(r)·sin(γ)
```

**From the vector basis** (K_v radial functions each):
```
psi_v_cos_fft:   (K_v, nlat_out, bw_v, nfreq)  — FFT of ψ_k^v(r)·cos(γ)
psi_v_sin_fft:   (K_v, nlat_out, bw_v, nfreq)  — FFT of ψ_k^v(r)·sin(γ)
```

Each basis has its own banding width (bw_s, bw_v) and gather index, since the support radii may differ.

The scalar filter `psi_scalar` is isotropic (depends only on geodesic distance). The cos/sin tensors from the scalar basis are used for frame rotation in the vector→vector path. The cos/sin tensors from the vector basis provide the oriented (gradient/divergence) operations.

The two components of each vector filter are perpendicular by construction: cos(γ) and sin(γ) are the cosine and sine of the same angle, so one is the 90° rotation of the other. This is verified by tests (Pythagorean identity and weighted orthogonality) for all supported basis types.

### Contraction Calls

Using `contraction(psi, x)` to denote the standard FFT-based DISCO contraction:

**Scalar input contractions** (3 calls):
1. `f_s = contraction(psi_scalar, x_scalar)` — shape `(B, N_s, K_s, H, W)`, for W_ss
2. `f_vc = contraction(psi_v_cos, x_scalar)` — shape `(B, N_s, K_v, H, W)`, for W_sv
3. `f_vs = contraction(psi_v_sin, x_scalar)` — shape `(B, N_s, K_v, H, W)`, for W_sv

**Vector input contractions** (4 calls, on `[u, v]` concatenated):
4. `sc_uv = contraction(psi_s_cos, [u, v])` — shape `(B, 2·N_v, K_s, H, W)`, for W_vv
5. `ss_uv = contraction(psi_s_sin, [u, v])` — shape `(B, 2·N_v, K_s, H, W)`, for W_vv
6. `vc_uv = contraction(psi_v_cos, [u, v])` — shape `(B, 2·N_v, K_v, H, W)`, for W_vs
7. `vs_uv = contraction(psi_v_sin, [u, v])` — shape `(B, 2·N_v, K_v, H, W)`, for W_vs

Total: **7 contraction calls** when K_s ≠ K_v. When both bases are the same (K_s = K_v, same `FilterBasis` object), calls 4+6 and 5+7 use identical filters and can be shared, reducing to **5 calls**.

From the vector contractions, form rotationally invariant intermediates:
- **Frame-rotated vector** (from scalar basis): `conv_u = sc_uv[:,:N_v] − ss_uv[:,N_v:]`, `conv_v = ss_uv[:,:N_v] + sc_uv[:,N_v:]`
- **Divergence** (from vector basis): `div = vc_uv[:,:N_v] + vs_uv[:,N_v:]`
- **Curl** (from vector basis): `curl = vc_uv[:,N_v:] − vs_uv[:,:N_v]`

### Weight Contraction

The weight tensor has separate blocks for each type interaction. Within each block, u and v pathways share the same weight values, enforcing rotational invariance. Each interaction involving a vector has a trailing `[2]` dimension representing the two perpendicular components of the vector filter.

**Scalar output from scalar input** — `W_ss`: shape `(N_s_out, N_s_in, K_s)`:
```
s_out += einsum("ock,bckxy->boxy", W_ss, f_s)
```

**Scalar output from vector input** — `W_vs`: shape `(N_s_out, N_v_in, K_v, 2)`:
```
s_out += einsum("ockd,bckxyd->boxy", W_vs, stack([div, curl], dim=-1))
```
Component 0 computes divergence (vector filter aligned with the vector), component 1 computes curl (vector filter perpendicular to the vector).

**Vector output from scalar input** — `W_sv`: shape `(N_v_out, N_s_in, K_v, 2)`:
```
u_out += einsum("ockd,bckxyd->boxy", W_sv, stack([f_vc, -f_vs], dim=-1))
v_out += einsum("ockd,bckxyd->boxy", W_sv, stack([f_vs,  f_vc], dim=-1))
```
The same weight produces both u and v, with cos/sin factors swapped. Component 0 is the gradient, component 1 is the perpendicular gradient (e.g., geostrophic wind from pressure).

**Vector output from vector input** — `W_vv`: shape `(N_v_out, N_v_in, K_s, 2)`:
```
u_out += einsum("ockd,bckxyd->boxy", W_vv, stack([conv_u, -conv_v], dim=-1))
v_out += einsum("ockd,bckxyd->boxy", W_vv, stack([conv_v,  conv_u], dim=-1))
```
Component 0 preserves vector direction (scaling, advection-like). Component 1 rotates vectors 90° (Coriolis-like).

### Symmetry Properties

The convolution is **exactly equivariant under longitude shifts** (azimuthal rotations), guaranteed by the FFT structure.

For non-azimuthal rotations (e.g., 180° rotation about the x-axis, which flips both latitude and longitude), the **scalar→scalar and vector→vector paths are equivariant**, but the **scalar↔vector cross-type paths are not**. This is because the cos(γ)/sin(γ) angular factors are invariant under such rotations while the output meridian frame rotates — the cross-type paths produce local frame components that don't track the frame change. This is inherent to the meridian-frame design and acceptable for atmospheric modeling, where the Coriolis force already breaks full SO(3) symmetry.

## Nonlinearities

Between convolution layers, **nonlinearities are applied only to scalar channels**. Standard activations (ReLU, GELU, etc.) can be applied freely to scalars without breaking any symmetry.

Vector channels do **not** receive direct nonlinearities. Applying independent nonlinearities to u and v (e.g., `ReLU(u)`, `ReLU(v)`) would break rotational structure because the result would depend on the frame orientation. Instead, vectors gain nonlinear behavior indirectly: scalar channels pass through nonlinearities, and the next convolution layer's scalar-to-vector pathway (vector filters applied to scalars) produces new vector features that depend nonlinearly on the original inputs.

## Performance Characteristics

### Compute Cost

For a layer with `N_s` scalar and `N_v` vector input channels, using scalar basis size K_s and vector basis size K_v:

| Operation | Contraction calls | Filter basis used |
|---|---|---|
| scalar + scalar filter → scalar | 1 over `N_s` channels | scalar (K_s) |
| scalar + vector filter → vector | 2 over `N_s` channels each | vector (K_v) |
| vector + scalar filter → vector | 2 over `2·N_v` channels each | scalar (K_s) |
| vector + vector filter → scalar | 2 over `2·N_v` channels each | vector (K_v) |

Total: 7 contraction calls when K_s ≠ K_v, or 5 when K_s = K_v (sharing cos/sin tensors). The cost per call scales with `(channels × K × nlat × nlon × log(nlon))`.

### Memory Cost

Five banded FFT tensors are stored per layer: 3 from the scalar basis (K_s each) and 2 from the vector basis (K_v each), plus two gather index tensors. When K_s = K_v with the same basis, this reduces to 3 unique tensors (the cos/sin tensors are shared).

## Module Interface

```python
class VectorFilterBasis:
    """Pairs scalar and vector radial filter bases."""
    def __init__(self, scalar_basis: FilterBasis, vector_basis: FilterBasis): ...

    scalar_kernel_size: int  # K_s
    vector_kernel_size: int  # K_v

def get_vector_filter_basis(
    scalar_kernel_shape, vector_kernel_shape, basis_type="piecewise linear",
) -> VectorFilterBasis: ...

class VectorDiscoConvS2(nn.Module):
    def __init__(
        self,
        in_channels_scalar: int,
        in_channels_vector: int,
        out_channels_scalar: int,
        out_channels_vector: int,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        vector_filter_basis: VectorFilterBasis | None = None,
        kernel_shape: int | tuple[int, ...] | None = None,  # convenience
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

When `kernel_shape` is provided instead of `vector_filter_basis`, a `VectorFilterBasis` is created with equal scalar and vector bases (K_s = K_v).

## Testing Plan

1. **Precomputation correctness.** Scalar filter values and indices match the scalar-only DISCO convolution precomputation. Pythagorean identity cos²(γ) + sin²(γ) = 1 at all support points. Filter component orthogonality verified for all basis types (piecewise linear, morlet, zernike).

2. **Frame rotation geometry.** Zero frame rotation on the same meridian (Δlon = 0). sin(γ) antisymmetric under Δlon → −Δlon. Scalar FFT tensors match the scalar-only DISCO convolution pipeline end-to-end.

3. **Longitude shift equivariance.** Shift scalar and vector inputs by N longitude points. Verify outputs shift by N points (both scalar and vector channels).

4. **R_x(π) rotation equivariance.** 180° rotation about the x-axis maps the equiangular grid to itself exactly. Verify scalar→scalar and vector→vector paths are equivariant. (Cross-type paths are not, by design.)

5. **Scalar gradient direction.** Apply scalar→vector to a longitude-varying scalar field. Verify the output vector points in the eastward direction at the equator.

6. **Divergence and curl detection.** Apply vector→scalar with divergence/curl weights to fields with known structure. Verify the output varies as expected.

7. **Scalar-only consistency.** When `N_v_in = N_v_out = 0`, verify the module produces identical results to the scalar-only `DiscreteContinuousConvS2`.

8. **Independent K_s and K_v.** Verify correct output shapes and weight dimensions when the scalar and vector bases have different kernel sizes.

9. **Gradient flow.** Verify gradients flow through all four weight blocks (W_ss, W_sv, W_vs, W_vv).

10. **Bias correctness.** Bias is added to scalar outputs only. With zero weights and nonzero bias, scalar output equals the bias everywhere and vector output is zero.

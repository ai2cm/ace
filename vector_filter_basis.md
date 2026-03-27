# DISCO Convolution with Vector-Typed Hidden Features

## Overview

This document specifies an approach for handling directional (vector-valued) data in DISCO convolution on the sphere. The design distinguishes between two types of filters:

- **Scalar filters** — isotropic radial basis functions `ψ_k(r)` that depend only on geodesic distance.
- **Vector filters** — oriented filter pairs formed by modulating a radial basis with the bearing angle `φ` (the direction from the output point to the input point, in the output's local frame). The two components `(ψ_k(r)·cos(φ), ψ_k(r)·sin(φ))` are perpendicular by construction, since they are the cosine and sine of the same angle.

The type of the output is determined by the types of the input and filter:

| Input | Filter | Output | Physical example |
|---|---|---|---|
| scalar | scalar | scalar | smoothing, diffusion |
| scalar | vector | vector | pressure gradient |
| vector | scalar | vector | advection of momentum |
| vector | vector | scalar | divergence, curl |

All operations are defined in the output point's local meridian frame. When the input is a vector, it is first rotated from its own meridian frame into the output's frame (via the frame rotation angle γ), then the filter is applied. This separation — frame rotation for vectors, bearing angle for directional filters — gives all four paths full equivariance under rotations.

Each interaction involving a vector (either as input, filter, or output) carries two weight components — one for the operation and one for its 90° rotation. These two components correspond to a single weight on the two-component vector filter. The weight is shared between the u and v pathways, so the network cannot break rotational symmetry.

The scalar and vector filter bases can use different numbers of radial basis functions (`K_s` and `K_v`), allowing independent radial resolution for isotropic and oriented operations. They are paired in a `VectorFilterBasis`.

All angular factors (`φ`, `β`, `γ`) depend only on `(lat_out, lat_in, lon_in - lon_out)`, so all filter tensors are compatible with the FFT-based cross-convolution optimization.

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

## Geometric Angles

Three angles characterize the relationship between an output point and each input point in its filter support. All three depend only on `(lat_out, lat_in, Δlon)` and are computed during the precomputation step.

### The bearing angle φ

The Euler rotation brings the output point to the north pole. In the Euler-rotated frame, each input point has an azimuthal position `φ = atan2(y', x')`. This is the **bearing angle**: the direction from the output point toward the input point, measured in the output's local frame.

At the output point's location (θ → 0 in the Euler frame), φ = 0 corresponds to geographic south and φ = π/2 corresponds to geographic east. So the bearing from the output toward an input at azimuth φ has components `(sin(φ), −cos(φ))` in the output's (east, north) frame.

The vector filter uses `cos(φ)` and `sin(φ)` as its two perpendicular components. Under a rotation of the sphere, the bearing rotates with the output's frame, so the vector filter output transforms correctly as a vector.

### The frame correction angle β

The angle β measures the orientation of the input point's meridian frame relative to the Euler-rotated frame's local basis. It is computed from the Euler-rotated geographic north direction of the input point:

```
cos(β) = ê_N' · (−θ̂)
sin(β) = ê_N' · φ̂
```

where `ê_N'` is the input's geographic north after Euler rotation, and `−θ̂`, `φ̂` are the local basis vectors at the input's Euler-rotated position.

### The frame rotation angle γ = φ − β

The frame rotation `γ` is the total angle between the input's and output's meridian frames. It combines the bearing (φ) with the frame correction (β). This is the angle used to rotate input vectors from the input's frame into the output's frame:

```
[u_rotated]   [cos γ  −sin γ] [u_in]
[v_rotated] = [sin γ   cos γ] [v_in]
```

### Which angle is used where

| Path | Angular factor | Reason |
|---|---|---|
| vv (vector→vector) | γ (frame rotation) | Rotates input vector to output frame |
| sv (scalar→vector) | φ (bearing) | Bearing direction is a vector at the output |
| vs (vector→scalar) | β = φ − γ (frame correction) | Equivalent to: rotate input vector to output frame, then project onto bearing |

The vs path uses β because the operation is conceptually two steps: (1) rotate the input vector to the output's frame using γ, then (2) project onto the bearing direction using φ. Expanding the composition `cos(φ)·[cos(γ)·u − sin(γ)·v] + sin(φ)·[sin(γ)·u + cos(γ)·v]` simplifies to `cos(β)·u + sin(β)·v`, where `β = φ − γ`.

## Computing the Geometric Angles

### Setup

The DISCO precomputation uses a YZY Euler rotation (rotation about the y-axis by `α = −θ_out`, where `θ_out` is the output colatitude) to bring the output point to the north pole. For each input point at colatitude `γ_in` and longitude `λ`, the rotated Cartesian position is:

```
x' = cos(α) cos(λ) sin(γ_in) + sin(α) cos(γ_in)
y' = sin(λ) sin(γ_in)
z' = −sin(α) cos(λ) sin(γ_in) + cos(α) cos(γ_in)
```

From this, `θ = arccos(z')` gives the geodesic distance and `φ = atan2(y', x')` gives the bearing angle.

### Computing β

The Euler-rotated geographic north of the input point is:

```
ê_N'_x = −cos(α) cos(γ_in) cos(λ) + sin(α) sin(γ_in)
ê_N'_y = −cos(γ_in) sin(λ)
ê_N'_z =  sin(α) cos(γ_in) cos(λ) + cos(α) sin(γ_in)
```

The local basis vectors at the rotated position `(θ, φ)`:

```
−θ̂ = (−cos θ cos φ,  −cos θ sin φ,  sin θ)
 φ̂ = (−sin φ,          cos φ,         0    )
```

Then:

```
cos(β) = ê_N' · (−θ̂)
sin(β) = ê_N' · φ̂
```

### Computing γ from φ and β

```
cos(γ) = cos(φ − β) = cos φ · cos β + sin φ · sin β
sin(γ) = sin(φ − β) = sin φ · cos β − cos φ · sin β
```

All three angle pairs `(cos φ, sin φ)`, `(cos β, sin β)`, `(cos γ, sin γ)` are computed in the precomputation loop and stored at each support point.

## Convolution Operation

### Filter Basis: VectorFilterBasis

A `VectorFilterBasis` pairs two radial `FilterBasis` instances:

- **Scalar basis** (K_s radial functions): used for the scalar→scalar path (isotropic filtering) and the vector→vector path (isotropic filtering with frame rotation).
- **Vector basis** (K_v radial functions): used for the scalar→vector path (bearing-based gradient) and the vector→scalar path (bearing-based divergence/curl).

K_s and K_v can differ, allowing independent radial resolution for isotropic and oriented operations.

### Filter Tensors

Seven banded FFT tensors are precomputed from two bases:

**From the scalar basis** (K_s radial functions each):
```
psi_scalar_fft:    (K_s, nlat_out, bw_s, nfreq)  — FFT of ψ_k^s(r)
psi_s_cos_γ_fft:   (K_s, nlat_out, bw_s, nfreq)  — FFT of ψ_k^s(r)·cos(γ)
psi_s_sin_γ_fft:   (K_s, nlat_out, bw_s, nfreq)  — FFT of ψ_k^s(r)·sin(γ)
```

**From the vector basis** (K_v radial functions each):
```
psi_v_cos_φ_fft:   (K_v, nlat_out, bw_v, nfreq)  — FFT of ψ_k^v(r)·cos(φ)
psi_v_sin_φ_fft:   (K_v, nlat_out, bw_v, nfreq)  — FFT of ψ_k^v(r)·sin(φ)
psi_v_cos_β_fft:   (K_v, nlat_out, bw_v, nfreq)  — FFT of ψ_k^v(r)·cos(β)
psi_v_sin_β_fft:   (K_v, nlat_out, bw_v, nfreq)  — FFT of ψ_k^v(r)·sin(β)
```

Each basis has its own banding width (bw_s, bw_v) and gather index, since the support radii may differ.

The scalar filter `psi_scalar` is isotropic. The γ-modulated tensors from the scalar basis handle frame rotation for the vv path. The φ-modulated tensors from the vector basis provide the bearing-based gradient for sv. The β-modulated tensors provide the bearing-projected divergence/curl for vs.

The two components of each angular pair are perpendicular by construction (cosine and sine of the same angle).

### Contraction Calls

Using `contraction(psi, x)` to denote the standard FFT-based DISCO contraction:

**Scalar input contractions** (3 calls):
1. `f_s = contraction(psi_scalar, x_scalar)` — shape `(B, N_s, K_s, H, W)`, for W_ss
2. `f_vc = contraction(psi_v_cos_φ, x_scalar)` — shape `(B, N_s, K_v, H, W)`, for W_sv
3. `f_vs = contraction(psi_v_sin_φ, x_scalar)` — shape `(B, N_s, K_v, H, W)`, for W_sv

**Vector input contractions** (4 calls, on `[u, v]` concatenated):
4. `sc_uv = contraction(psi_s_cos_γ, [u, v])` — shape `(B, 2·N_v, K_s, H, W)`, for W_vv
5. `ss_uv = contraction(psi_s_sin_γ, [u, v])` — shape `(B, 2·N_v, K_s, H, W)`, for W_vv
6. `bc_uv = contraction(psi_v_cos_β, [u, v])` — shape `(B, 2·N_v, K_v, H, W)`, for W_vs
7. `bs_uv = contraction(psi_v_sin_β, [u, v])` — shape `(B, 2·N_v, K_v, H, W)`, for W_vs

Total: **7 contraction calls** always.

From the vector contractions, form rotationally invariant intermediates:
- **Frame-rotated vector** (from scalar basis, using γ): `conv_u = sc_uv[:,:N_v] − ss_uv[:,N_v:]`, `conv_v = ss_uv[:,:N_v] + sc_uv[:,N_v:]`
- **Divergence** (from vector basis, using β): `div = bc_uv[:,:N_v] + bs_uv[:,N_v:]`
- **Curl** (from vector basis, using β): `curl = bc_uv[:,N_v:] − bs_uv[:,:N_v]`

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
Component 0 computes divergence (bearing-aligned projection of the frame-rotated input vector). Component 1 computes curl (perpendicular projection).

**Vector output from scalar input** — `W_sv`: shape `(N_v_out, N_s_in, K_v, 2)`:
```
u_out += einsum("ockd,bckxyd->boxy", W_sv, stack([f_vc, -f_vs], dim=-1))
v_out += einsum("ockd,bckxyd->boxy", W_sv, stack([f_vs,  f_vc], dim=-1))
```
The same weight produces both u and v, with cos/sin factors swapped. Component 0 is the gradient (bearing-aligned), component 1 is the perpendicular gradient (e.g., geostrophic wind from pressure).

**Vector output from vector input** — `W_vv`: shape `(N_v_out, N_v_in, K_s, 2)`:
```
u_out += einsum("ockd,bckxyd->boxy", W_vv, stack([conv_u, -conv_v], dim=-1))
v_out += einsum("ockd,bckxyd->boxy", W_vv, stack([conv_v,  conv_u], dim=-1))
```
Component 0 preserves vector direction (scaling, advection-like). Component 1 rotates vectors 90° (Coriolis-like).

### Symmetry Properties

The convolution is **exactly equivariant under longitude shifts** (azimuthal rotations), guaranteed by the FFT structure.

All four paths are equivariant under non-azimuthal rotations (e.g., 180° rotation about the x-axis). This is because:

- The **vv path** uses the frame rotation γ to correctly transform input vectors to the output frame, then applies an isotropic filter. Both the frame rotation and the isotropic filter are equivariant.
- The **sv path** uses the bearing angle φ, which transforms as a vector at the output point. Under rotation, the bearing rotates with the output frame, so the vector output transforms correctly.
- The **vs path** uses β = φ − γ, which is equivalent to first frame-rotating the input vector (via γ) then projecting onto the bearing (via φ). Since both steps are equivariant, the composition is equivariant, producing a true scalar.
- The **ss path** is isotropic and trivially equivariant.

## Nonlinearities

Between convolution layers, **nonlinearities are applied only to scalar channels**. Standard activations (ReLU, GELU, etc.) can be applied freely to scalars without breaking any symmetry.

Vector channels do **not** receive direct nonlinearities. Applying independent nonlinearities to u and v (e.g., `ReLU(u)`, `ReLU(v)`) would break rotational structure because the result would depend on the frame orientation. Instead, vectors gain nonlinear behavior indirectly: scalar channels pass through nonlinearities, and the next convolution layer's scalar-to-vector pathway (vector filters applied to scalars) produces new vector features that depend nonlinearly on the original inputs.

## Performance Characteristics

### Compute Cost

For a layer with `N_s` scalar and `N_v` vector input channels, using scalar basis size K_s and vector basis size K_v:

| Operation | Contraction calls | Angular factor | Filter basis |
|---|---|---|---|
| scalar + scalar filter → scalar | 1 over `N_s` channels | (none) | scalar (K_s) |
| scalar + vector filter → vector | 2 over `N_s` channels each | φ (bearing) | vector (K_v) |
| vector + scalar filter → vector | 2 over `2·N_v` channels each | γ (frame rotation) | scalar (K_s) |
| vector + vector filter → scalar | 2 over `2·N_v` channels each | β (frame correction) | vector (K_v) |

Total: 7 contraction calls. The cost per call scales with `(channels × K × nlat × nlon × log(nlon))`.

### Memory Cost

Seven banded FFT tensors are stored per layer: 3 from the scalar basis (K_s each) and 4 from the vector basis (K_v each), plus two gather index tensors.

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

1. **Precomputation correctness.** Scalar filter values and indices match the scalar-only DISCO convolution precomputation. Pythagorean identity cos² + sin² = 1 holds for each angular pair (φ, β, γ) at all support points. Filter component orthogonality verified for all basis types (piecewise linear, morlet, zernike).

2. **Frame rotation geometry.** Zero frame rotation on the same meridian (Δlon = 0). sin(γ) antisymmetric under Δlon → −Δlon. Scalar FFT tensors match the scalar-only DISCO convolution pipeline end-to-end.

3. **Longitude shift equivariance.** Shift scalar and vector inputs by N longitude points. Verify outputs shift by N points (both scalar and vector channels).

4. **R_x(π) rotation equivariance.** 180° rotation about the x-axis maps the equiangular grid to itself exactly. Verify all four paths (ss, sv, vs, vv) are equivariant under this rotation.

5. **Scalar gradient direction.** Apply scalar→vector to a longitude-varying scalar field. Verify the output vector points in the eastward direction at the equator.

6. **Divergence and curl detection.** Apply vector→scalar with divergence/curl weights to fields with known structure. Verify the output varies as expected.

7. **Scalar-only consistency.** When `N_v_in = N_v_out = 0`, verify the module produces identical results to the scalar-only `DiscreteContinuousConvS2`.

8. **Independent K_s and K_v.** Verify correct output shapes and weight dimensions when the scalar and vector bases have different kernel sizes.

9. **Gradient flow.** Verify gradients flow through all four weight blocks (W_ss, W_sv, W_vs, W_vv).

10. **Bias correctness.** Bias is added to scalar outputs only. With zero weights and nonzero bias, scalar output equals the bias everywhere and vector output is zero.

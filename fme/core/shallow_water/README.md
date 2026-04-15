# Shallow Water: Neural-Network-Like Atmospheric Dynamics

This module implements a hierarchy of atmospheric dynamical models using the
VectorDiscoConvS2 spherical convolution as the core differential operator.
Each model uses frozen (fixed-weight) convolutions and pointwise operations
to represent the exact physics, providing a foundation that can later be
relaxed into a learnable neural network.

## Model Hierarchy

The models form a progression from simplest to most general:

### 1. ShallowWaterStepper (`stepper.py`)

Single-level linearized shallow water equations. The simplest model:

    dh'/dt = -H0 div(V)
    dV/dt  = -g grad(h') - f x V

Demonstrates the basic architecture: a VectorDiscoBlock computes gradient
and divergence via its W_sv and W_vs weight paths, while ScalarVectorProduct
handles Coriolis forcing.

### 2. PrimitiveEquationsStepper (`primitive_equations.py`)

Multi-level hydrostatic primitive equations on fixed pressure levels.
Extends the shallow water model with:

- **Hydrostatic coupling**: temperature at lower levels drives geopotential
  at upper levels via phi_k = phi_{k-1} + R T_{k-1} ln(p_{k-1}/p_k)
- **Nonlinear momentum advection** in Lamb form: (V . nabla)V = nabla(KE) - zeta x V
- **Full dry-adiabatic thermodynamic equation**: dT/dt includes horizontal
  advection, adiabatic compression heating, and vertical temperature advection
- **Humidity as passive tracer**: dq/dt = -V . nabla(q)
- **Vertical velocity omega**: diagnosed from continuity (div V integrated upward)

### 3. SigmaCoordinateStepper (`sigma_equations.py`)

Primitive equations in sigma coordinates (sigma = p / p_s) with prognostic
surface pressure. Extends the isobaric model with:

- **Prognostic surface pressure**: dp_s/dt from column-integrated mass divergence
- **Sigma-dot vertical velocity**: diagnosed from continuity in sigma coordinates
- **Vertical advection**: -sigma_dot dX/dsigma for velocity and temperature
- **Modified pressure gradient force**: includes the R T nabla(ln p_s) correction

### 4. HybridCoordinateStepper (`hybrid_equations.py`)

The most general vertical coordinate: p_k = a_k + b_k p_s. This is the
coordinate system used by operational weather models (CAM, IFS). Special cases:

- Pure sigma: a_k = 0, b_k = sigma_k
- Pure pressure: a_k = p_k (constant), b_k = 0

Adds spatially-varying layer thicknesses and a more complex vertical velocity
integration compared to the sigma formulation.

## Block Steppers (Levels as Channels)

Each multi-level stepper has a "block" variant that encodes the entire
multi-level dynamics in a single VectorDiscoBlock forward pass:

### PrimitiveEquationsBlockStepper (`primitive_equations_block.py`)

Packs all K levels into channel dimensions:
- Scalars: [T_0, KE_0, zeta_0, delta_0, ..., T_{K-1}, KE_{K-1}, zeta_{K-1}, delta_{K-1}, f]
- Vectors: [V_0, V_1, ..., V_{K-1}]

The W_sv weights are lower-triangular across levels to encode the hydrostatic
relation (phi_k depends on T at all levels below k). A separate VectorDiscoConvS2
with the W_vs2 pathway handles temperature and humidity advection.

### HybridCoordinateBlockStepper (`hybrid_equations_block.py`)

Extends the block approach to hybrid coordinates, replacing Python for-loops
with frozen Conv2d layers for vertical operations:
- Hydrostatic recurrence (lower-triangular Conv2d)
- Vertical velocity integration (lower-triangular Conv2d)
- Vertical finite differences (tridiagonal Conv2d)

Only two DISCO passes total (one block + one advection conv).

## Motivation

The purpose of this hierarchy is to establish that the VectorDiscoConvS2
architecture can exactly represent the differential operators needed for
atmospheric dynamics. By starting with frozen physics weights and
progressively relaxing them, we can build neural networks that are
initialized near the correct physics and can learn corrections from data.

The block steppers demonstrate that a single forward pass through the
standard network architecture (VectorDiscoBlock) is sufficient to encode
multi-level dynamics, making the transition from physics model to neural
network straightforward.

## Supporting Modules

- **`block.py`**: VectorDiscoBlock — the repeating unit (conv + skip + activation + MLP + ScalarVectorProduct + residual)
- **`scalar_vector_product.py`**: ScalarVectorProduct (pointwise rotation/scaling) and VectorDotProduct (frame-invariant |V|^2)

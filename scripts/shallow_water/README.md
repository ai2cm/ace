# Shallow Water Scripts

Evaluation and demonstration scripts for the multi-level atmospheric
dynamics hierarchy implemented in `fme/core/shallow_water/`.

## Scripts

### `geostrophic_adjustment.py`

Classical geostrophic adjustment test using `PrimitiveEquationsStepper`.

Initializes a warm temperature dome at 40N with zero wind and integrates
forward in time. The warm anomaly creates a geopotential gradient that
drives divergent outflow, which is then deflected by the Coriolis force
into an anticyclonic (clockwise in NH) circulation. The adjustment
timescale is the inertial period f^{-1} ~ 17 hours.

This test validates:
- Hydrostatic coupling (temperature drives geopotential)
- Pressure gradient force direction and magnitude
- Coriolis deflection producing balanced geostrophic flow
- Vertical structure: upper levels show stronger anticyclone
  due to hydrostatic integration amplifying the PGF with height

The script produces PNG snapshots of the evolving fields and prints
diagnostic information (wind speed maxima) at each output frame.

# +4 K surface energy budget of ACE AIMIP models

Control and +4 K uniform-SST AIMIP rollouts (1978-10 to 2024-12, production member r1) of three
models, writing the surface flux, radiation and moisture budget terms that the AIMIP
submission configs omit. Purpose: test whether the near-surface response to +4 K SST is
consistent with the surface fluxes, and whether the energy budget holds.

| model | checkpoint dataset | ace ref |
|---|---|---|
| `ace21` | ACE2.1-ERA5 (AIMIP submission) | `ea505f1b7` |
| `ace22` | ACE2.2-ERA5 seed 3 (AIMIP submission) | `394e41b5e` |
| `ace21ec` | ACE2.1 recipe trained with the energy corrector (`ace2-1-aimip-era5-energy-corrector-*-rs0`) | `59a8a3e46` |

Outputs per run: `monthly_mean` (full rollout) and `6hourly_2000` (instantaneous snapshots for
2000, for flux calculations that need sub-monthly covariances).

Constraints:

- Each model runs at its own ref and image (`run-inference.sh`). ACE2.1's checkpoint predates
  the `secondary_decoder` restructuring and does not load on current code.
- `ace21` must use the 2024 ERA5 build's initial condition and forcing; its checkpoint checks
  the vertical coordinate against them. `ace22` and `ace21ec` use the 2026 build's.
- Every writer name must be an output of that checkpoint; `total_frozen_precipitation_rate`
  exists only for `ace21ec`.

Launch with `bash run-inference.sh`; `SMOKE=1` runs 400 steps per job.

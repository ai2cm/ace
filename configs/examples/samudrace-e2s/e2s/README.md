# SamudrACE earth2studio inference job

Single-GPU Beaker/gantry job running earth2studio's `run.deterministic`
through the `SamudrACE` prognostic wrapper with the SamudrACE CM4 piControl
checkpoint. Its outputs are compared against the vanilla fme reference run
in `../fme/`.

## What it runs

- earth2studio from the fork branch `feature/samudrace-predict-paired`
  (`jpdunc23/earth2studio`), entrypoint `run_inference.py`.
- Checkpoint `samudrACE_CM4_piControl_ckpt.tar`, forcing scenario `0311`,
  initial condition `0311-01-01T00:00:00`, 24 coupled cycles (120 days; 480
  six-hourly atmosphere steps at `n_inner_steps=20`), all read from the
  Beaker artifact dataset mounted at `/samudrace-artifacts` (mirrors the
  `allenai/SamudrACE-CM4-piControl` HF repo layout).
- Everything is overridable by environment variable
  (`SAMUDRACE_N_COUPLED_CYCLES`, `SAMUDRACE_SCENARIO`, `SAMUDRACE_IC_TIME`,
  `SAMUDRACE_ARTIFACTS`, `SAMUDRACE_OUTPUT_DIR`, `SAMUDRACE_DEVICE`,
  `SAMUDRACE_ALLOW_CPU`), which is how the payload is smoke-tested for one
  cycle on CPU (`SAMUDRACE_DEVICE=cpu SAMUDRACE_ALLOW_CPU=1`).

## No silent CPU fallback

`SAMUDRACE_DEVICE` now defaults to `cuda`, not "cuda if available". If CUDA is
unavailable (or `SAMUDRACE_DEVICE=cpu`), the payload raises before loading the
checkpoint unless `SAMUDRACE_ALLOW_CPU=1` is set; the error names the torch
version and the CUDA version it was built for. This is deliberate: the old
default quietly degraded to CPU when `torch.cuda.is_available()` returned
False (see "GPU wheel pin" below).

## GPU wheel pin

`earth2studio` only requires `torch>=2.5.0`, which resolves on PyPI to torch
2.13 — a **CUDA 13** build (it pulls `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`,
etc.). The ai2 A100 nodes run driver 570.124.06 / CUDA 12.8, and CUDA 13
requires driver r580+, so `torch.cuda.is_available()` is False on an otherwise
perfectly healthy GPU node. `submit.sh` therefore installs
`torch==2.9.1` from `https://download.pytorch.org/whl/cu128` before
`pip install '.[samudrace]'` (2.9.1 satisfies both earth2studio's `>=2.5.0`
and fme's `>=2.4.0`, so the project install leaves it alone), and the install
step ends by printing `torch.__version__`, `torch.version.cuda`, and
`torch.cuda.is_available()` so the build log records the outcome.

## cuDNN autotuning

`run_inference.py` sets `torch.backends.cudnn.benchmark = True` on GPU,
matching what fme's inference entrypoint
(`fme/coupled/inference/inference.py`) does. earth2studio calls the stepper
directly and never runs that entrypoint, so the flag would otherwise keep its
torch default of `False`, and cuDNN would pick convolution algorithms by
heuristic instead of by timing. Comparing the two 24-cycle runs showed the
consequence: the SFNO atmosphere matched the reference bit-for-bit through a
whole coupled window (its convolutions are all 1x1 and dispatch as GEMMs),
while the dilated convolutions in the Samudra ocean network landed on a
different algorithm and differed by ~1.5e-3 K rms in sst at the first coupled
step -- a seed that then amplifies chaotically and saturates by ~day 40.

## Local paths, not HuggingFace

`run_inference.py` reads every model input from the artifact directory:

- Checkpoint: `Package(/samudrace-artifacts)` — earth2studio's `Package`
  resolves a plain directory through the local filesystem, so
  `SamudrACE.load_model` loads the tar without touching `hf://`. The
  commit-pinned `load_default_package` is deliberately not used.
- Initial conditions and forcing: `LocalSamudrACEData` /
  `LocalSamudrACEForcingData`, thin subclasses of the branch's data sources
  that override the file-fetch hook to resolve repository-relative paths
  against the artifact root (the mount mirrors the repo layout exactly, so
  the same relative paths work) and raise if a file is missing.
- `HF_HUB_OFFLINE=1` is set on the job as a backstop.

## Outputs

`/results/samudrace_forecast.zarr` (gantry's default Beaker result-dataset
directory), one zarr array per variable with dims
`(time, lead_time, lat, lon)`:

| earth2studio name | FME name | field |
| --- | --- | --- |
| `t2m` | `TMP2m` | atmosphere 2 m temperature |
| `u10m` | `UGRD10m` | atmosphere 10 m eastward wind |
| `v10m` | `VGRD10m` | atmosphere 10 m northward wind |
| `skt` | `surface_temperature` | atmosphere surface temperature |
| `sst` | `sst` | ocean sea surface temperature |
| `siconc` | `ocean_sea_ice_fraction` | ocean sea-ice fraction |

`lead_time` covers 0 to 480 steps of 6 h (`timedelta64[s]`); `time` is the
initial-condition timestamp (`datetime64[s]`), so valid times are
`time + lead_time`. Latitude is north-to-south, matching the earth2studio
convention (the fme reference output from `../fme/` is south-to-north and needs flipping at
comparison time). Atmosphere fields update every step; ocean fields update at
cycle boundaries (every 20th step) and are held constant in between, so the
ocean series is directly comparable with the reference run's 5-daily ocean
output.

## Time precision

CM4 model years (e.g. 311) overflow `datetime64[ns]`, which the stock workflow
helpers `to_time_array` and `fetch_data` cast to unconditionally — year 311
silently wraps to 2064. The payload therefore substitutes second-precision
equivalents into the `earth2studio.run` namespace before calling
`run.deterministic`; nothing else in the workflow needs changing. The zarr
backend itself is fine: it creates coordinate arrays with the incoming numpy
dtype, and `datetime64[s]` round-trips through zarr and xarray unchanged
(verified).

## Environment / install

No `--beaker-image`: gantry installs from the cloned repo with
`--system-python` and a pip `--install` command (see "GPU wheel pin" above for
the torch pin that precedes `pip install '.[samudrace]'`).

- The repo is uv-managed, but the `samudrace` extra is plain PEP 621 metadata
  resolvable by pip — `fme==2026.4.0`, `pandas`, `scipy`, plus base deps, all
  PyPI releases, with no `[tool.uv.sources]` git pins in play
  (torch-harmonics comes from fme's own pin).
- Pip keeps the flags identical in shape to the reference-side job in
  `../fme/`.
- Fallback if pip resolution diverges from `uv.lock` on the GPU image:
  `--install "uv sync --frozen --extra samudrace"` with the entrypoint
  prefixed by `uv run`.

## Payload delivery

`run_inference.py` is job tooling rather than library code, and gantry ships
only what is on the remote at the checked-out ref, so `submit.sh`
base64-encodes it into `RUN_INFERENCE_B64` and the entrypoint decodes it to
`/tmp/run_inference.py`. This keeps the branch diff limited to the wrapper,
lexicon, and data sources, and lets the payload be edited without another
push.

## Prerequisites

- The branch `feature/samudrace-predict-paired` must be pushed to
  `jpdunc23/earth2studio` before submission — `submit.sh` clones it from the
  remote and will fail otherwise.
- `SAMUDRACE_ARTIFACTS_DATASET`: the Beaker dataset with the checkpoint,
  forcing, and initial conditions. Defaults in `submit.sh` to the uploaded
  dataset [`01KYQZBGSVF220C1QGMHP08GFT`](https://beaker.org/orgs/ai2/workspaces/ace/datasets/01KYQZBGSVF220C1QGMHP08GFT);
  set the env var to override.

## Preemption

The job is submitted with `--min-runtime 2h` (gantry >= 3.5) rather than
`--preemptible`. `--preemptible/--not-preemptible` is deprecated in gantry
3.7.0 and is rejected if combined with `--min-runtime`, so it is removed
entirely. `--min-runtime` guarantees the job runs for at least the given
duration before Beaker may preempt it; `--priority high` is unchanged and
orthogonal.

## Expected runtime

Environment build ~10-20 min (fme plus CUDA torch wheels). One coupled cycle
takes ~3.5 min on CPU and seconds on an A100, so 24 cycles is minutes of GPU
time. Budget well under an hour end to end.

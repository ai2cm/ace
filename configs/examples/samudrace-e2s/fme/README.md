# SamudrACE fme reference inference job (ace v2026.4.0)

Single-GPU Beaker/gantry job running vanilla `fme.coupled.inference` with the
SamudrACE CM4 piControl checkpoint, as the reference against which the
earth2studio SamudrACE port (`../e2s/`) is validated.

## What it runs

- ace at tag `v2026.4.0`, entrypoint `python -m fme.coupled.inference`
  (single GPU, so no `torchrun`).
- Config: `inference-config.yaml` — checkpoint
  `samudrACE_CM4_piControl_ckpt.tar`, forcing scenario `0311`, initial
  condition `0311-01-01T00:00:00`, 24 coupled cycles (120 days; 480
  six-hourly atmosphere steps), read from the Beaker artifact dataset
  mounted at `/samudrace-artifacts` (mirrors the
  `allenai/SamudrACE-CM4-piControl` HF repo layout).
- The config validates against the tag's schema:
  `python -m fme.coupled.validate_config inference-config.yaml --config_type inference`
  (pure schema check; it does not resolve paths, so the mount paths are fine
  locally). `submit.sh` re-runs this check when `fme` is importable.

## Outputs

`experiment_dir: /results`, gantry's default Beaker result-dataset directory,
so all outputs land in the job's result dataset. The prediction writers save
per-step NetCDF for atmosphere (6-hourly) and ocean (cycle boundaries, i.e.
5-daily) with full coordinates and timestamps; monthly files are also saved.
Comparison fields covered (plus everything else the checkpoint predicts):

- atmosphere 2m temperature
- atmosphere 10m winds
- atmosphere surface temperature
- ocean SST
- ocean sea-ice fraction

## Environment / install

No `--beaker-image`: gantry installs the environment from the cloned repo at
the tag with `--system-python` and a pip `--install` command. The tag's
`pyproject.toml` is a plain setuptools package whose deps install cleanly from
PyPI.

**GPU wheel pin.** fme's `torch>=2.4.0` resolves on PyPI to torch 2.13, which
is a **CUDA 13** build (`nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, ...). The ai2
A100 nodes run driver 570.124.06 / CUDA 12.8 and CUDA 13 needs r580+, so
`torch.cuda.is_available()` is False there. `submit.sh` therefore installs
`torch==2.9.1` from `https://download.pytorch.org/whl/cu128` before
`pip install .` (2.9.1 satisfies `torch>=2.4.0`, so the project install leaves
it in place), and the install step prints `torch.__version__`,
`torch.version.cuda`, and `torch.cuda.is_available()` into the build log.

**No silent CPU fallback.** The entrypoint asserts `torch.cuda.is_available()`
before decoding the config and starting inference, and exits with a loud
message naming the torch/CUDA versions otherwise. Set `SAMUDRACE_ALLOW_CPU=1`
on the job to override (smoke tests only).

## Preemption

The job is submitted with `--min-runtime 2h` rather than `--preemptible`.
`--preemptible/--not-preemptible` is deprecated in gantry 3.7.0 and is
rejected if combined with `--min-runtime`, so it is removed entirely.
`--min-runtime` guarantees at least that much runtime before Beaker may
preempt the job; `--priority high` is unchanged and orthogonal.

## Config delivery

The config is not part of the ace repo, and gantry ships only what is on the
remote at the checked-out ref, so `submit.sh` base64-encodes
`inference-config.yaml` into the `INFERENCE_CONFIG_B64` env var and the job
entrypoint decodes it to `/tmp/inference-config.yaml` before running.

## Prerequisites

- `SAMUDRACE_ARTIFACTS_DATASET`: the Beaker dataset with the checkpoint,
  forcing, and initial conditions. Defaults in `submit.sh` to the uploaded
  dataset [`01KYQZBGSVF220C1QGMHP08GFT`](https://beaker.org/orgs/ai2/workspaces/ace/datasets/01KYQZBGSVF220C1QGMHP08GFT);
  set the env var to override.

## Expected runtime

Environment build ~10-15 min; 24 coupled cycles of SamudrACE on a single
A100 is minutes, not hours. Budget well under an hour end to end.

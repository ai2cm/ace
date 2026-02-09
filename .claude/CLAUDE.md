# ACE: AI2 Climate Emulator

## Project Overview

ACE (AI2 Climate Emulator) is a fast machine learning climate model that simulates global atmospheric variability in a changing climate over timescales from hours to centuries. It's a PyTorch-based emulator trained on physics-based climate model outputs (ERA5, SHiELD, E3SM).

## Technology Stack

- **Python:** 3.11+ (required)
- **PyTorch:** 2.0+
- **Key Libraries:** xarray, zarr, netCDF4, wandb, omegaconf
- **Optional:** healpix (requires special compilation), graphcast dependencies
- **Build Tools:** uv (fast pip replacement), conda
- **Critical Constraint:** `torch-harmonics==0.8.0` is pinned (uses private API)

## Project Structure

```
fme/
├── core/           # Core functionality (data loading, distributed training, metrics)
├── ace/            # ACE model implementations (stepper, inference, models)
├── coupled/        # Coupled model components (ocean, prescriber, radiation)
├── downscaling/    # Downscaling models and modules
├── diffusion/      # Diffusion models
└── sht_fix.py      # Spherical harmonic transform fixes

configs/            # YAML configuration files for training/inference
scripts/            # Data processing, evaluation, and utility scripts
docs/               # Sphinx documentation
```

**Vendorized Code (DO NOT MODIFY):**
- `fme/ace/models/makani_fcn3/` - From NVIDIA Makani project
- `fme/downscaling/modules/physicsnemo_unets_v2/` - From PhysicsNeMo
- `fme/core/cuhpx/` - Custom HEALPix implementation

These directories are excluded from ruff linting and mypy checking.

## Common Commands

**Testing:**
```bash
make test              # Run all tests
make test_fast         # Skip slow tests (--fast)
make test_very_fast    # Run only very fast tests (<5s, --very-fast)
make test_cov          # Run with coverage report
pytest --no-timeout    # Disable test timeouts (useful for debugging)
```

**Environment Setup:**
```bash
make create_environment  # Create conda env with all dependencies
# Installs: base deps, dev deps, docs deps, graphcast, healpix, analysis deps
```

**Docker/Container:**
```bash
make build_docker_image       # Build production image
make enter_docker_image       # Enter interactive shell
make push_shifter_image       # Push to NERSC registry
make build_beaker_image       # Build Beaker image for AI2
```

**Code Quality:**
```bash
ruff check .           # Lint code
ruff format .          # Format code
mypy fme/              # Type checking
pre-commit run --all-files  # Run all pre-commit hooks
```

## Testing Conventions

**Test Speed Tiers:**
- **Very Fast:** < 5 seconds (use `--very-fast`)
- **Fast:** < 60 seconds (skip with `--fast`)
- **Full:** All tests (default)

**Timeout Enforcement:**
- Very fast tests: 5s timeout (auto-enforced when `--very-fast`)
- Fast tests: 60s timeout (auto-enforced when `--fast`)
- Full tests: 180s timeout (default)
- Disabled with `--pdb` or `--no-timeout` flags
- Implemented via signal.alarm() in conftest.py

**Test Organization:**
- Tests are co-located with source code (`test_*.py` files)
- Use `skip_slow` fixture to mark slow tests
- Coverage configured to omit `test_*.py` files

## Code Quality Standards

**Linting & Formatting (Ruff):**
- Enabled: D (docstrings), E (errors), F (pyflakes), I (isort), W (warnings), UP (pyupgrade)
- Ignored: D1 (missing docstrings in some cases), E203, W293
- Convention: Google-style docstrings
- Exclusions: vendorized code, `__init__.py` allows F401 (unused imports)

**Type Checking (MyPy):**
- Errors ignored for `fme.downscaling.modules.physicsnemo_unets_v2.*`
- Use type hints for new code

**Pre-commit Hooks:**
- Ruff formatting and linting
- MyPy type checking
- Trailing whitespace, end-of-file fixes
- YAML/JSON validation

**Docstrings:**
- Use Google-style format
- Required for public APIs
- Not required for tests or scripts

## Development Patterns

**Configuration Management:**
- YAML configs in `configs/` directory
- Parsed with omegaconf (supports variable interpolation)
- Configs are hierarchical: base + overrides

**Data Handling:**
- Primary formats: zarr
- Libraries: xarray for multi-dimensional arrays

**Distributed Training:**
- Uses PyTorch DistributedDataParallel (DDP)
- fme.core.distributed for utilities
- Supports multi-node GPU training

**Logging & Tracking:**
- Weights & Biases (wandb) for experiment tracking
- Structured logging throughout

**GPU/Device Management:**
- `fme.get_device()` for device selection
- Meta tensors for testing without GPU (`--meta-get-device`)
- Tests can run on CPU

## Critical Constraints & Gotchas

**Python Version:**
- Minimum: Python 3.11
- Some features use newer syntax/libraries

**Dependency Pinning:**
- `torch-harmonics==0.8.0` is pinned (uses private API that may break)
- `constraints.txt` provides upper bounds for compatibility

**HEALPix (Optional):**
- Requires special compilation (C++ extensions)
- Optional dependency: code should work without it
- Install via: `uv pip install --no-build-isolation -c constraints.txt -r requirements-healpix.txt`

**Spherical Harmonics:**
- `fme/sht_fix.py` contains patches for torch-harmonics
- Critical for certain model architectures

**Memory Management:**
- Climate data is large; be mindful of memory usage
- Tests use `gc.collect()` and `torch.cuda.empty_cache()` between runs

**Git History:**
- Recent breaking history change (see MIGRATION.md)
- Main branch has been rewritten for open development

## Documentation

- **Full Docs:** [ReadTheDocs](https://ai2-climate-emulator.readthedocs.io/)
- **Quickstart:** [Guide](https://ai2-climate-emulator.readthedocs.io/en/latest/quickstart.html)
- **API Reference:** Generated from docstrings via Sphinx

## Quick Reference

**Install from PyPI:**
```bash
pip install fme
```

**Install for Development:**
```bash
# Using make (recommended)
make create_environment

# Or manually with uv
conda create -n fme python=3.11 pip
conda activate fme
pip install uv
uv pip install -c constraints.txt -e .[dev,docs,analysis]
```

**Run Tests:**
```bash
pytest --very-fast    # Quick smoke tests
pytest --fast         # Most tests
pytest                # All tests (can be slow)
```

**Format & Lint:**
```bash
ruff format .
ruff check . --fix
mypy fme/
```

**Build Documentation:**
```bash
cd docs
make html
```

## Claude Code Skills

Custom skills are available in `.claude/commands/` for common workflows:

**PR Review:**
```bash
/review-pr 123           # Comprehensive code review of PR #123
Review PR #456           # Natural language trigger
```

**PR Re-review:**
```bash
/rereview-pr 123         # Check if review comments addressed
Re-review PR #456 from commit abc1234  # Delta review from specific commit
```

**Prerequisites:**
- Install GitHub CLI: `brew install gh` (macOS) or `conda install -c conda-forge gh`
- Authenticate: `gh auth login`

See `.claude/commands/README.md` for detailed skill documentation.

## Beaker Integration (AI2 Internal)

- Workspace: `ai2/ace`
- Used for running experiments on AI2's compute cluster
- Image building: `make build_beaker_image`
- Launch session: `make launch_beaker_session`

## Container Targets

Docker builds have multiple stages:
- `deps-only`: Just dependencies (for CI caching)
- `production`: Full production image
- `nsight`: NVIDIA Nsight profiling tools included

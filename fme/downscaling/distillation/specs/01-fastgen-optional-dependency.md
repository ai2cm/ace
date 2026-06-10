# 01 — Install FastGen as an optional pip extra; drop the submodule

## Goal

`pip install fme[distillation]` (or the conda-env equivalent) provides a
working distillation environment.  The `FastGen/` git submodule, the
`PYTHONPATH` instructions in `fme/downscaling/distillation/README.md`, and
the `COPY FastGen/` Docker layer are removed.  `import
fme.downscaling.distillation` (the package itself) must not require fastgen;
modules that need it fail with an actionable error.

## Verified current state

- `FastGen` is a git submodule (see `.gitmodules`) pinned to
  `NVlabs/FastGen @ 123e6a2f92d5c851403b75ad6cb5ee4337c88e3c`, with **no
  local modifications** — replacing it with a pip install of the same commit
  is lossless.
- `docker/Dockerfile` has a `distillation` stage (after `deps-only`):

  ```dockerfile
  FROM deps-only AS distillation
  COPY FastGen/ /tmp/fastgen/
  # annotators/ is git-ignored in FastGen repo so has no __init__.py; ...
  RUN touch /tmp/fastgen/fastgen/third_party/annotators/__init__.py && \
      uv pip install /tmp/fastgen/ --system && rm -rf /tmp/fastgen/
  ```

  The `annotators/__init__.py` hack exists because that dir is git-ignored
  upstream; check whether a `pip install git+https://...` of the pinned
  commit hits the same problem (it should not — the ignored dir won't exist
  in a fresh clone, so `find_packages()` simply won't see it; verify the
  package imports without it).
- `pyproject.toml` (repo root) already has extras: `dev`, `docs`, `deploy`,
  `graphcast` under `[project.optional-dependencies]`.  Add `distillation`.
- FastGen is not on PyPI (assume; verify).  Pin as a direct reference:
  `fastgen @ git+https://github.com/NVlabs/FastGen@123e6a2f92d5c851403b75ad6cb5ee4337c88e3c`.
  Note PyPI rejects direct references if `fme` is ever published with this
  extra — acceptable for now since `deploy` builds should exclude it; flag
  in the PR if `python -m build` complains.
- FastGen's own install requirements may be incomplete; the spike README
  lists `hydra-core`, `boto3`, `torchvision` as "FastGen extras not in fme
  env", and import probing showed `loguru`, `ftfy`, `diffusers`, `imageio`
  are also required at import time.  Whatever `pip install fastgen@...`
  does not pull in transitively must be added to the extra explicitly.

## Import-guard design

The package `__init__.py` (currently 2 lines) must stay importable without
fastgen.  Modules importing fastgen at top level today:
`fastgen_teacher.py` (`fastgen.networks.network`,
`fastgen.networks.noise_schedule`), `fastgen_train.py` (deferred to inside
`main()` already, by design — keep that), `best_student_callback.py` and
`fastgen_loader.py` (check their imports; `fastgen_loader.py` imports
fastgen only inside a method).  Pattern to apply where needed:

```python
try:
    from fastgen.networks.network import FastGenNetwork
except ImportError as e:
    raise ImportError(
        "fme.downscaling.distillation requires the 'distillation' extra: "
        "pip install fme[distillation]"
    ) from e
```

Do NOT make classes conditionally defined (no `if FASTGEN_AVAILABLE:`
class definitions) — fail loudly at import of the *submodule*, keep the
package root clean.

## Steps

1. Add the `distillation` extra to `pyproject.toml` with the pinned git ref
   plus any missing transitive deps (verify by `pip install` into a fresh
   venv and `python -c "import fastgen.trainer"`).
2. Re-point `docker/Dockerfile`'s `distillation` stage at the extra
   (`uv pip install --system "fme[distillation] @ ..."` or install the git
   ref directly); delete the `COPY FastGen/` + `touch` lines.
3. Remove the submodule: `git submodule deinit FastGen`, `git rm FastGen`,
   delete the `.gitmodules` entry.
4. Add import guards per above; update
   `fme/downscaling/distillation/README.md` Prerequisites section (remove
   submodule + PYTHONPATH instructions; replace with the extra).
5. Update `make create_environment` / environment files if they are expected
   to provide distillation deps (decide: probably not — keep the extra
   opt-in; document).

## Acceptance criteria

- Fresh env: `pip install -e .[distillation]` →
  `python -m fme.downscaling.distillation.fastgen_train --help` works.
- Env *without* the extra: `import fme.downscaling` and the full
  non-distillation test suite work; importing
  `fme.downscaling.distillation.fastgen_teacher` raises the actionable
  ImportError.
- `make build_docker_image` (distillation target) succeeds with the
  submodule gone.
- No references to `FastGen/`, `git submodule`, or `PYTHONPATH` remain in
  distillation docs.

## Out of scope

Upgrading the FastGen pin (stay on `123e6a2`); test-suite gating (spec 06).

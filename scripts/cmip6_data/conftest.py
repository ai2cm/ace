"""Pytest configuration for ``scripts/cmip6_data/``.

The cmip6_data tests rely on packages that aren't part of the main
``fme`` conda environment (``cf_xarray``, ``xesmf``, ``dask``,
``bottleneck``). CI runs ``pytest .`` from the repo root, which
discovers tests here but executes them in the ``fme`` env, producing
spurious ``ModuleNotFoundError`` / ``chunk manager 'dask' not
available`` failures.

This conftest skips collection of every test under this directory
unless all the required imports succeed, so the fme test job stays
green while cmip6_data tests still run under the cmip6_data env.
"""

import importlib

import pytest

_REQUIRED = ("cf_xarray", "xesmf", "dask", "bottleneck")
_MISSING = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]

collect_ignore_glob: list[str] = []
if _MISSING:
    pytest.skip(
        f"cmip6_data tests require {_REQUIRED}; missing: {_MISSING}. "
        "Install via ``scripts/cmip6_data/requirements.txt`` (or the "
        "cmip6-processing docker image) to run them.",
        allow_module_level=True,
    )

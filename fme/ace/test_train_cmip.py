"""Integration tests for CMIP6-style multi-dataset training.

Held in a separate file from ``test_train.py`` so the substantial
multi-source fixtures here don't conflict with the canonical
single-dataset training tests when they merge with ``main``.

Covered:

- End-to-end training through ``train_main`` against a synthetic
  ``Cmip6DataConfig`` cohort with heterogeneous variable coverage
  (the key v2-cohort property: ``allow_missing_variables=True`` on
  the step lets each dataset contribute the variables it has and
  the network masks the rest).
- Cohort-norm and per-source-norm variants — the two configs that
  live in ``configs/experiments/2026-06-01-cmip6-daily-multimodel/``.
"""

import dataclasses
import os
import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fme.ace.data_loading.cmip6 import Cmip6DataConfig
from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.step.cmip6 import Cmip6StepConfig
from fme.ace.stepper.single_module import StepperConfig, TrainStepperConfig
from fme.ace.train.train import run_train_from_config
from fme.ace.train.train_config import InlineValidationConfig, TrainConfig
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.loss import StepLossConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.optimization import OptimizationConfig
from fme.core.per_source_normalizer import PerSourceNormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.step import StepSelector

# Universal variables every synthetic CMIP6 dataset carries. Mirrors
# the v2 cohort's "always present" set — surface met + a couple of
# plev levels + statics + the log-CO2 forcing.
_UNIVERSAL_VARS = (
    "HGTsfc",
    "land_fraction",
    "log_input4mips_co2",
    "TMP2m",
    "PRATEsfc",
    "psl",
    "hus250",
    "hus700",
)

# Sub-universal — present on some datasets, missing on others.
# Exercises the ``allow_missing_variables`` path end-to-end.
_SUB_UNIVERSAL_VARS = (
    "eday_ts",
    "sfcWind",
    "DLWRFsfc",
)

# Input-only fields (statics + exogenous forcings), mirroring the
# training_run_1 §Variables policy.
_INPUT_ONLY = ("HGTsfc", "land_fraction", "log_input4mips_co2")


def _make_cmip6_zarr(
    zarr_path: pathlib.Path,
    varnames: tuple[str, ...],
    *,
    n_time: int = 8,
    n_lat: int = 4,
    n_lon: int = 8,
) -> None:
    """Write a tiny CMIP6-style zarr containing ``varnames``.

    All variables share the ``(time, lat, lon)`` dim convention used
    in the v2 cohort post-pipeline (statics + forcings are stamped to
    the daily axis at ingest time). Values are deterministic so the
    network sees enough signal to compute a non-degenerate loss.
    """
    times = xr.date_range(
        "2010-01-01", periods=n_time, freq="D", use_cftime=True, calendar="noleap"
    )
    rng = np.random.default_rng(seed=hash(zarr_path.as_posix()) & 0xFFFF_FFFF)
    coords = {
        "time": times,
        "lat": np.linspace(-60, 60, n_lat),
        "lon": np.linspace(0, 315, n_lon),
    }
    data_vars: dict[str, xr.DataArray] = {}
    for name in varnames:
        # Different physical scales per variable so the normalizer
        # has work to do.
        if name == "psl":
            arr = rng.normal(101325, 100, size=(n_time, n_lat, n_lon))
        elif name == "PRATEsfc":
            arr = np.abs(rng.normal(3e-5, 1e-5, size=(n_time, n_lat, n_lon)))
        elif name in ("hus250", "hus700"):
            arr = np.abs(rng.normal(5e-3, 1e-3, size=(n_time, n_lat, n_lon)))
        elif name == "land_fraction":
            arr = rng.uniform(0, 1, size=(n_time, n_lat, n_lon))
        elif name == "HGTsfc":
            arr = rng.uniform(0, 3000, size=(n_time, n_lat, n_lon))
        elif name == "log_input4mips_co2":
            arr = np.full((n_time, n_lat, n_lon), np.log(400.0))
        else:  # TMP2m, eday_ts, sfcWind, DLWRFsfc, ...
            arr = rng.normal(280, 5, size=(n_time, n_lat, n_lon))
        data_vars[name] = xr.DataArray(
            arr.astype(np.float32), dims=("time", "lat", "lon"), coords=coords
        )
    ds = xr.Dataset(data_vars, coords=coords)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(zarr_path, mode="w", consolidated=True, zarr_format=3)


def _make_cmip6_index(data_dir: pathlib.Path, rows: list[dict]) -> None:
    """Write ``index.csv`` covering the synthetic dataset rows.

    Schema mirrors ``rebuild_index.py`` / the real v2 cohort: one row
    per ``(source_id, experiment, variant_label)`` with status=ok and
    a label used by ``Cmip6DataConfig`` for label-conditional builders
    and per-source normalization.
    """
    records = []
    for row in rows:
        records.append(
            {
                "source_id": row["source_id"],
                "experiment": row["experiment"],
                "variant_label": row["variant_label"],
                "variant_r": row.get("variant_r", 1),
                "variant_i": 1,
                "variant_p": 1,
                "variant_f": 1,
                "label": row["source_id"] + ".p1",
                "output_zarr": os.path.join(
                    str(data_dir),
                    row["source_id"],
                    row["experiment"],
                    row["variant_label"],
                    "data.zarr",
                ),
                "status": "ok",
                "skip_reason": "",
            }
        )
    pd.DataFrame(records).to_csv(data_dir / "index.csv", index=False)


@pytest.fixture
def cmip6_cohort_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Three synthetic CMIP6 datasets with **heterogeneous variable
    coverage** — the key v2-cohort property the training pipeline
    has to handle.

    Layout:

    - ``ModelA``: every universal + every sub-universal variable.
    - ``ModelB``: universal + ``eday_ts`` only (no sfcWind, no DLWRFsfc).
    - ``ModelC``: universal only (no sub-universal).

    Together they exercise the per-sample variable-masking path: a
    single training run sees batches that have different sets of
    present variables, and ``allow_missing_variables=True`` on the
    builder lets the network treat missing channels as masked rather
    than dropping the dataset.
    """
    data_dir = tmp_path / "cmip6"
    data_dir.mkdir()
    rows = [
        {
            "source_id": "ModelA",
            "experiment": "historical",
            "variant_label": "r1i1p1f1",
            "vars": _UNIVERSAL_VARS + _SUB_UNIVERSAL_VARS,
        },
        {
            "source_id": "ModelB",
            "experiment": "historical",
            "variant_label": "r1i1p1f1",
            "vars": _UNIVERSAL_VARS + ("eday_ts",),
        },
        {
            "source_id": "ModelC",
            "experiment": "historical",
            "variant_label": "r1i1p1f1",
            "vars": _UNIVERSAL_VARS,
        },
    ]
    _make_cmip6_index(data_dir, rows)
    for row in rows:
        source_id = str(row["source_id"])
        experiment = str(row["experiment"])
        variant = str(row["variant_label"])
        zarr_path = data_dir / source_id / experiment / variant / "data.zarr"
        _make_cmip6_zarr(zarr_path, tuple(row["vars"]))
    return data_dir


def _explicit_normalizer(varnames: tuple[str, ...]) -> NormalizationConfig:
    """Cheap explicit normalizer (means / stds dicts) so the test
    doesn't need to ship netCDF stats files. Uses approximately the
    physical scale for each variable so normalized values land in
    sensible ranges.
    """
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in varnames:
        if name == "psl":
            means[name], stds[name] = 101325.0, 100.0
        elif name == "PRATEsfc":
            means[name], stds[name] = 3e-5, 1e-5
        elif name in ("hus250", "hus700"):
            means[name], stds[name] = 5e-3, 1e-3
        elif name == "land_fraction":
            means[name], stds[name] = 0.5, 0.3
        elif name == "HGTsfc":
            means[name], stds[name] = 1500.0, 1000.0
        elif name == "log_input4mips_co2":
            means[name], stds[name] = np.log(400.0), 0.1
        else:
            means[name], stds[name] = 280.0, 5.0
    # fill_nans_on_normalize=True is required for the
    # allow_missing_variables path — missing-variable samples come
    # through as NaN slabs and would otherwise produce NaN-valued
    # loss tensors before the per-sample mask is applied.
    return NormalizationConfig(means=means, stds=stds, fill_nans_on_normalize=True)


def _write_per_source_norm_files(
    cohort_dir: pathlib.Path,
    varnames: tuple[str, ...],
    sources: tuple[str, ...],
    subdirectory: str = "per_source_normalization",
) -> None:
    """Write the per-source ``centering.nc`` + ``scaling.nc`` files
    that ``PerSourceNormalizationConfig.load()`` expects, one
    subdirectory per source-label."""
    norm_dir = cohort_dir / subdirectory
    norm_dir.mkdir(exist_ok=True)
    for source in sources:
        label = f"{source}.p1"
        label_dir = norm_dir / label
        label_dir.mkdir(exist_ok=True)
        centering = _explicit_normalizer(varnames)
        # Slightly different per-source scale to make the per-source
        # path actually do something different from the cohort path.
        scaling = NormalizationConfig(
            means={k: v + 0.01 for k, v in centering.means.items()},
            stds={k: v * 1.1 for k, v in centering.stds.items()},
        )
        means_ds = xr.Dataset({k: ((), v) for k, v in centering.means.items()})
        stds_ds = xr.Dataset({k: ((), v) for k, v in scaling.stds.items()})
        means_ds.to_netcdf(label_dir / "centering.nc")
        stds_ds.to_netcdf(label_dir / "scaling.nc")


def _build_cmip6_train_config(
    *,
    cohort_dir: pathlib.Path,
    experiment_dir: pathlib.Path,
    per_source_norm: bool = False,
) -> TrainConfig:
    """Build a tight TrainConfig hitting Cmip6DataConfig +
    Cmip6StepConfig + NoiseConditionedSFNO (conditional). One epoch,
    one forward step, tiny architecture so the test runs fast."""
    in_names = list(_UNIVERSAL_VARS + _SUB_UNIVERSAL_VARS)
    out_names = [v for v in in_names if v not in _INPUT_ONLY]
    all_names = tuple(in_names)

    if per_source_norm:
        sources = ("ModelA", "ModelB", "ModelC")
        _write_per_source_norm_files(cohort_dir, all_names, sources)
        per_source_config: PerSourceNormalizationConfig | None = (
            PerSourceNormalizationConfig(
                data_dir=str(cohort_dir),
                subdirectory="per_source_normalization",
            )
        )
    else:
        per_source_config = None

    network_norm = _explicit_normalizer(all_names)

    stepper = StepperConfig(
        step=StepSelector(
            type="cmip6",
            config=dataclasses.asdict(
                Cmip6StepConfig(
                    in_names=in_names,
                    out_names=out_names,
                    normalization=NetworkAndLossNormalizationConfig(
                        network=network_norm
                    ),
                    per_source_normalization=per_source_config,
                    builder=ModuleSelector(
                        type="NoiseConditionedSFNO",
                        conditional=True,
                        allow_missing_variables=True,
                        config={
                            "embed_dim": 12,
                            "num_layers": 2,
                            "label_embed_dim": 4,
                        },
                    ),
                    corrector=AtmosphereCorrectorConfig(conserve_dry_air=False),
                )
            ),
        ),
    )

    train_loader = DataLoaderConfig(
        dataset=Cmip6DataConfig(
            data_dir=str(cohort_dir),
            experiments=["historical"],
        ),
        batch_size=2,
        num_data_workers=0,
    )
    validation = InlineValidationConfig(
        loader=DataLoaderConfig(
            dataset=Cmip6DataConfig(
                data_dir=str(cohort_dir),
                experiments=["historical"],
            ),
            batch_size=2,
            num_data_workers=0,
        ),
    )

    return TrainConfig(
        train_loader=train_loader,
        validation=validation,
        stepper=stepper,
        optimization=OptimizationConfig(
            optimizer_type="Adam",
            lr=1e-4,
            kwargs={"weight_decay": 0.01},
        ),
        logging=LoggingConfig(
            log_to_screen=False,
            log_to_wandb=False,
            log_to_file=False,
            project="fme",
            entity="ai2cm",
        ),
        max_epochs=1,
        save_checkpoint=True,
        experiment_dir=str(experiment_dir),
        stepper_training=TrainStepperConfig(
            loss=StepLossConfig(type="MSE"),
            n_forward_steps=1,
        ),
    )


@pytest.mark.parametrize("per_source_norm", [False, True])
def test_train_cmip6_heterogeneous_cohort(
    tmp_path: pathlib.Path,
    cmip6_cohort_dir: pathlib.Path,
    per_source_norm: bool,
):
    """End-to-end: train Cmip6Step on three datasets with different
    variables present.

    Verifies the load-bearing v2-cohort property — that
    ``allow_missing_variables=True`` on the conditional NoiseSFNO
    builder lets a single training run feed batches from datasets
    with different variable subsets. Without it the data loader
    would refuse to assemble batches and training would never start.

    Parametrized over cohort-vs-per-source normalization (the two
    YAML variants under ``configs/experiments/2026-06-01-cmip6-
    daily-multimodel/``); both paths need to work.
    """
    experiment_dir = tmp_path / "results"
    config = _build_cmip6_train_config(
        cohort_dir=cmip6_cohort_dir,
        experiment_dir=experiment_dir,
        per_source_norm=per_source_norm,
    )

    run_train_from_config(config)

    # Training completed → best-checkpoint artifact must exist.
    best_ckpt = experiment_dir / "training_checkpoints" / "best_ckpt.tar"
    assert (
        best_ckpt.exists()
    ), f"best_ckpt.tar missing at {best_ckpt} — training didn't complete"

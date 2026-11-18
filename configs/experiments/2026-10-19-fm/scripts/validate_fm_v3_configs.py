"""Check the FM v3 run configs (create_fm_v3_jobs.py) on CPU, without the
stats files or the training data.

1. Every v3 config (the 12 runs and the smoke test) parses strictly into
   TrainConfig, as `python -m fme.ace.validate_config` does.
2. The LR schedule of the 120-epoch runs and of the smoke test, stepped
   through torch's SequentialLR: LR 1e-4 through the last single-step epoch,
   decreasing over the cooldown epochs and 0 at the end.
3. For each of c1/c2 x scalarloss/cellloss (the s0 config) and the smoke
   test: the config's stepper and training settings, with the stats paths
   pointed at tiny synthetic files on an 8 x 16 lat-lon grid (per-cell
   entries for every variable except the scalar HGTsfc, land_fraction,
   ocean_fraction, sea_ice_fraction and global_mean_co2, as in STATS_DIR; the
   V2_DIR files scalar), the SFNO shrunk to embed_dim 128 and 2 layers and the
   optimizer switched to Adam (FusedAdam needs CUDA), build a TrainStepper and
   train one batch of 4 samples labelled c96, era5, e3sm, xshield in the
   last 1-step epoch and one in the first 3-step epoch. It
   checks the network, grouped and loss normalizers are per-cell exactly
   where the arm says, and that the scalarloss loss normalizer equals the v2
   one (loss defaulting to the network block, residual overriding it).

With --stats-dir-local DIR (a local copy of STATS_DIR), it also checks the
real per-cell files: every variable the configs normalize is present in the
pooled files and in each group of groups/ (pinned variables excepted), with
the five scalars scalar, on one shared lat-lon grid.

Usage:
    FME_FORCE_CPU=1 python validate_fm_v3_configs.py [--stats-dir-local DIR]
"""

import argparse
import copy
import pathlib
import sys
import tempfile

import dacite
import numpy as np
import torch
import xarray as xr
import yaml

# This checkout's fme, not whichever one the environment has installed.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from create_fm_v3_jobs import (  # noqa: E402
    COOLDOWN_EPOCHS,
    GROUP_LABELS,
    ITERATIONS_PER_EPOCH,
    MAX_EPOCHS,
    SMOKE_COOLDOWN_EPOCHS,
    SMOKE_MAX_EPOCHS,
    SMOKE_SAMPLES_PER_EPOCH,
    V2_DIR,
    Run,
    iterations_per_epoch,
    runs,
    smoke_run,
)

from fme.ace.data_loading.batch_data import BatchData  # noqa: E402
from fme.ace.stepper.single_module import StepperConfig  # noqa: E402
from fme.ace.train.train_config import TrainConfig  # noqa: E402
from fme.core.coordinates import (  # noqa: E402
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
)
from fme.core.dataset_info import DatasetInfo  # noqa: E402
from fme.core.device import get_device  # noqa: E402
from fme.core.labels import BatchLabels  # noqa: E402
from fme.core.normalizer import NetworkAndLossNormalizationConfig  # noqa: E402
from fme.core.step.single_module import SingleModuleStepConfig  # noqa: E402

SCALAR_NAMES = (
    "HGTsfc",
    "land_fraction",
    "ocean_fraction",
    "sea_ice_fraction",
    "global_mean_co2",
)
FILES = ("centering", "scaling-full-field", "scaling-residual")
N_LAT, N_LON = 8, 16
LAT = np.linspace(-78.75, 78.75, N_LAT)
LON = np.arange(N_LON) * 360.0 / N_LON
N_LEVELS = 8
SAMPLE_LABELS = ("c96", "era5", "e3sm", "xshield")
N_STEPS = 3
LR = 1e-4


def load(run: Run) -> dict:
    return yaml.safe_load(run.path.read_text())


def parse(data: dict) -> TrainConfig:
    return dacite.from_dict(
        data_class=TrainConfig, data=data, config=dacite.Config(strict=True)
    )


# --- 2. LR schedule --------------------------------------------------------------


def check_schedule(cfg: dict, max_epochs: int, cooldown: int, per_epoch: int):
    scheduler_cfg = cfg["optimization"]["scheduler"]
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=LR)
    schedulers = [
        getattr(torch.optim.lr_scheduler, s["type"])(optimizer, **s["kwargs"])
        for s in scheduler_cfg["schedulers"]
    ]
    assert all(s["step_each_iteration"] for s in scheduler_cfg["schedulers"])
    sequential = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers, milestones=scheduler_cfg["milestones"]
    )
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(max_epochs * per_epoch):
        optimizer.step()
        sequential.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    cooldown_start = (max_epochs - cooldown) * per_epoch
    assert np.allclose(lrs[: cooldown_start + 1], LR), "LR not constant"
    tail = np.array(lrs[cooldown_start:])
    assert np.all(np.diff(tail) <= 1e-15), "LR not decreasing in the cooldown"
    assert abs(lrs[-1]) < 1e-12, lrs[-1]
    expected = LR * (1 - np.sqrt(np.arange(len(tail)) / (len(tail) - 1)))
    err = np.max(np.abs(tail - expected)) / LR
    epochs = {
        e: lrs[(e - 1) * per_epoch]
        for e in range(max_epochs - cooldown, max_epochs + 1)
    }
    print(
        f"  schedule: {len(lrs) - 1} iterations, LR 1e-4 for iterations "
        f"0..{cooldown_start}, cooldown over the last {len(tail) - 1}; max "
        f"deviation from 1-sqrt {err:.2%} of peak; LR at the start of epochs "
        + ", ".join(f"{e}: {lr:.3g}" for e, lr in epochs.items())
        + f", end: {lrs[-1]:.3g}"
    )


# --- 3. stepper build ------------------------------------------------------------


def _names(step_config: dict) -> list[str]:
    return sorted(set(step_config["in_names"]) | set(step_config["out_names"]))


def _write(path: pathlib.Path, names, rng, per_cell: bool, std: bool) -> None:
    data_vars: dict[str, tuple[tuple[str, ...], object]] = {}
    for name in names:
        if per_cell and name not in SCALAR_NAMES:
            value = (
                rng.uniform(0.5, 2.0, (N_LAT, N_LON))
                if std
                else rng.normal(size=(N_LAT, N_LON))
            )
            data_vars[name] = (("lat", "lon"), value)
        else:
            data_vars[name] = ((), rng.uniform(0.5, 2.0) if std else rng.normal())
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(data_vars, coords={"lat": LAT, "lon": LON}).to_netcdf(path)


def write_synthetic_stats(root: pathlib.Path, names: list[str]) -> dict[str, str]:
    """Synthetic stats roots; returns {real root: synthetic root}."""
    rng = np.random.default_rng(0)
    cell, v2 = root / "percell", root / "v2"
    for directory, per_cell in ((cell, True), (v2, False)):
        subdirs = [directory] + [
            directory / "groups" / label for label in (*GROUP_LABELS, "e3sm-filled")
        ]
        for subdir in subdirs:
            for name in FILES:
                _write(subdir / f"{name}.nc", names, rng, per_cell, name != "centering")
    return {"percell": str(cell), "v2": str(v2)}


def _replace_roots(node, stats_dir: str, roots: dict[str, str]):
    if isinstance(node, dict):
        return {k: _replace_roots(v, stats_dir, roots) for k, v in node.items()}
    if isinstance(node, list):
        return [_replace_roots(v, stats_dir, roots) for v in node]
    if isinstance(node, str) and node.startswith(stats_dir + "/"):
        return roots["percell"] + node[len(stats_dir) :]
    if isinstance(node, str) and node.startswith(V2_DIR + "/"):
        return roots["v2"] + node[len(V2_DIR) :]
    return node


def stats_dir_of(cfg: dict) -> str:
    path = cfg["stepper"]["step"]["config"]["normalization"]["network"][
        "global_means_path"
    ]
    return path.removesuffix("/centering.nc")


def dataset_info() -> DatasetInfo:
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.tensor(LAT, dtype=torch.float32),
            lon=torch.tensor(LON, dtype=torch.float32),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.linspace(0, 1000, N_LEVELS + 1),
            bk=torch.linspace(0, 1, N_LEVELS + 1),
        ),
        timestep=__import__("datetime").timedelta(days=1),
        all_labels=set(GROUP_LABELS),
    )


def batch(names: list[str], epoch: int) -> BatchData:
    device = get_device()
    n = len(SAMPLE_LABELS)
    shape = (n, N_STEPS + 1, N_LAT, N_LON)
    generator = torch.Generator().manual_seed(0)
    data = {}
    for name in names:
        value = torch.rand(shape, generator=generator)
        if name.endswith("fraction"):
            value = value / 3
        elif name.startswith(("PRES",)):
            value = 1e5 + 1e3 * value
        data[name] = value.to(device)
    index = xr.date_range("2000", freq="1D", periods=N_STEPS + 1, use_cftime=True)
    time = xr.DataArray(np.stack(n * [index]), dims=["sample", "time"])
    labels = BatchLabels(
        tensor=torch.tensor(
            [[float(lab == g) for g in GROUP_LABELS] for lab in SAMPLE_LABELS]
        ).to(device),
        names=list(GROUP_LABELS),
    )
    return BatchData.new_on_device(data=data, time=time, epoch=epoch, labels=labels)


def check_build(run: Run, tmp: pathlib.Path) -> None:
    cfg = copy.deepcopy(load(run))
    step_config = cfg["stepper"]["step"]["config"]
    names = _names(step_config)
    roots = write_synthetic_stats(tmp / run.stem, names)
    stats_dir = stats_dir_of(cfg)
    cfg = _replace_roots(cfg, stats_dir, roots)
    step_config = cfg["stepper"]["step"]["config"]
    step_config["builder"]["config"].update(embed_dim=128, num_layers=2)
    cfg["optimization"]["optimizer_type"] = "Adam"
    dacite.from_dict(StepperConfig, cfg["stepper"], config=dacite.Config(strict=True))
    config = parse(cfg)
    normalization = dacite.from_dict(
        NetworkAndLossNormalizationConfig,
        step_config["normalization"],
        config=dacite.Config(strict=True),
    )
    # A run config builds its stepper from a StepperConfig, never a checkpoint.
    assert isinstance(config.stepper, StepperConfig)
    train_stepper = config.stepper_training.get_train_stepper(
        config.stepper, dataset_info()
    )
    stepper = train_stepper._stepper
    step = stepper._step_obj

    network = step.normalizer
    assert network.is_per_grid_cell
    for name in SCALAR_NAMES:
        assert network.means[name].ndim == 0, name
    grouped = getattr(step, "_grouped_normalizer", None)
    if run.network == "c2":
        assert grouped is not None and grouped.is_per_grid_cell
    else:
        assert grouped is None

    # The loss normalizer the stepper's loss is built with.
    built = stepper._get_loss_normalizer()
    if run.loss == "cellloss":
        assert built.is_per_grid_cell
        assert normalization.loss.per_grid_cell_min_std_fraction == 0.1
        assert normalization.residual.per_grid_cell_min_std_fraction == 0.1
        for name in SCALAR_NAMES:
            if name in built.stds:
                assert built.stds[name].ndim == 0, name
    else:
        assert not built.is_per_grid_cell
        # This step config with the parent v2 run's normalization block, on
        # the same synthetic V2_DIR files (it has no loss block: loss = its
        # network block, overridden by its residual block).
        parent = yaml.safe_load(run.parent_config.read_text())
        v2_step = copy.deepcopy(step_config)
        v2_step["normalization"] = _replace_roots(
            parent["stepper"]["step"]["config"]["normalization"], stats_dir, roots
        )
        assert "loss" not in v2_step["normalization"]
        reference = dacite.from_dict(
            SingleModuleStepConfig, v2_step, config=dacite.Config(strict=True)
        ).get_loss_normalizer()
        assert set(built.means) == set(reference.means)
        for name in built.means:
            assert torch.equal(built.means[name], reference.means[name]), name
            assert torch.equal(built.stds[name], reference.stds[name]), name

    optimization = config.optimization.build(
        train_stepper.modules, max_epochs=config.max_epochs
    )
    multi_step_from = cfg["stepper_training"]["n_forward_steps"]["milestones"][0][
        "epoch"
    ]
    trained = []
    # Epochs are 1-based (Trainer passes epochs_trained + 1).
    for epoch, expected_steps in ((multi_step_from - 1, 1), (multi_step_from, 3)):
        train_stepper.set_epoch(epoch)
        output = train_stepper.train_on_batch(batch(names, epoch), optimization)
        loss = float(output.metrics["loss"])
        assert np.isfinite(loss), loss
        # gen_data is [sample, ensemble member, time, lat, lon], with the
        # initial condition first.
        n_steps = next(iter(output.gen_data.values())).shape[2] - 1
        assert n_steps == expected_steps, (epoch, n_steps)
        trained.append(f"epoch {epoch}: {n_steps}-step, loss {loss:.4g}")
    print(
        f"  {run.stem}: built (network per-cell, grouped "
        f"{'per-cell' if grouped is not None else 'none'}, loss "
        f"{'per-cell' if run.loss == 'cellloss' else 'scalar = v2'}); trained "
        + "; ".join(trained)
    )


# --- real stats ------------------------------------------------------------------


def check_real_stats(stats_dir: pathlib.Path) -> None:
    cfg = load(runs()[-1])  # a c2 config: has the pinned variables
    step_config = cfg["stepper"]["step"]["config"]
    names = _names(step_config)
    pinned = set(step_config["normalization"]["grouped"]["pinned_variables"])
    grids = set()
    for subdir, required in [(stats_dir, names)] + [
        (stats_dir / "groups" / label, [n for n in names if n not in pinned])
        for label in GROUP_LABELS
    ]:
        for name in FILES:
            ds = xr.open_dataset(subdir / f"{name}.nc")
            missing = sorted(set(required) - set(ds.data_vars))
            assert not missing, f"{subdir}/{name}.nc lacks {missing}"
            for var in required:
                if var in SCALAR_NAMES:
                    assert ds[var].ndim == 0, f"{subdir}/{name}.nc {var} not scalar"
                elif ds[var].ndim:
                    grids.add(tuple(ds[var].dims) + tuple(ds[var].shape))
    print(f"  real stats in {stats_dir}: all variables present; grids {grids}")
    assert len(grids) == 1, grids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir-local", type=pathlib.Path)
    args = parser.parse_args()
    all_runs = runs() + [smoke_run()]

    print("1. Strict TrainConfig parse:")
    for run in all_runs:
        parse(load(run))
    print(f"  {len(all_runs)} configs parse")

    print("2. LR schedule:")
    check_schedule(load(runs()[0]), MAX_EPOCHS, COOLDOWN_EPOCHS, ITERATIONS_PER_EPOCH)
    check_schedule(
        load(smoke_run()),
        SMOKE_MAX_EPOCHS,
        SMOKE_COOLDOWN_EPOCHS,
        iterations_per_epoch(SMOKE_SAMPLES_PER_EPOCH),
    )

    print("3. Stepper build + a 1-step and a 3-step training batch on synthetic stats:")
    builds = [r for r in runs() if r.seed == 0] + [smoke_run()]
    with tempfile.TemporaryDirectory() as tmp:
        for run in builds:
            check_build(run, pathlib.Path(tmp))

    if args.stats_dir_local is not None:
        print("4. Real per-cell stats:")
        check_real_stats(args.stats_dir_local)
    print("OK")


if __name__ == "__main__":
    main()

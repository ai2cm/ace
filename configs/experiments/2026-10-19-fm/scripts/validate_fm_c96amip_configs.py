"""Check the FM C96 AMIP-only and era5-1979 run configs
(create_fm_c96amip_jobs.py) on CPU, without the stats files or the training
data.

1. Every config (the 12 runs and the 2 smoke tests) parses strictly into
   TrainConfig, as `python -m fme.ace.validate_config` does, also with the
   `experiment_dir=$AGENT_OUTPUT_DIR` override the jobs apply. Every training,
   validation and inline inference dataset carries the arm's label (c96 /
   era5), the only label_sampling weight is that label's 1.0, the training
   data is the arm's store over 1979-1993 and 1995-2008, and best_ckpt /
   best_inference_ckpt are selected on the arm's own data alone (C96 arm:
   val_c96 and aimip_checkpoint_c96; era5-1979: val and
   aimip_checkpoint_era5, the only positive weights).
2. The LR schedule of the SFNO and Swin runs and of the smoke tests, stepped
   through torch's SequentialLR: Swin at 0.01 and 0.505 of 1e-4 over epochs
   1 and 2; then 1e-4 through the last single-step epoch, decreasing over the
   cooldown epochs and 0 at the end.
3. For the s0 config of each architecture and arm and each smoke test: with the
   stats paths pointed at tiny synthetic scalar files, the model shrunk
   (SFNO embed_dim 128 and 2 layers, Swin embed_dim 64) and the optimizer
   switched to Adam (FusedAdam needs CUDA), build a TrainStepper on the
   45 x 90 grid and train one batch of 2 samples with the arm's label in the last
   1-step epoch and one in the first 3-step epoch. It checks that the network
   normalizer is scalar with no grouped normalizer, that the loss normalizer
   uses the residual stats for the prognostic names and the full-field stats
   otherwise, and that the label encoding is the arm's one label.

Usage:
    FME_FORCE_CPU=1 python validate_fm_c96amip_configs.py
"""

import copy
import datetime
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
from create_fm_c96amip_jobs import (  # noqa: E402
    ARM,
    C96_AMIP_DATA_PATH,
    C96_SELECTION,
    COOLDOWN_EPOCHS,
    ERA5_FILE,
    ERA5_SELECTION,
    ITERATIONS_PER_EPOCH,
    MAX_EPOCHS,
    SFNO,
    SMOKE_COOLDOWN_EPOCHS,
    SMOKE_MAX_EPOCHS,
    SMOKE_SAMPLES_PER_EPOCH,
    SWIN,
    TRAIN_SPLITS,
    Run,
    iterations_per_epoch,
    runs,
    smoke_runs,
)
from submit_fm_v3_jobs import AGENT_OUTPUT_ROOT  # noqa: E402

from fme.ace.data_loading.batch_data import BatchData  # noqa: E402
from fme.ace.stepper.single_module import StepperConfig  # noqa: E402
from fme.ace.train.train_config import TrainConfig  # noqa: E402
from fme.core.config import update_dict_with_dotlist  # noqa: E402
from fme.core.coordinates import (  # noqa: E402
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
)
from fme.core.dataset_info import DatasetInfo  # noqa: E402
from fme.core.device import get_device  # noqa: E402
from fme.core.labels import BatchLabels  # noqa: E402

FILES = ("centering", "scaling-full-field", "scaling-residual")
N_LAT, N_LON = 45, 90
LAT = np.linspace(-88, 88, N_LAT)
LON = np.arange(N_LON) * 360.0 / N_LON
N_LEVELS = 8
N_SAMPLES = 2
N_STEPS = 3
LR = 1e-4
WARMUP_FACTORS = (0.01, 0.505)
SHRUNK_BUILDER = {SFNO: {"embed_dim": 128, "num_layers": 2}, SWIN: {"embed_dim": 64}}


def load(run: Run) -> dict:
    return yaml.safe_load(run.path.read_text())


def parse(data: dict) -> TrainConfig:
    return dacite.from_dict(
        data_class=TrainConfig, data=data, config=dacite.Config(strict=True)
    )


# --- 1. labels and selection -------------------------------------------------------


def check_labels_and_selection(run: Run, cfg: dict) -> None:
    label = run.label
    loader = cfg["train_loader"]
    assert loader["label_sampling"]["weights"] == {label: 1.0}
    store = (
        (C96_AMIP_DATA_PATH, "ic_0001.zarr")
        if run.arm == ARM
        else ("/climate-default", ERA5_FILE)
    )
    train = [
        (d["data_path"], d["file_pattern"], d["subset"]["start_time"])
        for d in loader["dataset"]["concat"]
    ]
    assert train == [(*store, a) for a, _ in TRAIN_SPLITS], train
    datasets = list(loader["dataset"]["concat"])
    for entry in cfg["validation"]:
        datasets += entry["loader"]["dataset"]["concat"]
    datasets += [entry["loader"]["dataset"] for entry in cfg["inference"]]
    labels = {label for d in datasets for label in d["labels"]}
    assert labels == {label}, labels
    val_weights = {e["name"]: e["weight"] for e in cfg["validation"]}
    c96_arm = run.arm == ARM
    assert val_weights == {"val": float(not c96_arm), "val_c96": float(c96_arm)}
    positive = [e["name"] for e in cfg["inference"] if e["weight"] > 0]
    assert positive == [C96_SELECTION if c96_arm else ERA5_SELECTION], positive
    assert "grouped" not in cfg["stepper"]["step"]["config"]["normalization"]


# --- 2. LR schedule ----------------------------------------------------------------


def check_schedule(
    cfg: dict, max_epochs: int, cooldown: int, per_epoch: int, warmup: bool
) -> None:
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
    constant_start = 0
    if warmup:
        for i, factor in enumerate(WARMUP_FACTORS):
            epoch = lrs[i * per_epoch : (i + 1) * per_epoch]
            assert np.allclose(epoch, factor * LR), (i + 1, epoch[0], epoch[-1])
        constant_start = len(WARMUP_FACTORS) * per_epoch
    cooldown_start = (max_epochs - cooldown) * per_epoch
    assert np.allclose(lrs[constant_start : cooldown_start + 1], LR), "not constant"
    tail = np.array(lrs[cooldown_start:])
    assert np.all(np.diff(tail) <= 1e-15), "LR not decreasing in the cooldown"
    assert abs(lrs[-1]) < 1e-12, lrs[-1]
    expected = LR * (1 - np.sqrt(np.arange(len(tail)) / (len(tail) - 1)))
    err = np.max(np.abs(tail - expected)) / LR
    epochs = {e: lrs[(e - 1) * per_epoch] for e in (1, 2, 3)} | {
        e: lrs[(e - 1) * per_epoch]
        for e in range(max(max_epochs - cooldown, 1), max_epochs + 1)
    }
    print(
        f"  {len(lrs) - 1} iterations; LR at the start of epochs "
        + ", ".join(f"{e}: {lr:.3g}" for e, lr in sorted(epochs.items()))
        + f", end: {lrs[-1]:.3g}; cooldown max deviation from 1-sqrt "
        f"{err:.2%} of peak"
    )


# --- 3. stepper build --------------------------------------------------------------


def _names(step_config: dict) -> list[str]:
    return sorted(set(step_config["in_names"]) | set(step_config["out_names"]))


def write_synthetic_stats(root: pathlib.Path, names: list[str]) -> None:
    """Scalar stats; residual stds 0.1x the full-field ones."""
    rng = np.random.default_rng(0)
    means = {n: rng.normal() for n in names}
    stds = {n: rng.uniform(0.5, 2.0) for n in names}
    values = {
        "centering": means,
        "scaling-full-field": stds,
        "scaling-residual": {n: 0.1 * s for n, s in stds.items()},
    }
    root.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        xr.Dataset({n: ((), v) for n, v in values[name].items()}).to_netcdf(
            root / f"{name}.nc"
        )


def _replace_root(node, old: str, new: str):
    if isinstance(node, dict):
        return {k: _replace_root(v, old, new) for k, v in node.items()}
    if isinstance(node, list):
        return [_replace_root(v, old, new) for v in node]
    if isinstance(node, str) and node.startswith(old + "/"):
        return new + node[len(old) :]
    return node


def dataset_info(label: str) -> DatasetInfo:
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.tensor(LAT, dtype=torch.float32),
            lon=torch.tensor(LON, dtype=torch.float32),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.linspace(0, 1000, N_LEVELS + 1),
            bk=torch.linspace(0, 1, N_LEVELS + 1),
        ),
        timestep=datetime.timedelta(days=1),
        all_labels={label},
    )


def batch(names: list[str], epoch: int, label: str) -> BatchData:
    device = get_device()
    shape = (N_SAMPLES, N_STEPS + 1, N_LAT, N_LON)
    generator = torch.Generator().manual_seed(0)
    data = {}
    for name in names:
        value = torch.rand(shape, generator=generator)
        if name.endswith("fraction"):
            value = value / 3
        elif name.startswith("PRES"):
            value = 1e5 + 1e3 * value
        data[name] = value.to(device)
    index = xr.date_range("2000", freq="1D", periods=N_STEPS + 1, use_cftime=True)
    time = xr.DataArray(np.stack(N_SAMPLES * [index]), dims=["sample", "time"])
    labels = BatchLabels(tensor=torch.ones(N_SAMPLES, 1).to(device), names=[label])
    return BatchData.new_on_device(data=data, time=time, epoch=epoch, labels=labels)


def check_build(run: Run, tmp: pathlib.Path) -> None:
    cfg = copy.deepcopy(load(run))
    step_config = cfg["stepper"]["step"]["config"]
    names = _names(step_config)
    stats_dir = step_config["normalization"]["network"]["global_means_path"]
    stats_dir = stats_dir.removesuffix("/centering.nc")
    synthetic = tmp / run.stem
    write_synthetic_stats(synthetic, names)
    cfg = _replace_root(cfg, stats_dir, str(synthetic))
    step_config = cfg["stepper"]["step"]["config"]
    step_config["builder"]["config"].update(SHRUNK_BUILDER[run.arch])
    cfg["optimization"]["optimizer_type"] = "Adam"
    dacite.from_dict(StepperConfig, cfg["stepper"], config=dacite.Config(strict=True))
    config = parse(cfg)
    assert isinstance(config.stepper, StepperConfig)
    train_stepper = config.stepper_training.get_train_stepper(
        config.stepper, dataset_info(run.label)
    )
    stepper = train_stepper._stepper
    step = stepper._step_obj

    network = step.normalizer
    assert not network.is_per_grid_cell
    assert getattr(step, "_grouped_normalizer", None) is None
    full = xr.open_dataset(synthetic / "scaling-full-field.nc")
    residual = xr.open_dataset(synthetic / "scaling-residual.nc")
    for name in names:
        assert np.isclose(float(network.stds[name]), float(full[name])), name
    loss_normalizer = stepper._get_loss_normalizer()
    prognostic = set(step_config["in_names"]) & set(step_config["out_names"])
    for name in step_config["out_names"]:
        expected = residual if name in prognostic else full
        assert np.isclose(
            float(loss_normalizer.stds[name]), float(expected[name])
        ), name
    labels = step.get_state()["module"]["label_encoding"]["labels"]
    assert labels == [run.label], labels

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
        output = train_stepper.train_on_batch(
            batch(names, epoch, run.label), optimization
        )
        loss = float(output.metrics["loss"])
        assert np.isfinite(loss), loss
        # gen_data is [sample, ensemble member, time, lat, lon], with the
        # initial condition first.
        n_steps = next(iter(output.gen_data.values())).shape[2] - 1
        assert n_steps == expected_steps, (epoch, n_steps)
        trained.append(f"epoch {epoch}: {n_steps}-step, loss {loss:.4g}")
    print(
        f"  {run.stem}: built (scalar network normalizer, no grouped, loss = "
        f"residual for {len(prognostic)} prognostics / full-field otherwise, "
        f"labels {labels}); trained " + "; ".join(trained)
    )


def main() -> None:
    all_runs = runs() + smoke_runs()
    print("1. Strict TrainConfig parse (also with the job's experiment_dir):")
    override = [f"experiment_dir={AGENT_OUTPUT_ROOT}/agent-alexeyy-example-abcd"]
    for run in all_runs:
        parse(load(run))
        overridden = parse(dict(update_dict_with_dotlist(load(run), override)))
        assert overridden.experiment_dir == override[0].split("=", 1)[1]
        check_labels_and_selection(run, load(run))
    print(
        f"  {len(all_runs)} configs parse; every dataset carries the arm's label; "
        "training data and selection on the arm's own data"
    )

    print("2. LR schedule:")
    for run in runs():
        if run.seed == 0:
            print(f" {run.stem}:")
            check_schedule(
                load(run),
                MAX_EPOCHS,
                COOLDOWN_EPOCHS,
                ITERATIONS_PER_EPOCH,
                warmup=run.arch == SWIN,
            )
    for run in smoke_runs():
        print(f" {run.stem}:")
        check_schedule(
            load(run),
            SMOKE_MAX_EPOCHS,
            SMOKE_COOLDOWN_EPOCHS,
            iterations_per_epoch(SMOKE_SAMPLES_PER_EPOCH),
            warmup=False,
        )

    print("3. Stepper build + a 1-step and a 3-step training batch:")
    builds = [r for r in runs() if r.seed == 0] + smoke_runs()
    with tempfile.TemporaryDirectory() as tmp:
        for run in builds:
            check_build(run, pathlib.Path(tmp))
    print("OK")


if __name__ == "__main__":
    main()

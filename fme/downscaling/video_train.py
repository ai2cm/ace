"""Training entry point for the endpoint-conditioned video diffusion model."""

import argparse
import contextlib
import dataclasses
import logging
import os
import shutil
import time
import uuid

import dacite
import torch
import yaml

from fme.core.cli import prepare_directory, remove_stale_tmp_checkpoints
from fme.core.device import get_device
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.ema import EMAConfig, EMATracker
from fme.core.generics.trainer import count_parameters
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import NullOptimization, OptimizationConfig
from fme.core.wandb import WandB
from fme.downscaling.aggregators.main import LossVsNoiseAggregator
from fme.downscaling.data import PairedDataLoaderConfig, PairedVideoBatchData
from fme.downscaling.video_models import (
    VideoDiffusionModelConfig,
    _linear_interp_endpoints,
)


def _video_comparison_figure(gt, pred, name: str, example_idx: int):
    """4-row (GT / Linear / Pred / Pred-GT) x T-frame grid for one var + example."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_times = gt.shape[0]
    lin = _linear_interp_endpoints(gt)
    diff = pred - gt
    vmin, vmax = float(gt.min()), float(gt.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    dmax = max(float(diff.abs().max()), 1e-6)
    rows = [
        ("GT", gt, dict(vmin=vmin, vmax=vmax, cmap="viridis")),
        ("Linear", lin, dict(vmin=vmin, vmax=vmax, cmap="viridis")),
        ("Pred", pred, dict(vmin=vmin, vmax=vmax, cmap="viridis")),
        ("Pred-GT", diff, dict(vmin=-dmax, vmax=dmax, cmap="RdBu_r")),
    ]
    fig, axes = plt.subplots(4, n_times, figsize=(1.4 * n_times, 5.8), squeeze=False)
    for r, (label, data, kw) in enumerate(rows):
        for t in range(n_times):
            ax = axes[r][t]
            ax.imshow(data[t].numpy(), origin="lower", **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                obs = "\n(obs)" if t in (0, n_times - 1) else ""
                ax.set_title(f"{3 * t}h{obs}", fontsize=7)
            if t == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(f"{name} - val example {example_idx}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _save_checkpoint(trainer: "VideoTrainer", path: str) -> None:
    temporary_location = os.path.join(os.path.dirname(path), f".{uuid.uuid4()}.tmp")
    try:
        torch.save(
            {
                "module": trainer.model.module.state_dict(),
                "ema": trainer.ema.get_state(),
                "optimization": trainer.optimization.get_state(),
                "num_batches_seen": trainer.num_batches_seen,
                "startEpoch": trainer.startEpoch,
                "best_valid_loss": trainer.best_valid_loss,
            },
            temporary_location,
        )
        os.replace(temporary_location, path)
    finally:
        if os.path.exists(temporary_location):
            os.remove(temporary_location)


def restore_checkpoint(trainer: "VideoTrainer") -> None:
    checkpoint = torch.load(
        trainer.epoch_checkpoint_path, map_location="cpu", weights_only=False
    )
    trainer.model.module.load_state_dict(checkpoint["module"])
    trainer.optimization.load_state(checkpoint["optimization"])
    trainer.num_batches_seen = checkpoint["num_batches_seen"]
    trainer.startEpoch = checkpoint["startEpoch"]
    trainer.best_valid_loss = checkpoint["best_valid_loss"]
    trainer.ema = EMATracker.from_state(checkpoint["ema"], trainer.model.modules)


@dataclasses.dataclass
class VideoTrainerConfig:
    """Configuration for the video diffusion Trainer."""

    model: VideoDiffusionModelConfig
    optimization: OptimizationConfig
    train_data: PairedDataLoaderConfig
    validation_data: PairedDataLoaderConfig
    max_epochs: int
    experiment_dir: str
    logging: LoggingConfig
    # Optional held-out test split, evaluated every test_interval epochs.
    test_data: PairedDataLoaderConfig | None = None
    test_interval: int = 10
    save_checkpoints: bool = True
    ema: EMAConfig = dataclasses.field(default_factory=EMAConfig)
    validate_using_ema: bool = False
    generate_n_samples: int = 1
    # Number of val/test examples to visualize on W&B; 0 disables.
    num_visualization_examples: int = 3
    segment_epochs: int | None = None
    validate_interval: int = 1
    resume_results_dir: str | None = None
    # Cap batches per train/val/test pass (None iterates the full loader).
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_test_batches: int | None = None
    log_loss_vs_noise: bool = False

    def __post_init__(self):
        datasets = [
            ("train_data", self.train_data),
            ("validation_data", self.validation_data),
        ]
        if self.test_data is not None:
            datasets.append(("test_data", self.test_data))
        for name, data in datasets:
            if data.n_timesteps != self.model.n_timesteps:
                raise ValueError(
                    f"{name}.n_timesteps ({data.n_timesteps}) must equal "
                    f"model.n_timesteps ({self.model.n_timesteps})."
                )

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.experiment_dir, "checkpoints")

    def build(self) -> "VideoTrainer":
        requirements = self.model.data_requirements
        train_data = self.train_data.build_video(train=True, requirements=requirements)
        validation_data = self.validation_data.build_video(
            train=False, requirements=requirements
        )
        test_data = (
            self.test_data.build_video(train=False, requirements=requirements)
            if self.test_data is not None
            else None
        )
        model = self.model.build()
        optimization = self.optimization.build(
            modules=[model.module], max_epochs=self.max_epochs
        )
        return VideoTrainer(
            model, optimization, train_data, validation_data, self, test_data
        )

    def configure_logging(self, log_filename: str):
        config = to_flat_dict(dataclasses.asdict(self))
        self.logging.configure_logging(
            self.experiment_dir, log_filename, config=config, resumable=True
        )


class VideoTrainer:
    def __init__(
        self, model, optimization, train_data, validation_data, config, test_data=None
    ):
        self.model = model
        self.optimization = optimization
        self.null_optimization = NullOptimization()
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.config = config
        self.ema = config.ema.build(self.model.modules)
        self.validate_using_ema = config.validate_using_ema
        self.num_batches_seen = 0
        self.startEpoch = 0
        self.segment_epochs = config.segment_epochs
        self.best_valid_loss = float("inf")

        wandb = WandB.get_instance()
        wandb.watch(self.model.modules)

        dist = Distributed.get_instance()
        self.epoch_checkpoint_path: str | None = None
        self.best_checkpoint_path: str | None = None
        if config.save_checkpoints:
            if dist.is_root():
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                remove_stale_tmp_checkpoints(config.checkpoint_dir)
            self.epoch_checkpoint_path = os.path.join(
                config.checkpoint_dir, "latest.ckpt"
            )
            self.best_checkpoint_path = os.path.join(config.checkpoint_dir, "best.ckpt")

    @property
    def resuming(self) -> bool:
        return self.epoch_checkpoint_path is not None and os.path.isfile(
            self.epoch_checkpoint_path
        )

    @contextlib.contextmanager
    def _ema_context(self):
        self.ema.store(parameters=self.model.modules.parameters())
        self.ema.copy_to(model=self.model.modules)
        try:
            yield
        finally:
            self.ema.restore(parameters=self.model.modules.parameters())

    @contextlib.contextmanager
    def _validation_context(self):
        if self.validate_using_ema:
            with self._ema_context():
                yield
        else:
            yield

    def train_one_epoch(self) -> None:
        self.model.module.train()
        wandb = WandB.get_instance()
        loss_vs_noise = (
            LossVsNoiseAggregator() if self.config.log_loss_vs_noise else None
        )
        batch: PairedVideoBatchData
        epoch_loss = 0.0
        n_batches = 0
        for i, batch in enumerate(self.train_data.loader):
            if (
                self.config.max_train_batches is not None
                and i >= self.config.max_train_batches
            ):
                break
            self.num_batches_seen += 1
            outputs = self.model.train_on_batch(batch, self.optimization)
            self.ema(self.model.modules)
            if loss_vs_noise is not None:
                loss_vs_noise.record_batch(outputs)
            batch_loss = outputs.loss.detach().cpu().item()
            epoch_loss += batch_loss
            n_batches += 1
            if i % 10 == 0:
                logging.info(f"Training batch {i + 1}, loss {batch_loss:.4f}")
            wandb.log({"train/batch_loss": batch_loss}, step=self.num_batches_seen)
        if n_batches == 0:
            raise RuntimeError("Empty training batch generator")
        self.optimization.step_scheduler(epoch_loss / n_batches)
        logs = {"train/epoch_loss": epoch_loss / n_batches}
        if loss_vs_noise is not None:
            logs.update(loss_vs_noise.get_wandb(prefix="train"))
        wandb.log(logs, step=self.num_batches_seen)

    @torch.no_grad()
    def valid_one_epoch(self) -> dict[str, float]:
        self.model.module.eval()
        total_loss = 0.0
        total_gen_mae = 0.0
        loss_vs_noise = (
            LossVsNoiseAggregator() if self.config.log_loss_vs_noise else None
        )
        n_batches = 0
        with self._validation_context():
            for j, batch in enumerate(self.validation_data.loader):
                if (
                    self.config.max_val_batches is not None
                    and j >= self.config.max_val_batches
                ):
                    break
                outputs = self.model.train_on_batch(batch, self.null_optimization)
                if loss_vs_noise is not None:
                    loss_vs_noise.record_batch(outputs)
                total_loss += outputs.loss.detach().cpu().item()
                total_gen_mae += self._interior_generation_mae(batch)
                n_batches += 1
        # Reduce across ranks; all ranks must reach this collective (empty check
        # uses the global count to avoid deadlocking at the WandB.log barrier).
        stats = torch.tensor(
            [total_loss, total_gen_mae, float(n_batches)], device=get_device()
        )
        Distributed.get_instance().reduce_sum(stats)
        total_loss, total_gen_mae, n_total = stats.tolist()
        if n_total == 0:
            raise RuntimeError("Empty validation batch generator")
        summary = {
            "validation/loss": total_loss / n_total,
            "validation/interior_mae": total_gen_mae / n_total,
        }
        logs = dict(summary)
        if loss_vs_noise is not None:
            logs.update(loss_vs_noise.get_wandb(prefix="validation"))
        WandB.get_instance().log(logs, step=self.num_batches_seen)
        return summary

    def _interior_generation_mae(self, batch: PairedVideoBatchData) -> float:
        """Ensemble-mean MAE over the generated interior frames."""
        generated = self.model.generate(batch, n_samples=self.config.generate_n_samples)
        n_times = self.model.n_timesteps
        errors = []
        for name, samples in generated.items():
            ens_mean = samples.mean(dim=1)
            truth = batch.fine.data[name].to(ens_mean.device)
            interior = slice(1, n_times - 1)
            errors.append(
                (ens_mean[:, interior] - truth[:, interior]).abs().mean().item()
            )
        return sum(errors) / len(errors)

    @torch.no_grad()
    def log_validation_visualizations(self) -> None:
        """Log GT-vs-prediction frame grids for a few val examples to W&B.

        Called by all ranks for the WandB.log barrier; only root logs figures.
        """
        if self.config.num_visualization_examples <= 0:
            return
        dist = Distributed.get_instance()
        wandb = WandB.get_instance()
        self.model.module.eval()
        with self._validation_context():
            batch = next(iter(self.validation_data.loader))
            generated = self.model.generate(batch, n_samples=1)
        logs: dict = {}
        if dist.is_root() and wandb.enabled:
            import matplotlib.pyplot as plt

            n = min(self.config.num_visualization_examples, len(batch.fine))
            for i in range(n):
                for name in self.model.out_names:
                    gt = batch.fine.data[name][i].float().cpu()
                    pred = generated[name][i, 0].float().cpu()
                    fig = _video_comparison_figure(gt, pred, name, i)
                    logs[f"val_viz/example_{i}/{name}"] = wandb.Image(fig)
                    plt.close(fig)
        wandb.log(logs, step=self.num_batches_seen)

    @torch.no_grad()
    def evaluate_test(self) -> None:
        """Evaluate the held-out test set: metrics + visualizations to W&B.

        Per-channel MAE/RMSE, relative error, and improvement over the linear
        baseline, summed across ranks. Called by all ranks; only root logs.
        """
        if self.test_data is None:
            return
        dist = Distributed.get_instance()
        wandb = WandB.get_instance()
        self.model.module.eval()
        names = self.model.out_names
        interior = slice(1, self.model.n_timesteps - 1)
        # per channel: [sum|err_model|, sum err_model^2, sum|err_linear|, count]
        acc = torch.zeros(len(names), 4, device=get_device())
        viz = None
        with self._validation_context():
            for b, batch in enumerate(self.test_data.loader):
                if (
                    self.config.max_test_batches is not None
                    and b >= self.config.max_test_batches
                ):
                    break
                generated = self.model.generate(
                    batch, n_samples=self.config.generate_n_samples
                )
                for c, name in enumerate(names):
                    gt = batch.fine.data[name].float()
                    pred = generated[name].float().mean(dim=1)
                    lin = _linear_interp_endpoints(gt)
                    gi, pi, li = gt[:, interior], pred[:, interior], lin[:, interior]
                    acc[c, 0] += (pi - gi).abs().sum()
                    acc[c, 1] += (pi - gi).pow(2).sum()
                    acc[c, 2] += (li - gi).abs().sum()
                    acc[c, 3] += gi.numel()
                if b == 0 and dist.is_root():
                    viz = (batch, generated)
        dist.reduce_sum(acc)

        logs: dict = {}
        for c, name in enumerate(names):
            s_abs, s_sq, s_abs_lin, count = acc[c].tolist()
            if count == 0:
                continue
            mae = s_abs / count
            rmse = (s_sq / count) ** 0.5
            mae_linear = s_abs_lin / count
            std = float(self.model.normalizer.stds[name].item())
            logs[f"test/{name}/mae"] = mae
            logs[f"test/{name}/rmse"] = rmse
            logs[f"test/{name}/rel_mae_pct"] = 100 * mae / std
            logs[f"test/{name}/rel_rmse_pct"] = 100 * rmse / std
            logs[f"test/{name}/linear_mae"] = mae_linear
            logs[f"test/{name}/mae_improvement_vs_linear_pct"] = (
                100 * (mae_linear - mae) / mae_linear if mae_linear else 0.0
            )

        if dist.is_root() and wandb.enabled and viz is not None:
            import matplotlib.pyplot as plt

            batch, generated = viz
            n = min(self.config.num_visualization_examples, len(batch.fine))
            for i in range(n):
                for name in names:
                    gt = batch.fine.data[name][i].float().cpu()
                    pred = generated[name][i].float().mean(dim=0).cpu()
                    fig = _video_comparison_figure(gt, pred, name, i)
                    logs[f"test_viz/example_{i}/{name}"] = wandb.Image(fig)
                    plt.close(fig)
        wandb.log(logs, step=self.num_batches_seen)

    def save_best_checkpoint(self, summary: dict[str, float]) -> None:
        if self.best_checkpoint_path is None:
            return
        context = self._ema_context if self.validate_using_ema else contextlib.nullcontext
        if summary["validation/loss"] < self.best_valid_loss:
            logging.info("Saving best checkpoint")
            self.best_valid_loss = summary["validation/loss"]
            with context():
                _save_checkpoint(self, self.best_checkpoint_path)

    def save_epoch_checkpoint(self) -> None:
        if self.epoch_checkpoint_path is not None:
            _save_checkpoint(self, self.epoch_checkpoint_path)

    def train(self) -> None:
        logging.info("Running initial validation.")
        self.valid_one_epoch()
        wandb = WandB.get_instance()
        dist = Distributed.get_instance()

        if self.segment_epochs is None:
            segment_max_epochs = self.config.max_epochs
        else:
            segment_max_epochs = min(
                self.startEpoch + self.segment_epochs, self.config.max_epochs
            )

        for epoch in range(self.startEpoch, segment_max_epochs):
            logging.info(f"Training epoch {epoch + 1}")
            start = time.time()
            self.train_one_epoch()
            self.startEpoch = epoch + 1
            wandb.log({"epoch": epoch}, step=self.num_batches_seen)
            if epoch % self.config.validate_interval == 0:
                summary = self.valid_one_epoch()
                if dist.is_root() and self.config.save_checkpoints:
                    self.save_best_checkpoint(summary)
                self.log_validation_visualizations()  # all ranks (barrier-safe)
            if epoch % self.config.test_interval == 0:
                self.evaluate_test()  # all ranks (barrier-safe)
            if dist.is_root() and self.config.save_checkpoints:
                self.save_epoch_checkpoint()
            wandb.log(
                {"epoch_seconds": time.time() - start}, step=self.num_batches_seen
            )


def _resume_from_results_dir_if_not_preempted(experiment_dir, resume_results_dir):
    resuming_from_preempt = os.path.isfile(
        os.path.join(experiment_dir, "checkpoints/latest.ckpt")
    )
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)
    if resume_results_dir is not None and not resuming_from_preempt:
        if not os.path.isdir(resume_results_dir):
            raise ValueError(
                f"Existing results directory {resume_results_dir} does not exist."
            )
        shutil.copytree(resume_results_dir, experiment_dir, dirs_exist_ok=True)
        remove_stale_tmp_checkpoints(os.path.join(experiment_dir, "checkpoints"))


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_config: VideoTrainerConfig = dacite.from_dict(
        data_class=VideoTrainerConfig,
        data=config,
        config=dacite.Config(strict=True),
    )

    if train_config.resume_results_dir is not None:
        _resume_from_results_dir_if_not_preempted(
            experiment_dir=train_config.experiment_dir,
            resume_results_dir=train_config.resume_results_dir,
        )
    prepare_directory(train_config.experiment_dir, config)
    train_config.configure_logging(log_filename="out.log")
    logging.info("Starting video diffusion training")
    trainer = train_config.build()
    if trainer.resuming:
        logging.info(f"Resuming training from {trainer.epoch_checkpoint_path}")
        restore_checkpoint(trainer)
    logging.info(f"Number of parameters: {count_parameters(trainer.model.modules)}")
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Video downscaling train script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Distributed.context():
        main(args.config_path)

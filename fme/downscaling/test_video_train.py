from fme.core.ema import EMAConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import OptimizationConfig
from fme.core.dataset.xarray import XarrayDataConfig
from fme.downscaling.data.config import PairedDataLoaderConfig
from fme.downscaling.data.utils import ClosedInterval
from fme.downscaling.test_utils import data_paths_helper
from fme.downscaling.video_models import VideoDiffusionModelConfig
from fme.downscaling.video_train import VideoTrainerConfig, restore_checkpoint

OUT_NAMES = ["var0", "var1"]
N_TIMESTEPS = 5


def _data_config(paths):
    return PairedDataLoaderConfig(
        fine=[XarrayDataConfig(paths.fine)],
        coarse=[XarrayDataConfig(paths.fine)],  # temporal-only: same store
        batch_size=2,
        num_data_workers=0,
        strict_ensemble=False,
        lat_extent=ClosedInterval(1, 6),
        lon_extent=ClosedInterval(0, 8),
        n_timesteps=N_TIMESTEPS,
    )


def _trainer_config(tmp_path):
    paths = data_paths_helper(tmp_path, num_timesteps=18)
    model = VideoDiffusionModelConfig(
        out_names=OUT_NAMES,
        n_timesteps=N_TIMESTEPS,
        normalization=NormalizationConfig(
            means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
        ),
        model_channels=16,
        n_heads=2,
        num_freqs=3,
        num_diffusion_generation_steps=4,
    )
    return VideoTrainerConfig(
        model=model,
        optimization=OptimizationConfig(lr=1e-3),
        train_data=_data_config(paths),
        validation_data=_data_config(paths),
        test_data=_data_config(paths),
        test_interval=1,
        max_epochs=2,
        experiment_dir=str(tmp_path / "exp"),
        logging=LoggingConfig(
            log_to_screen=False, log_to_wandb=False, log_to_file=False
        ),
        save_checkpoints=True,
        validate_using_ema=True,
        ema=EMAConfig(decay=0.99),
        generate_n_samples=1,
    )


def test_video_trainer_runs_and_checkpoints(tmp_path):
    config = _trainer_config(tmp_path)
    trainer = config.build()
    assert len(trainer.train_data.loader) > 0
    trainer.train()

    # ran all epochs, recorded a finite best validation loss, wrote checkpoints
    assert trainer.startEpoch == 2
    assert trainer.num_batches_seen > 0
    assert trainer.best_valid_loss < float("inf")
    import os

    assert os.path.isfile(os.path.join(config.checkpoint_dir, "latest.ckpt"))
    assert os.path.isfile(os.path.join(config.checkpoint_dir, "best.ckpt"))


def test_video_trainer_resume(tmp_path):
    config = _trainer_config(tmp_path)
    trainer = config.build()
    trainer.train()
    seen, epoch = trainer.num_batches_seen, trainer.startEpoch

    # a fresh trainer over the same dir resumes from the latest checkpoint
    resumed = config.build()
    assert resumed.resuming
    restore_checkpoint(resumed)
    assert resumed.num_batches_seen == seen
    assert resumed.startEpoch == epoch
    assert resumed.best_valid_loss == trainer.best_valid_loss

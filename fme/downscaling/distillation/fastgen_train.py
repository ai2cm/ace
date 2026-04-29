# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
r"""
Entry point for ACE downscaling distillation via FastGen.

Usage::

    python -m fme.downscaling.distillation.fastgen_train \\
        --config fme/downscaling/distillation/configs/sft_spike.py \\
        --teacher-checkpoint /path/to/teacher.ckpt \\
        --data-yaml /path/to/coarse-only.yaml \\
        [- model.net_optimizer.lr=2e-5 ...]

The ``-`` separator before FastGen key=value overrides is required by
FastGen's ``override_config_with_opts`` (mirrors its own CLI convention).

The data YAML only needs a coarse dataset — fine ground-truth data is not
required.  The teacher's EDM sampler generates x0 targets at training time.
Minimal example::

    train_data:
      coarse:
        - data_path: /path/to/coarse.zarr
          file_pattern: "*.nc"
          engine: zarr
      batch_size: 1
      num_data_workers: 4
      strict_ensemble: false

Environment variables:
    ACE_TEACHER_CKPT  — fallback when ``--teacher-checkpoint`` is not set.
    FASTGEN_OUTPUT_ROOT — where FastGen writes logs/checkpoints (FastGen default).
"""

from __future__ import annotations

import argparse
import copy
import importlib
import multiprocessing
import os
import warnings
from typing import TYPE_CHECKING

import dacite
import torch
import yaml

from fme.core.distributed.distributed import Distributed
from fme.downscaling.data.config import DataLoaderConfig
from fme.downscaling.data.datasets import GriddedData
from fme.downscaling.distillation.best_student_callback import (
    BestStudentCheckpointCallback,
)
from fme.downscaling.distillation.fastgen_loader import AceConditionBuilder
from fme.downscaling.distillation.fastgen_teacher import AceDiffusionTeacher
from fme.downscaling.models import CheckpointModelConfig

# FastGen imports are deferred to main() so that `--help` and import-time
# checks work without a fully-installed FastGen environment.
if TYPE_CHECKING:
    from fastgen.configs.config import BaseConfig


# Must be module-level for pickling under DDP / checkpointing.
def _copy_ace_teacher(teacher: AceDiffusionTeacher) -> AceDiffusionTeacher:
    """Return an independent deepcopy of *teacher*.

    Used as a ``LazyCall`` target so each ``instantiate(config.model.net)``
    call in FastGen (for student, teacher-copy, EMA, ...) returns a distinct
    ``AceDiffusionTeacher`` with no shared weight tensors.
    """
    return copy.deepcopy(teacher)


class _CheckpointPruner:
    """Deletes old checkpoints after each save, keeping only the most recent N.

    Injected directly into ``Trainer.callbacks._callbacks`` after trainer init
    so it doesn't need a FastGen ``_target_`` DictConfig entry.

    Only rank-0 performs deletions; other ranks skip silently.

    Args:
        save_dir: Local directory where FastGen writes ``{iter:07d}.pth`` files.
        max_to_keep: Number of most-recent checkpoints to retain.
    """

    def __init__(self, save_dir: str, max_to_keep: int) -> None:
        self._save_dir = save_dir
        self._max_to_keep = max_to_keep

    # Satisfy FastGen's ``assert isinstance(_callback, Callback)`` check — we
    # implement only the one hook we need rather than inheriting the full class.
    def on_save_checkpoint_success(
        self, model=None, iteration: int = 0, path: str = ""
    ) -> None:
        try:
            from fastgen.utils.distributed import is_rank0
        except ImportError:
            return
        if not is_rank0():
            return
        try:
            names = [f for f in os.listdir(self._save_dir) if f.endswith(".pth")]
        except FileNotFoundError:
            return
        iterations = []
        for name in names:
            try:
                iterations.append(int(name.split(".")[0]))
            except ValueError:
                pass
        iterations.sort()
        to_delete = iterations[: max(0, len(iterations) - self._max_to_keep)]
        for it in to_delete:
            target = os.path.join(self._save_dir, f"{it:07d}.pth")
            try:
                os.remove(target)
            except OSError:
                pass

    # No-op stubs for all other FastGen callback hooks.
    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None


class AceInfiniteDataLoader:
    """Infinite iterator wrapping ACE's ``GriddedData`` for FastGen.

    FastGen's ``Trainer`` reads ``.batch_size`` and ``.sampler_start_idx``
    directly from ``config.dataloader_train``, then calls
    ``instantiate(config.dataloader_train)`` which returns this object
    unchanged (FastGen's ``instantiate`` passes through non-DictConfig objects
    as-is), and finally iterates it via ``iter()`` / ``next()``.

    Args:
        data: Built ``GriddedData`` from ``DataLoaderConfig.build()``.
        condition_builder: ``AceConditionBuilder`` wrapping the teacher model.
        batch_size: Per-GPU batch size (mirrors the ACE data config value).
        patch_extent_yx: Optional ``(lat_tiles, lon_tiles)`` in *coarse* grid
            units for patch-based training.  ``None`` uses full-domain batches.
        shuffle: If True, shuffle patches within each time step. Only
            meaningful when patch_extent_yx is set.
        random_offset: If True, apply a random spatial offset each epoch so
            patch boundaries vary. Only meaningful when patch_extent_yx is set.
    """

    def __init__(
        self,
        data: GriddedData,
        condition_builder: AceConditionBuilder,
        batch_size: int,
        patch_extent_yx: tuple[int, int] | None = None,
        shuffle: bool = False,
        random_offset: bool = False,
    ) -> None:
        self.batch_size = batch_size
        # Trainer may set this for sampler resumption; we accept but ignore it
        # since ACE's data generator is stateless w.r.t. iteration count.
        self.sampler_start_idx: int | None = None
        self._data = data
        self._builder = condition_builder
        self._patch_extent_yx = patch_extent_yx
        self._shuffle = shuffle
        self._random_offset = random_offset

    def __iter__(self):
        while True:
            yield from self._builder.iter_fastgen_batches(
                self._data,
                patch_extent_yx=self._patch_extent_yx,
                shuffle=self._shuffle,
                random_offset=self._random_offset,
            )


def _load_data_config(yaml_path: str) -> DataLoaderConfig:
    """Parse a coarse-only data YAML into a ``DataLoaderConfig``.

    Accepts both a bare ``DataLoaderConfig`` dict and the standard training
    YAML format that nests the data config under a ``train_data:`` key.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    data_dict = raw.get("train_data", raw)
    return dacite.from_dict(
        data_class=DataLoaderConfig,
        data=data_dict,
        config=dacite.Config(strict=False),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACE downscaling distillation via FastGen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="SPIKE.py",
        help="Path to the FastGen spike config Python file.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=os.environ.get("ACE_TEACHER_CKPT", ""),
        metavar="CKPT",
        help="Path to the pre-trained ACE teacher checkpoint. "
        "Defaults to $ACE_TEACHER_CKPT.",
    )
    parser.add_argument(
        "--data-yaml",
        required=True,
        metavar="YAML",
        help="Downscaling training YAML with a train_data: section.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        dest="log_level",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    parser.add_argument(
        "--teacher-num-steps",
        type=int,
        default=0,
        dest="teacher_num_steps",
        help=(
            "Teacher EDM sampler steps per training batch. 0 uses the "
            "checkpoint default."
        ),
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Load teacher, build one batch, run one forward pass, and exit.",
    )
    parser.add_argument(
        "--val-dataset",
        default=None,
        dest="val_dataset",
        metavar="ZARR",
        help=(
            "Path to a pre-saved teacher validation zarr produced by "
            "generate_val_dataset.py (time, ensemble, lat, lon). "
            "When set together with --val-data-yaml, the best student "
            "checkpoint (by validation RMSE) is saved in ACE format."
        ),
    )
    parser.add_argument(
        "--val-data-yaml",
        default=None,
        dest="val_data_yaml",
        metavar="YAML",
        help=(
            "Downscaling data YAML describing the coarse validation split. "
            "Must cover the same time steps as --val-dataset."
        ),
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="FastGen config overrides in '- path.key=value' form.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", "Grad strides do not match bucket view strides")

    # Set forkserver before any CUDA/distributed init so zarr DataLoader workers
    # start in a clean process without an inherited CUDA context, preventing hangs.
    multiprocessing.set_start_method("forkserver", force=True)

    # Parse args first so that `--help` exits before any FastGen imports.
    args = _parse_args()

    # FastGen loads ACE checkpoints with weights_only=True, but ACE checkpoints
    # contain numpy C types that aren't in PyTorch's default allowlist. Patch
    # torch.load globally before FastGen is imported so every load site is covered.
    import functools

    import torch as _torch

    _orig_load = _torch.load

    @functools.wraps(_orig_load)
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    _torch.load = _patched_load

    # Defer all FastGen imports until after arg parsing so `--help` works
    # in environments without the full FastGen dependency chain installed.
    # Patch FastGen's to_wandb to handle non-RGB tensors (ACE outputs C != 3).
    # WandbCallback.to_wandb asserts C == 3; we replicate the first channel to
    # RGB so sample logging works regardless of the number of output channels.
    from omegaconf import DictConfig

    import fastgen.callbacks.wandb as _wandb_mod
    import fastgen.utils.distributed.ddp as _fastgen_ddp
    import fastgen.utils.logging_utils as logger
    from fastgen.configs.config_utils import override_config_with_opts, serialize_config
    from fastgen.trainer import Trainer
    from fastgen.utils import LazyCall as L
    from fastgen.utils import instantiate
    from fastgen.utils.distributed import clean_up, is_rank0, synchronize, world_size
    from fastgen.utils.io_utils import set_env_vars
    from fastgen.utils.scripts import set_cuda_backend

    _orig_to_wandb = _wandb_mod.to_wandb

    def _ace_to_wandb(tensor: torch.Tensor, *args, **kwargs):
        if tensor.ndim >= 3 and tensor.shape[-3] != 3:
            first_chan = tensor.narrow(tensor.ndim - 3, 0, 1)
            expand_shape = list(tensor.shape)
            expand_shape[-3] = 3
            tensor = first_chan.expand(expand_shape)
        return _orig_to_wandb(tensor, *args, **kwargs)

    _wandb_mod.to_wandb = _ace_to_wandb

    logger.set_log_level(args.log_level)

    if not args.teacher_checkpoint:
        raise ValueError("Provide --teacher-checkpoint or set $ACE_TEACHER_CKPT.")

    # ------------------------------------------------------------------
    # 1. Import and optionally override the FastGen spike config.
    # ------------------------------------------------------------------
    if not args.config.endswith(".py"):
        raise ValueError(f"--config must be a .py file, got: {args.config!r}")
    config_module_name = args.config.replace("/", ".").replace(".py", "")
    spike_module = importlib.import_module(config_module_name)
    config: BaseConfig = spike_module.create_config()
    if args.opts:
        config = override_config_with_opts(config, args.opts)

    # ------------------------------------------------------------------
    # 2. DDP init — must precede any CUDA allocation.
    # ------------------------------------------------------------------
    if config.trainer.ddp or config.trainer.fsdp:
        if not torch.distributed.is_available():
            raise RuntimeError(
                "torch.distributed not available. Check your PyTorch install."
            )
        _fastgen_ddp.init()
        logger.info(f"DDP initialised, world_size={world_size()}")
    else:
        logger.info("Running without DDP/FSDP.")

    set_cuda_backend(
        config.trainer.cudnn.deterministic,
        config.trainer.cudnn.benchmark,
        config.trainer.tf32_enabled,
    )

    # ------------------------------------------------------------------
    # 3. Load ACE teacher checkpoint.
    # ------------------------------------------------------------------
    ckpt_config = CheckpointModelConfig(
        checkpoint_path=args.teacher_checkpoint,
        fine_coordinates_path="/climate-default/2026-02-23-X-SHiELD-AMIP-plus-4K-downscaling/3km.zarr",
    )
    requirements = ckpt_config.data_requirements
    teacher_model = ckpt_config.build()
    teacher = AceDiffusionTeacher(teacher_model)
    logger.info(f"Teacher loaded from {args.teacher_checkpoint!r}")

    # ------------------------------------------------------------------
    # 3b. Auto-configure the DMD2 discriminator from the teacher's UNet.
    #
    #     The default Discriminator_EDM_CIFAR10_Config uses in_channels=256
    #     at all_res=[32,16,8], which won't match ACE's 512×512 UNet.  We
    #     introspect the loaded UNet's encoder blocks and replace the
    #     discriminator with one that matches the deepest encoder level
    #     (bottleneck features — most semantically rich, single channel count).
    # ------------------------------------------------------------------
    if hasattr(config.model, "discriminator"):
        from fastgen.networks.discriminators import Discriminator_EDM as _DiscEDM

        enc_info = teacher.encoder_feature_info()
        # enc_info: [(block_key, out_channels, resolution), ...] finest→coarsest
        deepest_idx = len(enc_info) - 1
        _, disc_channels, _ = enc_info[deepest_idx]
        all_res = [res for _, _, res in enc_info]
        config.model.discriminator = L(_DiscEDM)(
            feature_indices={deepest_idx},
            all_res=all_res,
            in_channels=disc_channels,
        )
        logger.info(
            f"DMD2 discriminator: feature_index={deepest_idx}, "
            f"all_res={all_res}, in_channels={disc_channels}"
        )

    # ------------------------------------------------------------------
    # 4. Inject teacher into FastGen config as a deepcopy factory.
    #
    #    FastGen calls instantiate(config.model.net) twice: once for the
    #    student (self.net) and once for the teacher copy (self.teacher).
    #    Using L(_copy_ace_teacher) ensures each call returns an independent
    #    deepcopy so freezing the teacher-copy never affects the student.
    # ------------------------------------------------------------------
    config.model.net = L(_copy_ace_teacher)(teacher=teacher)
    config.model.pretrained_model_path = ""  # weights come from deepcopy above
    config.model.use_ema = []

    # Strip EMA callbacks — they fire on EMA objects that no longer exist.
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )

    # ------------------------------------------------------------------
    # 5. Build ACE data pipeline (coarse-only).
    #
    #    We pass a NonDistributed-backed Distributed so ACE's sampler logic
    #    stays on the single-rank path.  FastGen's DDP wrapper handles
    #    gradient synchronisation across model replicas; each rank sees the
    #    same full dataset (acceptable for the spike).
    # ------------------------------------------------------------------
    data_cfg = _load_data_config(args.data_yaml)
    data_cfg.shuffle = True
    ace_dist = Distributed(force_non_distributed=True)
    train_data = data_cfg.build(requirements=requirements, dist=ace_dist)
    condition_builder = AceConditionBuilder(
        teacher_model, teacher, teacher_num_steps=args.teacher_num_steps
    )

    fine_h, fine_w = config.model.input_shape[1], config.model.input_shape[2]
    coarse_patch_yx = (
        fine_h // teacher_model.downscale_factor,
        fine_w // teacher_model.downscale_factor,
    )
    domain_h, domain_w = train_data.shape
    # Only patch when the domain is strictly larger than the patch in both dims.
    # If domain <= patch in either dim, drop_partial_patches would silently drop
    # every patch and the loader would yield nothing.
    if domain_h > coarse_patch_yx[0] and domain_w > coarse_patch_yx[1]:
        patch_extent_yx: tuple[int, int] | None = coarse_patch_yx
    else:
        patch_extent_yx = None
    logger.info(
        f"domain={domain_h}x{domain_w} coarse_patch={coarse_patch_yx} "
        f"patch_extent_yx={patch_extent_yx}"
    )
    ace_loader = AceInfiniteDataLoader(
        data=train_data,
        condition_builder=condition_builder,
        batch_size=data_cfg.batch_size,
        patch_extent_yx=patch_extent_yx,
        shuffle=patch_extent_yx is not None,
        random_offset=patch_extent_yx is not None,
    )
    config.dataloader_train = ace_loader

    # ------------------------------------------------------------------
    # 6. Recompute gradient-accumulation rounds using the real batch size.
    # ------------------------------------------------------------------
    if getattr(config.trainer, "batch_size_global", None) is not None:
        per_gpu = data_cfg.batch_size
        total = per_gpu * world_size()
        accum_rounds = max(config.trainer.batch_size_global // total, 1)
        effective_global = accum_rounds * total
        if effective_global != config.trainer.batch_size_global:
            logger.critical(
                f"batch_size_global={config.trainer.batch_size_global} is not "
                f"divisible by per_gpu × world_size ({per_gpu} × {world_size()} "
                f"= {total}). Effective global batch size: {effective_global}."
            )
        config.trainer.grad_accum_rounds = accum_rounds

    # ------------------------------------------------------------------
    # 7. Checkpointer, S3 credentials, config snapshot.
    # ------------------------------------------------------------------
    config.trainer.checkpointer.save_dir = (
        f"{config.log_config.save_path}/{config.trainer.checkpointer.save_dir}"
    )
    set_env_vars(config.trainer.checkpointer.s3_credential)
    if is_rank0():
        serialize_config(
            config,
            return_type="file",
            path=config.log_config.save_path,
            filename="config.yaml",
        )

    # ------------------------------------------------------------------
    # 8. Dryrun smoke test — verify shapes before launching real training.
    # ------------------------------------------------------------------
    if args.dryrun:
        logger.info("Dryrun: verifying teacher sampling and batch shapes.")
        teacher.freeze()
        batch = next(iter(ace_loader))
        logger.info(
            f"Dryrun OK — real: {tuple(batch['real'].shape)}, "
            f"condition: {tuple(batch['condition'].shape)}"
        )
        return

    # ------------------------------------------------------------------
    # 9. Instantiate FastGen model and run training.
    # ------------------------------------------------------------------
    config.model_class.config = config.model
    model = instantiate(config.model_class)
    config.model_class.config = None
    synchronize()

    logger.info("Initialising FastGen Trainer...")
    fastgen_trainer = Trainer(config)
    fastgen_trainer.callbacks._callbacks["ckpt_pruner"] = _CheckpointPruner(
        save_dir=config.trainer.checkpointer.save_dir,
        max_to_keep=20,
    )

    if args.val_dataset and args.val_data_yaml:
        val_data_cfg = _load_data_config(args.val_data_yaml)
        val_data_cfg.shuffle = False
        ace_dist_val = Distributed(force_non_distributed=True)
        coarse_val_data = val_data_cfg.build(
            requirements=requirements, dist=ace_dist_val
        )
        best_student_path = os.path.join(
            config.trainer.checkpointer.save_dir, "best_student.ckpt"
        )
        fastgen_trainer.callbacks._callbacks["best_student"] = (
            BestStudentCheckpointCallback(
                val_dataset_path=args.val_dataset,
                coarse_val_data=coarse_val_data,
                teacher_model=teacher_model,
                best_checkpoint_path=best_student_path,
            )
        )
        logger.info(
            f"BestStudentCheckpointCallback active: val={args.val_dataset}, "
            f"best_ckpt={best_student_path}"
        )

    synchronize()

    fastgen_trainer.run(model)
    synchronize()

    clean_up()
    logger.info("Training finished.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", "Grad strides do not match bucket view strides")
    os.environ.setdefault("FME_DISTRIBUTED_BACKEND", "none")
    with Distributed.context():
        main()

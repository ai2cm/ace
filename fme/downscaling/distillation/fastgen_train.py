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
import os
import warnings
from typing import TYPE_CHECKING

import dacite
import torch
import yaml

from fme.core.distributed.distributed import Distributed
from fme.downscaling.data.config import DataLoaderConfig
from fme.downscaling.data.datasets import GriddedData
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
    """

    def __init__(
        self,
        data: GriddedData,
        condition_builder: AceConditionBuilder,
        batch_size: int,
        patch_extent_yx: tuple[int, int] | None = None,
    ) -> None:
        self.batch_size = batch_size
        # Trainer may set this for sampler resumption; we accept but ignore it
        # since ACE's data generator is stateless w.r.t. iteration count.
        self.sampler_start_idx: int | None = None
        self._data = data
        self._builder = condition_builder
        self._patch_extent_yx = patch_extent_yx

    def __iter__(self):
        while True:
            yield from self._builder.iter_fastgen_batches(
                self._data, patch_extent_yx=self._patch_extent_yx
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
        "--dryrun",
        action="store_true",
        help="Load teacher, build one batch, run one forward pass, and exit.",
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
    import multiprocessing

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
    from omegaconf import DictConfig

    import fastgen.utils.distributed.ddp as _fastgen_ddp
    import fastgen.utils.logging_utils as logger
    from fastgen.configs.config_utils import override_config_with_opts, serialize_config
    from fastgen.trainer import Trainer
    from fastgen.utils import LazyCall as L
    from fastgen.utils import instantiate
    from fastgen.utils.distributed import clean_up, is_rank0, synchronize, world_size
    from fastgen.utils.io_utils import set_env_vars
    from fastgen.utils.scripts import set_cuda_backend

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
    # 4. Inject teacher into FastGen config as a deepcopy factory.
    #
    #    FastGen calls instantiate(config.model.net) twice: once for the
    #    student (self.net) and once for the teacher copy (self.teacher).
    #    Using L(_copy_ace_teacher) ensures each call returns an independent
    #    deepcopy so freezing the teacher-copy never affects the student.
    # ------------------------------------------------------------------
    config.model.net = L(_copy_ace_teacher)(teacher=teacher)
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
    ace_dist = Distributed(force_non_distributed=True)
    train_data = data_cfg.build(requirements=requirements, dist=ace_dist)
    condition_builder = AceConditionBuilder(teacher_model, teacher)

    fine_h, fine_w = config.model.input_shape[1], config.model.input_shape[2]
    coarse_patch_yx = (
        fine_h // teacher_model.downscale_factor,
        fine_w // teacher_model.downscale_factor,
    )
    domain_h, domain_w = train_data.shape
    patch_extent_yx = (
        coarse_patch_yx if (domain_h, domain_w) != coarse_patch_yx else None
    )
    ace_loader = AceInfiniteDataLoader(
        data=train_data,
        condition_builder=condition_builder,
        batch_size=data_cfg.batch_size,
        patch_extent_yx=patch_extent_yx,
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

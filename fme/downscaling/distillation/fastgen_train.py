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


def _channels_to_grid(
    tensor: torch.Tensor, channel_names: list[str], max_samples: int = 8
) -> tuple[torch.Tensor, str]:
    """Tile a ``[B, C, H, W]`` batch as a per-channel panel grid for logging.

    One row per sample, one column per output channel (variable), each channel
    independently min-max normalized to ``[0, 1]`` so disparate fields (e.g.
    PRMSL vs precip) are each visible.  Returns the ``[3, Hg, Wg]`` grid tensor
    (grayscale replicated to RGB) and a caption naming the columns.

    Replaces the previous behavior of showing only channel 0 — ACE has 4 output
    variables, so dropping channels 1-3 hid most of the state (see
    MOE_DISTILLATION_STATUS.md train-media note).
    """
    import torchvision

    b, c, h, w = tensor.shape
    n = min(b, max_samples)
    panels = tensor[:n].detach().float().cpu()
    # Per-channel min-max over the displayed samples → [0, 1].
    flat = panels.permute(1, 0, 2, 3).reshape(c, -1)
    lo = flat.min(dim=1).values.view(1, c, 1, 1)
    hi = flat.max(dim=1).values.view(1, c, 1, 1)
    panels = ((panels - lo) / (hi - lo).clamp(min=1e-8)).clamp(0.0, 1.0)
    # Sample-major flatten so make_grid lays out each sample's channels in a row.
    grid_in = panels.reshape(n * c, 1, h, w).expand(n * c, 3, h, w)
    grid = torchvision.utils.make_grid(grid_in, nrow=c, pad_value=1.0)
    names = list(channel_names) or [f"ch{i}" for i in range(c)]
    caption = "cols: " + " | ".join(names) + "  (rows: samples)"
    return grid, caption


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
    # strict=True is required so that XarrayDataConfig.subset resolves to
    # TimeSlice instead of silently falling back to an empty Slice (which
    # would load the entire underlying zarr regardless of start_time/stop_time).
    return dacite.from_dict(
        data_class=DataLoaderConfig,
        data=data_dict,
        config=dacite.Config(strict=True),
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
        help="Path to the pre-trained ACE teacher checkpoint (.ckpt). "
        "Defaults to $ACE_TEACHER_CKPT.",
    )
    parser.add_argument(
        "--teacher-moe-checkpoint",
        default=os.environ.get("ACE_TEACHER_MOE_CKPT", ""),
        metavar="CKPT",
        dest="teacher_moe_checkpoint",
        help="Path to a bundled DenoisingMoEPredictor checkpoint (.ckpt) "
        "produced by DenoisingMoEPredictor.save(). When set, this takes "
        "precedence over --teacher-checkpoint. Defaults to "
        "$ACE_TEACHER_MOE_CKPT.",
    )
    parser.add_argument(
        "--expert-index",
        type=int,
        default=(
            int(os.environ["ACE_EXPERT_INDEX"])
            if os.environ.get("ACE_EXPERT_INDEX", "") != ""
            else None
        ),
        dest="expert_index",
        metavar="I",
        help=(
            "Distil a single expert (by ascending-sigma index) of an MoE "
            "teacher in-domain over its own sigma range, with no dispatch. "
            "Per-expert distillation: 0 = low-noise (Student-Lo), 1 = "
            "high-noise (Student-Hi). Only valid with --teacher-moe-checkpoint. "
            "Defaults to $ACE_EXPERT_INDEX (full dispatch when unset)."
        ),
    )
    parser.add_argument(
        "--val-mode",
        default=os.environ.get("ACE_VAL_MODE", "from_noise"),
        dest="val_mode",
        choices=["from_noise", "lo_renoise"],
        help=(
            "How BestStudentCheckpointCallback produces the student ensemble "
            "for validation. 'from_noise' (default) denoises fresh noise to a "
            "clean x0 (full-range student); 'lo_renoise' re-noises the teacher "
            "target to the student's sigma_max and denoises from there "
            "(per-expert Student-Lo). Defaults to $ACE_VAL_MODE."
        ),
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
    # WandbCallback.to_wandb asserts C == 3 and would otherwise fail; instead we
    # render every output channel (variable) as its own per-channel-normalized
    # panel — one row per sample, one column per variable.  Column labels come
    # from the teacher out_packer, populated below once the teacher loads.
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
    from omegaconf import DictConfig

    _orig_to_wandb = _wandb_mod.to_wandb
    # Output-variable names for panel labels; filled once the teacher loads.
    _ace_channel_names: list[str] = []

    def _ace_to_wandb(tensor: torch.Tensor, *args, **kwargs):
        # Only the 4D image case (ACE) gets the per-channel panel treatment;
        # already-RGB or video tensors fall back to FastGen's renderer.  Guarded
        # so a viz failure can never crash training.
        if tensor.ndim != 4 or tensor.shape[-3] == 3:
            return _orig_to_wandb(tensor, *args, **kwargs)
        try:
            import torchvision.transforms.functional as tv_F

            grid, caption = _channels_to_grid(tensor, _ace_channel_names)
            return _wandb_mod.wandb.Image(tv_F.to_pil_image(grid), caption=caption)
        except Exception:  # noqa: BLE001 - sample viz is best-effort
            return _orig_to_wandb(tensor, *args, **kwargs)

    _wandb_mod.to_wandb = _ace_to_wandb

    logger.set_log_level(args.log_level)

    if not args.teacher_checkpoint and not args.teacher_moe_checkpoint:
        raise ValueError(
            "Provide --teacher-checkpoint / $ACE_TEACHER_CKPT "
            "or --teacher-moe-checkpoint / $ACE_TEACHER_MOE_CKPT."
        )

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
    #
    #    Supports two formats:
    #    - Single DiffusionModel (.ckpt): use --teacher-checkpoint
    #    - Bundled DenoisingMoEPredictor (.ckpt): use --teacher-moe-checkpoint
    # ------------------------------------------------------------------
    from fme.downscaling.models import DiffusionModel
    from fme.downscaling.predictors.serial_denoising import DenoisingMoEPredictor

    teacher_model: DiffusionModel | DenoisingMoEPredictor
    if args.teacher_moe_checkpoint:
        from fme.downscaling.predictors import DenoisingMoEBundledConfig

        moe_cfg = DenoisingMoEBundledConfig(
            mixture_of_experts_path=args.teacher_moe_checkpoint
        )
        requirements = moe_cfg.data_requirements
        teacher_model = moe_cfg.build()
        logger.info(f"MoE teacher loaded from {args.teacher_moe_checkpoint!r}")
    else:
        ckpt_config = CheckpointModelConfig(
            checkpoint_path=args.teacher_checkpoint,
            fine_coordinates_path="/climate-default/2026-02-23-X-SHiELD-AMIP-plus-4K-downscaling/3km.zarr",
        )
        requirements = ckpt_config.data_requirements
        teacher_model = ckpt_config.build()
        logger.info(f"Teacher loaded from {args.teacher_checkpoint!r}")
    if args.expert_index is not None and not isinstance(
        teacher_model, DenoisingMoEPredictor
    ):
        raise ValueError(
            "--expert-index / $ACE_EXPERT_INDEX requires an MoE teacher "
            "(--teacher-moe-checkpoint)."
        )
    teacher = AceDiffusionTeacher(teacher_model, expert_index=args.expert_index)
    if args.expert_index is not None:
        logger.info(
            f"Single-expert teacher: expert_index={args.expert_index}, "
            f"sigma range [{teacher._sigma_min}, {teacher._sigma_max}] "
            "(no dispatch)."
        )

    # ------------------------------------------------------------------
    # 3b. Override input_shape channel count from the teacher.
    #
    #     The spike configs derive C_out from ACE_C_OUT, which must be set
    #     manually and can disagree with the actual teacher (e.g. a MoE bundle
    #     whose primary expert has fewer output channels than the full ensemble).
    #     Overriding here after the teacher loads is the authoritative source.
    # ------------------------------------------------------------------

    _ref_model = (
        teacher_model._primary
        if isinstance(teacher_model, DenoisingMoEPredictor)
        else teacher_model
    )
    c_out_teacher = len(_ref_model.out_packer.names)
    # Label the sample-media panels with the output-variable names (see the
    # to_wandb patch above).
    _ace_channel_names[:] = list(_ref_model.out_packer.names)
    c_out_config = config.model.input_shape[0]
    if c_out_config != c_out_teacher:
        logger.warning(
            f"input_shape channel count ({c_out_config}) does not match "
            f"teacher out_packer ({c_out_teacher}); overriding."
        )
    config.model.input_shape[0] = c_out_teacher
    logger.info(
        f"input_shape set to {list(config.model.input_shape)} "
        f"(C_out={c_out_teacher} from teacher)"
    )

    # ------------------------------------------------------------------
    # 3c. Auto-configure the DMD2 discriminator from the teacher's UNet.
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
        # Separate directory so evaluation jobs can mount just these two files
        # without downloading all the raw .pth training checkpoints.
        best_student_dir = os.path.join(
            config.log_config.save_path, "student_checkpoints"
        )
        os.makedirs(best_student_dir, exist_ok=True)
        best_student_path = os.path.join(best_student_dir, "best_student.ckpt")
        best_student_tail_path = os.path.join(
            best_student_dir, "best_student_tail.ckpt"
        )
        _teacher_diffusion_model: DiffusionModel = (
            teacher_model._primary
            if isinstance(teacher_model, DenoisingMoEPredictor)
            else teacher_model
        )
        fastgen_trainer.callbacks._callbacks["best_student"] = (
            BestStudentCheckpointCallback(
                val_dataset_path=args.val_dataset,
                coarse_val_data=coarse_val_data,
                teacher_model=_teacher_diffusion_model,
                best_checkpoint_path=best_student_path,
                coarse_patch_yx=coarse_patch_yx,
                student_sample_steps=config.model.student_sample_steps,
                best_tail_checkpoint_path=best_student_tail_path,
                validation_mode=args.val_mode,
            )
        )
        logger.info(
            f"BestStudentCheckpointCallback active: val={args.val_dataset}, "
            f"best_ckpt={best_student_path}, "
            f"best_tail_ckpt={best_student_tail_path}, "
            f"validation_mode={args.val_mode}, "
            f"student_sample_steps={config.model.student_sample_steps}"
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

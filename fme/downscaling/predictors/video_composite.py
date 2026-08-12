import torch

from fme.core.typing_ import TensorDict
from fme.downscaling.data.datasets import PairedVideoBatchData, VideoBatchData
from fme.downscaling.data.patching import Patch, get_patches
from fme.downscaling.data.utils import scale_tuple
from fme.downscaling.predictors.composite import composite_patch_predictions
from fme.downscaling.video_models import VideoDiffusionModel


class VideoPatchPredictor:
    """Video analog of ``predictors.composite.PatchPredictor``: generates a
    full-extent video clip prediction by tiling the requested domain into
    patches matching the model's trained coarse patch size, generating each
    patch independently, and compositing the fine-resolution results back
    into the full extent (averaging overlapping regions).

    Unlike the plain spatial model, ``VideoDiffusionModel`` has no
    ``coarse_shape`` of its own (it's a property of the video UNet's
    *training* patch config, not stored on the built model), so
    ``coarse_yx_patch_extent`` must be given explicitly here rather than
    inferred. ``downscale_factor`` is likewise taken explicitly rather than
    read off ``model.downscale_factor``: that attribute is only set when the
    model is built with ``full_fine_coords``/``downscale_factor`` (needed
    for ``get_fine_coords_for_batch``/``generate_on_batch_no_target``), which
    ordinary paired-batch inference (``generate()``, what this predictor
    wraps for ``video_inference.py``) has no other reason to require --
    ``downscale_factor`` here is trivially available from the data loader
    instead (e.g. ``PairedVideoGriddedData.downscale_factor``).

    Each patch is generated as its own independent video-diffusion sample
    (independent ensemble noise per patch, no shared spatial context across
    patch boundaries) -- the same tradeoff ``PatchPredictor`` already makes
    for the plain model, addressed the same way: overlapping patches with
    the overlap region averaged away at composite time.
    """

    def __init__(
        self,
        model: VideoDiffusionModel,
        coarse_yx_patch_extent: tuple[int, int],
        downscale_factor: int,
        coarse_horizontal_overlap: int = 1,
    ):
        self.model = model
        self.modules = model.modules
        self.out_names = model.out_names
        self.coarse_yx_patch_extent = coarse_yx_patch_extent
        self.downscale_factor = downscale_factor
        self.coarse_horizontal_overlap = coarse_horizontal_overlap

    def _get_patches(
        self, coarse_yx_extent: tuple[int, int], fine_yx_extent: tuple[int, int]
    ) -> tuple[list[Patch], list[Patch]]:
        coarse_patches = get_patches(
            yx_extent=coarse_yx_extent,
            yx_patch_extent=self.coarse_yx_patch_extent,
            overlap=self.coarse_horizontal_overlap,
            drop_partial_patches=False,
        )
        fine_yx_patch_extent = scale_tuple(
            self.coarse_yx_patch_extent, self.downscale_factor
        )
        fine_patches = get_patches(
            yx_extent=fine_yx_extent,
            yx_patch_extent=fine_yx_patch_extent,
            overlap=self.coarse_horizontal_overlap * self.downscale_factor,
            drop_partial_patches=False,
        )
        return coarse_patches, fine_patches

    @torch.no_grad()
    def generate(
        self,
        batch: PairedVideoBatchData,
        n_samples: int = 1,
        frames: list[int] | None = None,
    ) -> TensorDict:
        """Patch, generate, and composite -- signature matches
        ``VideoDiffusionModel.generate`` exactly, so this is a drop-in
        replacement for it in ``video_inference.py``.
        """
        coarse_patches, fine_patches = self._get_patches(
            coarse_yx_extent=batch.coarse.horizontal_shape,
            fine_yx_extent=batch.fine.horizontal_shape,
        )
        predictions = []
        for data_patch in batch.generate_from_patches(coarse_patches, fine_patches):
            predictions.append(
                self.model.generate(data_patch, n_samples=n_samples, frames=frames)
            )
        return composite_patch_predictions(predictions, fine_patches)

    @torch.no_grad()
    def generate_on_batch_no_target(
        self,
        coarse: VideoBatchData,
        n_samples: int = 1,
        frames: list[int] | None = None,
    ) -> TensorDict:
        """Patch, generate, and composite from coarse data ALONE -- video
        analog of ``PatchPredictor.generate_on_batch_no_target``, matching
        ``VideoDiffusionModel.generate_on_batch_no_target``'s signature.
        """
        coarse_yx_extent = coarse.horizontal_shape
        fine_yx_extent = scale_tuple(coarse_yx_extent, self.downscale_factor)
        coarse_patches, fine_patches = self._get_patches(
            coarse_yx_extent=coarse_yx_extent, fine_yx_extent=fine_yx_extent
        )
        predictions = []
        for coarse_patch in coarse.generate_from_patches(coarse_patches):
            predictions.append(
                self.model.generate_on_batch_no_target(
                    coarse_patch, n_samples=n_samples, frames=frames
                )
            )
        return composite_patch_predictions(predictions, fine_patches)

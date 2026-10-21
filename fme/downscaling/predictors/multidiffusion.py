"""
MultiDiffusion generation over a domain larger than the model's trained patch
size (Bar-Tal et al. 2023, https://arxiv.org/abs/2302.08113).

Rather than sampling each patch independently and averaging the finished
samples where patches overlap, a single full-domain sample is denoised jointly:
at every sampler evaluation, each patch of the current full-domain state is
denoised by the model at its trained patch shape and the denoised patches are
averaged where they overlap. Because the EDM update is linear in the denoised
estimate, averaging the denoised estimates is the MultiDiffusion uniform-weight
fusion of the per-patch sampler steps.

The full-domain sampler state can optionally be kept in host memory, with each
patch moved to the GPU only for its network call, so that GPU memory scales
with the patch size rather than the domain size.
"""

import torch

from fme.core.device import get_device
from fme.core.typing_ import TensorDict
from fme.downscaling.data import BatchData
from fme.downscaling.data.patching import HorizontalSlice, Patch, PatchAverager
from fme.downscaling.models import DiffusionModel, _repeat_batch_by_samples
from fme.downscaling.predictors.serial_denoising import DenoisingMoEPredictor
from fme.downscaling.typing_ import Denoiser


def _scale_slice(patch_slice: HorizontalSlice, factor: int) -> HorizontalSlice:
    return HorizontalSlice(
        y=slice(patch_slice.y.start * factor, patch_slice.y.stop * factor),
        x=slice(patch_slice.x.start * factor, patch_slice.x.stop * factor),
    )


class MultiDiffusionDenoiser:
    """
    Wraps a patch denoiser ``net(x, x_lr, sigma)`` to denoise a full-domain
    state by denoising each patch and averaging overlapping patch estimates.

    The sampler state is a residual relative to the full-domain base prediction
    (zero if the model does not predict a residual), while each patch's network
    was trained against a base interpolated from that patch's coarse data alone.
    ``base_shifts`` (full-domain base minus patch base, over the patch region)
    converts the patch state into the patch's own residual space and back.

    The state may be on a different device than the network: each patch, its
    conditioning and its base shift are moved to ``device`` for the network
    call, and the denoised patch is moved back to the state's device.

    Args:
        net: Denoiser called on each patch at the model's trained patch shape.
        average: Averages the denoised patches over the full fine domain; its
            regions are the patches' regions of the domain, and its device is
            the state's.
        patch_inputs: Conditioning input for each patch, as the model would
            build it for that patch on its own.
        base_shifts: Per-patch base shift, or None for each patch if the
            model does not predict a residual.
        device: Device the network runs on.
    """

    def __init__(
        self,
        net: Denoiser,
        average: PatchAverager,
        patch_inputs: list[torch.Tensor],
        base_shifts: list[torch.Tensor | None],
        device: torch.device,
    ):
        if not (len(average.regions) == len(patch_inputs) == len(base_shifts)):
            raise ValueError(
                "average.regions, patch_inputs, and base_shifts must have the same "
                "length."
            )
        self.net = net
        self.average = average
        self.patch_inputs = patch_inputs
        self.base_shifts = base_shifts
        self.device = device

    def __call__(
        self, x: torch.Tensor, x_lr: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        # x_lr is unused: conditioning is per patch, held in patch_inputs
        if tuple(x.shape[-2:]) != tuple(self.average.count.shape):
            raise ValueError(
                f"State shape {tuple(x.shape[-2:])} does not match the patch "
                f"layout's domain shape {tuple(self.average.count.shape)}."
            )
        sigma = sigma.to(self.device)

        def denoised_patches():
            for fine_slice, inputs, shift in zip(
                self.average.regions, self.patch_inputs, self.base_shifts
            ):
                x_patch = x[..., fine_slice.y, fine_slice.x].to(self.device)
                if shift is not None:
                    shift = shift.to(self.device)
                    x_patch = x_patch + shift
                denoised = self.net(x_patch, inputs.to(self.device), sigma)
                denoised = denoised.to(x.dtype)
                if shift is not None:
                    denoised = denoised - shift
                yield denoised

        return self.average(denoised_patches())


@torch.no_grad()
def generate_multidiffusion(
    model: DiffusionModel | DenoisingMoEPredictor,
    batch: BatchData,
    coarse_patches: list[Patch],
    n_samples: int = 1,
    sampler_dtype: torch.dtype = torch.float64,
    sampler_state_on_cpu: bool = False,
) -> tuple[TensorDict, torch.Tensor]:
    """
    Generate a full-domain sample with MultiDiffusion, calling the model only
    on coarse patches of its trained shape.

    Args:
        model: Model to generate with.
        batch: Coarse data over the full domain.
        coarse_patches: Patches covering the full coarse domain. Each patch's
            input slice must be the model's trained coarse shape.
        n_samples: Number of samples to generate per batch element.
        sampler_dtype: Dtype of the full-domain sampler state and updates.
        sampler_state_on_cpu: If True, keep the full-domain sampler state and
            the per-patch conditioning in host memory, moving each patch to
            the GPU only for its network call.

    Returns:
        Denormalized generated outputs with shape (batch, sample, lat, lon)
        and the normalized generated tensor. Unlike ``DiffusionModel.generate``,
        the sampler's latent steps are not returned: each is a copy of the
        full-domain state, too large to keep for every step on large domains.
    """
    base_model = model.base_model
    downscale_factor = base_model.downscale_factor
    device = get_device()
    state_device = torch.device("cpu") if sampler_state_on_cpu else device
    full_base = base_model.get_base_prediction(batch.data)

    fine_slices: list[HorizontalSlice] = []
    patch_inputs: list[torch.Tensor] = []
    base_shifts: list[torch.Tensor | None] = []
    for patch, patch_batch in zip(
        coarse_patches, batch.generate_from_patches(coarse_patches)
    ):
        if tuple(patch_batch.horizontal_shape) != tuple(base_model.coarse_shape):
            raise ValueError(
                f"Patch shape {patch_batch.horizontal_shape} does not match the "
                f"model coarse shape {base_model.coarse_shape}."
            )
        fine_slice = _scale_slice(patch.input_slice, downscale_factor)
        fine_slices.append(fine_slice)
        static_inputs = base_model._subset_static_if_available(patch_batch)
        inputs = base_model._get_input_from_coarse(patch_batch.data, static_inputs)
        patch_inputs.append(
            _repeat_batch_by_samples(inputs, n_samples).to(state_device)
        )
        if full_base is None:
            base_shifts.append(None)
        else:
            patch_base = base_model.get_base_prediction(patch_batch.data)
            assert patch_base is not None
            shift = full_base[..., fine_slice.y, fine_slice.x] - patch_base
            base_shifts.append(
                _repeat_batch_by_samples(shift, n_samples).to(state_device)
            )

    n_batch = next(iter(batch.data.values())).shape[0]
    fine_shape = base_model._get_fine_shape(batch.horizontal_shape)
    latents = torch.randn(
        n_batch * n_samples, len(base_model.out_packer.names), *fine_shape
    ).to(device=state_device)
    # The sampler requires a conditioning tensor with a matching batch size, but
    # the MultiDiffusion denoiser carries its own per-patch conditioning.
    unused_inputs = latents.new_empty(latents.shape[0], 0)

    average = PatchAverager(fine_slices, fine_shape, device=state_device)

    def wrap_denoiser(net: Denoiser) -> Denoiser:
        return MultiDiffusionDenoiser(
            net, average, patch_inputs, base_shifts, device=device
        )

    def randn_like(x: torch.Tensor) -> torch.Tensor:
        # Draw sampler noise in float64 whatever the sampler dtype, so that on a
        # given device and seed, every sampler_dtype gets the same noise (and the
        # float64 default matches DiffusionModel.generate).
        return torch.randn(x.shape, dtype=torch.float64, device=x.device).to(x.dtype)

    generated_norm, _ = model.run_sampler(
        latents,
        unused_inputs,
        wrap_denoiser=wrap_denoiser,
        record_latent_steps=False,
        compute_dtype=sampler_dtype,
        randn_like=randn_like,
    )
    generated_norm = generated_norm.to(device)
    generated, generated_norm = base_model.postprocess_generated(
        generated_norm, batch.data, n_samples
    )
    return generated, generated_norm

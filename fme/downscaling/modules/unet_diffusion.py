import contextlib
import logging
import time

import torch

from fme.core.device import get_device

logger = logging.getLogger(__name__)


class _MemoryFormatConversionCounter:
    """Counts and logs memory format transitions by hooking into module forward passes.

    Uses forward hooks on leaf modules to detect when:
    - Input is channels_last but output is contiguous (channels_last → contiguous)
    - Input is contiguous but output is channels_last (contiguous → channels_last)
    """

    def __init__(self, model: torch.nn.Module):
        self.channels_last_to_contiguous = 0
        self.contiguous_to_channels_last = 0
        self._model = model
        self._hooks: list = []

    def __enter__(self):
        counter = self

        def make_hook(name):
            def hook(module, inputs, output):
                if not isinstance(output, torch.Tensor) or output.ndim != 4:
                    return
                # Check first 4D tensor input
                input_tensor = None
                for inp in inputs:
                    if isinstance(inp, torch.Tensor) and inp.ndim == 4:
                        input_tensor = inp
                        break
                if input_tensor is None:
                    return

                input_is_cl = input_tensor.is_contiguous(
                    memory_format=torch.channels_last
                )
                output_is_cl = output.is_contiguous(memory_format=torch.channels_last)

                if input_is_cl and not output_is_cl:
                    counter.channels_last_to_contiguous += 1
                    logger.info(f"  {name}: channels_last -> contiguous")
                elif not input_is_cl and output_is_cl:
                    counter.contiguous_to_channels_last += 1
                    logger.info(f"  {name}: contiguous -> channels_last")

            return hook

        # Register hooks on leaf modules only
        for name, module in self._model.named_modules():
            if len(list(module.children())) == 0:
                self._hooks.append(module.register_forward_hook(make_hook(name)))

        return self

    def __exit__(self, *args):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class UNetDiffusionModule(torch.nn.Module):
    """
    Maps interpolated coarse grid fields with fine grid latent variables,
    combined with embedded noise to a fine output using a U-Net.
    The latent variables are conditioned on the coarse image via simple
    concatenation in the channel dimension, which is why it is required to
    pass coarse fields interpolated to the target grid resolution. This
    is intended to be used for denoising diffusion models where the latent
    variables are noised fine grid targets and coarse are the paired coarse
    grid fields.

    Args:
        unet: The U-Net model.
        use_amp_bf16: use automatic mixed precision casting to bfloat16
            during forward pass
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        use_amp_bf16: bool = True,
        use_channels_last: bool = True,
    ):
        super().__init__()
        self.unet = unet.to(get_device())
        self.use_amp_bf16 = use_amp_bf16
        self.use_channels_last = use_channels_last
        self._forward_log_count = 0
        self._forward_log_limit = 10

        if self.use_amp_bf16:
            if get_device().type == "mps":
                raise ValueError("MPS does not support bfloat16 autocast.")
            self._amp_context = torch.amp.autocast(
                get_device().type, dtype=torch.bfloat16
            )
        else:
            self._amp_context = contextlib.nullcontext()

    def forward(
        self,
        latent: torch.Tensor,
        conditioning: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the UNetDiffusion module.

        Args:
            conditioning: The conditioning input fields (same shape as target latents).
            latent: The latent diffusion variable on the fine grid.
            noise_level: The noise level of each example in the batch.
        """
        should_log = self._forward_log_count < self._forward_log_limit
        if should_log:
            start_time = time.perf_counter()
            conversion_counter = _MemoryFormatConversionCounter(self.unet)
        else:
            conversion_counter = None

        device = get_device()
        latent = latent.to(device)
        conditioning = conditioning.to(device)
        noise_level = noise_level.to(device)

        if self.use_channels_last:
            latent = latent.to(memory_format=torch.channels_last)
            conditioning = conditioning.to(memory_format=torch.channels_last)

        counter_ctx = conversion_counter if should_log else contextlib.nullcontext()

        if self.use_amp_bf16:
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                with counter_ctx:
                    result = self.unet(
                        latent,
                        conditioning,
                        sigma=noise_level,
                        class_labels=None,
                    )
        else:
            with counter_ctx:
                result = self.unet(
                    latent,
                    conditioning,
                    sigma=noise_level,
                    class_labels=None,
                )

        if should_log:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time
            self._forward_log_count += 1
            logger.info(
                f"UNetDiffusionModule forward pass {self._forward_log_count}: "
                f"time={elapsed_time:.4f}s, "
                f"channels_last_to_contiguous="
                f"{conversion_counter.channels_last_to_contiguous}, "
                f"contiguous_to_channels_last="
                f"{conversion_counter.contiguous_to_channels_last}"
            )

        return result

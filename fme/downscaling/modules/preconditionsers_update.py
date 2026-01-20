import torch

from physicsnemo.models.diffusion.preconditioning import EDMPrecond as _EDMPrecond

class EDMPrecond(_EDMPrecond):
    def __init__(
        self,
        model,
        label_dim= 0,
        use_fp16=False,
        sigma_data=0.5,
    ):
        torch.nn.Module.__init__(self)
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, latent, conditioning, sigma, class_labels=None, force_fp32=False):
        # Rearranged order for physicsnemo compatibility
        # does not support **model_kwargs
        return super().forward(
            latent, sigma, condition=conditioning, class_labels=class_labels, force_fp32=force_fp32
        )
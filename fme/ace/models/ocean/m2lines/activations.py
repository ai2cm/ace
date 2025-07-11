import torch


class CappedGELU(torch.nn.Module):
    """
    Implements a GeLU with capped maximum value.
    """

    def __init__(self, cap_value: float = 1.0, **kwargs):
        """
        :param cap_value: float: value at which to clip activation
        :param kwargs: passed to torch.nn.LeadyReLU
        """
        super().__init__()
        self.add_module("gelu", torch.nn.GELU(**kwargs))
        # self.cap = torch.tensor(cap_value, dtype=torch.float32)
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor):
        x = self.gelu(inputs)
        # Convert cap to a scalar value for clamping (ignores grad)
        cap_value = self.cap.item()
        x = torch.clamp(x, max=cap_value)
        return x

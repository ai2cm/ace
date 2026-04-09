import dataclasses

import torch
from torch import nn

from .separated_module import SeparatedModuleConfig, SeparatedModuleSelector


class _SimpleSeparatedModule(nn.Module):
    """A trivial module for testing the separated interface."""

    def __init__(self, n_forcing: int, n_prognostic: int, n_diagnostic: int):
        super().__init__()
        n_in = n_forcing + n_prognostic
        self.prog_linear = nn.Linear(n_in, n_prognostic, bias=False)
        self.diag_linear = nn.Linear(n_in, n_diagnostic, bias=False)

    def forward(
        self,
        forcing: torch.Tensor,
        prognostic: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([forcing, prognostic], dim=-3)
        b, c, h, w = combined.shape
        flat = combined.permute(0, 2, 3, 1).reshape(-1, c)
        prog_out = self.prog_linear(flat).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        diag_out = self.diag_linear(flat).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return prog_out, diag_out


def register_test_types() -> None:
    """Register test-only module types. Call from tests before use."""

    @SeparatedModuleSelector.register("test_simple")
    @dataclasses.dataclass
    class _SimpleSeparatedBuilder(SeparatedModuleConfig):
        def build(
            self,
            n_forcing_channels,
            n_prognostic_channels,
            n_diagnostic_channels,
            dataset_info,
        ):
            return _SimpleSeparatedModule(
                n_forcing_channels, n_prognostic_channels, n_diagnostic_channels
            )

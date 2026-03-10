import numpy as np
import pytest
import torch
from torch import nn

import fme
from fme.core.distributed.distributed import Distributed
from fme.core.distributed.model_torch_distributed import ModelTorchDistributed
from fme.core.gridded_ops import LatLonOperations
from fme.core.optimization import OptimizationConfig
from fme.core.typing_ import TensorDict


class TinyConvNet(nn.Module):
    """
    Very small conv net that operates on [batch, channels, nlat, nlon].
    This is just to ensure gradients propagate through a nontrivial model.
    """

    def __init__(self, n_channels: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 4, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(4, n_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(x)))


def _build_latlon_ops(img_shape: tuple[int, int]) -> LatLonOperations:
    """
    Build LatLonOperations with simple area weights.
    In spatial-parallel mode, this will slice per-rank tiles internally.
    """
    nlat, nlon = img_shape
    # Use cos(lat) weights (approx) just to be realistic; could also use ones.
    lat = torch.linspace(-np.pi / 2, np.pi / 2, nlat, device="cpu")
    area_weights = torch.cos(lat).clamp_min(1e-3).unsqueeze(-1).expand(nlat, nlon)
    return LatLonOperations(area_weights=area_weights)


def _build_model_and_optimizer(
    img_shape: tuple[int, int],
) -> tuple[nn.Module, torch.optim.Optimizer, LatLonOperations]:
    """
    Build a DDP-wrapped TinyConvNet under ModelTorchDistributed and
    a simple optimizer. Also returns LatLonOperations for computing a global loss.
    """
    dist = Distributed.get_instance()
    assert isinstance(dist._distributed, ModelTorchDistributed)

    model = TinyConvNet(n_channels=2).to(fme.get_device())
    # Wrap with DDP over the data group only; spatial model parallelism is
    # handled by the model/layers and the backend.
    wrapped_model = dist._distributed.wrap_module(model)

    # Simple Adam optimizer via OptimizationConfig, to go through the same codepath
    # as real training.
    opt_config = OptimizationConfig(
        optimizer_type="Adam",
        lr=1e-3,
        enable_automatic_mixed_precision=False,
    )
    optimization = opt_config.build(
        modules=torch.nn.ModuleList([wrapped_model]),
        max_epochs=1,
    )

    gridded_ops = _build_latlon_ops(img_shape)
    return wrapped_model, optimization, gridded_ops


@pytest.mark.parametrize("img_shape", [(16, 32)])
@pytest.mark.parallel
def test_spatial_parallel_backward_step(img_shape):
    """
    Test: run forward + backward + optimizer step under
    ModelTorchDistributed with spatial parallelism.

    Asserts:
      - Loss is finite.
      - All data-parallel ranks see the same loss.
      - Parameter gradients are finite and data-parallel-consistent.
    """
    dist = Distributed.get_instance()
    if not isinstance(dist._distributed, ModelTorchDistributed):
        pytest.skip("ModelTorchDistributed backend is required for this test")

    torch.manual_seed(0)

    model, optimization, gridded_ops = _build_model_and_optimizer(img_shape)

    batch_size = 4
    n_channels = 2
    nlat, nlon = img_shape

    # Global tensors
    x_global = torch.randn(batch_size, n_channels, nlat, nlon, device=fme.get_device())
    y_global = torch.randn_like(x_global)

    global_inputs: TensorDict = {"x": x_global, "y": y_global}
    local_inputs = dist.scatter_spatial(global_inputs, img_shape=(nlat, nlon))

    x_local = local_inputs["x"]
    y_local = local_inputs["y"]

    # Forward pass + loss
    model.train()
    optimization.optimizer.zero_grad()

    with optimization.autocast():
        y_pred_local = model(x_local)

        # Compute a global, area-weighted MSE over [batch, channels, lat, lon],
        mse = (y_pred_local - y_local) ** 2
        mse_spatial = gridded_ops.area_weighted_mean(mse)
        loss = mse_spatial.mean()

    # Backward + optimizer step
    optimization.accumulate_loss(loss)
    loss_before_step = optimization.get_accumulated_loss().detach().clone()
    optimization.step_weights()

    # 1) Loss finite and the same on all data-parallel ranks.
    assert torch.isfinite(loss_before_step), "Loss is not finite on this rank"

    # Reduce mean loss across data group and broadcast to root for inspection.
    # ModelTorchDistributed.reduce_mean reduces over data group only.
    loss_reduced = dist.reduce_mean(loss_before_step.detach().clone())
    if dist.is_root():
        assert torch.isfinite(loss_reduced), "Reduced loss is not finite"

    # 2) Gradients finite and consistent across data-parallel ranks.
    # For a DDP-wrapped model, parameters are identical across data group,
    # so their gradients should also be identical after backward.
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), "Non-finite gradient detected"

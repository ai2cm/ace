import pathlib

import numpy as np
import pytest
import torch

import fme
from fme.core.distributed.distributed import Distributed
from fme.core.gridded_ops import LatLonOperations
from fme.core.typing_ import TensorDict

DATA_DIR = pathlib.Path(__file__).parent / "testdata"
BASELINE_FILE = DATA_DIR / "backward_step_baseline.pt"


def _run_forward_backward(
    img_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward + backward step and return:
      - loss on this rank
      - global gradient
    """
    dist = Distributed.get_instance()

    batch_size = 4
    n_channels = 2
    nlat, nlon = img_shape

    # Build 2D weights with correct spatial shape
    lat = torch.linspace(-np.pi / 2, np.pi / 2, nlat, device=fme.get_device())
    area_weights_lat = torch.cos(lat).clamp_min(1e-3)  # (nlat,)
    area_weights_global = area_weights_lat.unsqueeze(-1).repeat(1, nlon)  # (nlat, nlon)

    global_ops = LatLonOperations(area_weights=area_weights_global)

    # Global tensors
    torch.manual_seed(0)
    x_global = torch.randn(
        batch_size, n_channels, nlat, nlon, device=fme.get_device(), requires_grad=True
    )
    y_global = torch.randn_like(x_global)

    global_inputs: TensorDict = {
        "x": x_global,
        "y": y_global,
    }
    local_inputs = dist.scatter_spatial(global_inputs, img_shape=(nlat, nlon))

    x_local = local_inputs["x"]
    y_local = local_inputs["y"]
    x_local.retain_grad()

    sht = global_ops.get_real_sht().to(fme.get_device())
    isht = global_ops.get_real_isht().to(fme.get_device())
    # Forward: x -> sht -> isht -> y_pred
    y_hat_local = sht(x_local)
    y_pred_local = isht(y_hat_local)

    mse = (y_pred_local - y_local) ** 2
    # Global, area-weighted MSE over spatial dims via LatLonOperations
    mse_spatial = global_ops.area_weighted_mean(mse)
    loss = mse_spatial.mean()

    loss.backward()
    # Gather grad_x back to global grid
    grad_local = x_local.grad.detach()
    grad_global_dict = dist.gather_spatial({"x": grad_local}, img_shape=img_shape)
    grad_x_global = grad_global_dict["x"]

    return loss.detach().cpu(), grad_x_global.cpu()


@pytest.mark.parametrize("img_shape", [(16, 32)])
@pytest.mark.parallel
def test_spatial_parallel_backward_step(img_shape):
    """
    Test: run forward + backward under
    ModelTorchDistributed with spatial parallelism.

    Asserts:
      - Loss is same with sp decomp compared with NonDistributed baseline
      - Gradient is element-wise same with sp decomp compared with NonDistributed baseline
    """
    dist = Distributed.get_instance()
    torch.manual_seed(0)

    # Run forwards/backwards
    loss, grad = _run_forward_backward(img_shape)

    # Only root does I/O
    if not dist.is_root():
        return

    if not BASELINE_FILE.exists():
        # Baseline generation mode: expect non-distributed backend here.
        # Save loss and grads for later regression.
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "img_shape": img_shape,
                "loss": loss,
                "grad": grad,
            },
            BASELINE_FILE,
        )
        return

    # Regression mode: compare against existing baseline.
    baseline = torch.load(BASELINE_FILE, map_location="cpu")
    assert tuple(baseline["img_shape"]) == tuple(img_shape)

    baseline_loss = baseline["loss"].item()
    baseline_grad = baseline["grad"]

    # 1) Loss finite and close to baseline.
    assert torch.isfinite(loss), "Loss is not finite on this rank"

    # Compare loss (scalar) with a small relative tolerance
    actual_loss = loss.item()
    rel_loss = abs(actual_loss - baseline_loss) / max(abs(baseline_loss), 1e-12)
    assert rel_loss < 1e-6, (
        f"Loss deviates from baseline: "
        f"actual={actual_loss:.8f}, expected={baseline_loss:.8f}, rel_diff={rel_loss:.3e}"
    )
    max_rel = (
        ((grad - baseline_grad).abs() / baseline_grad.abs().clamp_min(1e-12))
        .max()
        .item()
    )
    assert torch.allclose(grad, baseline_grad, rtol=1e-6, atol=1e-7), (
        f"grad_x differs from baseline: "
        f"max_abs={(grad - baseline_grad).abs().max().item():.3e}, "
        f"max_rel={max_rel:.3e}"
    )

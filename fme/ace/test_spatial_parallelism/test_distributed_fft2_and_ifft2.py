import os

import torch
import torch.distributed as dist
import torch_harmonics.distributed as thd
from test_helper import _gather_helper, _split_helper, relative_error

from fme.ace.models.makani_mpu.fft import (
    DistributedInverseRealFFT2,
    DistributedRealFFT2,
)
from fme.ace.models.modulus.layers import InverseRealFFT2, RealFFT2
from fme.ace.utils import comm


def setup_test():
    if "LOCAL_RANK" in os.environ and dist.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

        if torch.cuda.is_available():
            if world_rank == 0:
                print("Running test on GPU")

            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            torch.cuda.manual_seed(333)
        else:
            if world_rank == 0:
                print("Running test on CPU")
            device = torch.device("cpu")
            torch.manual_seed(333)

        return world_rank, world_size, device

    else:
        world_rank = 0
        world_size = 1
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("Running test in single-process mode (Not launched by torchrun)")
        return world_rank, world_size, device


def _init_comms():
    # set up distributed
    grid_size_h = int(os.getenv("GRID_H", 1))
    grid_size_w = int(os.getenv("GRID_W", 1))
    grid_size_e = int(os.getenv("GRID_E", 1))

    if not dist.is_initialized() and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    comm.init(
        model_parallel_sizes=[grid_size_h, grid_size_w, 1, 1],
        model_parallel_names=["h", "w", "fin", "fout"],
        data_parallel_sizes=[grid_size_e, -1],
        data_parallel_names=["ensemble", "batch"],
    )
    # The world_rank returned here is now derived from the PDC environment
    world_rank = dist.get_rank()

    # store comm group parameters
    wrank = comm.get_rank("w")
    hrank = comm.get_rank("h")
    erank = comm.get_rank("ensemble")
    w_group = comm.get_group("w")
    h_group = comm.get_group("h")
    e_group = comm.get_group("ensemble")

    # thd.init also needs to work with PyTorch groups
    thd.init(h_group, w_group)

    if world_rank == 0:
        print(
            f"Running distributed tests on grid H x W x E = {grid_size_h} x {grid_size_w} x {grid_size_e}"
        )

    return (
        w_group,
        h_group,
        e_group,
        world_rank,
        dist.get_world_size(),
    )  # Use PDC world size


def test_distributed_fft2():
    verbose = True
    world_rank, world_size, device = setup_test()
    w_group, h_group, e_group, world_rank, world_size = _init_comms()

    # 256, 512, 0, 32,  8, 1e-6
    # nlat, nlon, nalt, batch_size, num_chan, tol,
    tol = 1e-6
    B, C, H, W = 32, 8, 256, 512

    # set up handles
    forward_transform_local = RealFFT2(nlat=H, nlon=W).to(device)
    forward_transform_dist = DistributedRealFFT2(nlat=H, nlon=W).to(device)

    # create tensors
    inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=device)

    #############################################################
    # local transform
    #############################################################
    # FWD pass
    inp_full.requires_grad = True
    out_full = forward_transform_local(inp_full)

    # create grad for backward
    with torch.no_grad():
        # create full grad
        ograd_full = torch.randn_like(out_full)

    # BWD pass
    out_full.backward(ograd_full)
    igrad_full = inp_full.grad.clone()

    #############################################################
    # distributed transform
    #############################################################
    # FWD pass
    inp_local = _split_helper(inp_full, w_group, h_group)
    inp_local.requires_grad = True
    out_local = forward_transform_dist(inp_local)

    # BWD pass
    ograd_local = _split_helper(ograd_full, w_group, h_group)
    out_local = forward_transform_dist(inp_local)
    out_local.backward(ograd_local)
    igrad_local = inp_local.grad.clone()

    # set eval dims
    dims = (-1, -2, -3)

    #############################################################
    # evaluate FWD pass
    #############################################################
    with torch.no_grad():
        out_gather_full = _gather_helper(out_local, w_group, h_group)
        err = relative_error(out_gather_full, out_full)
        if verbose and (world_rank == 0):
            print(f"final relative error of output: {err.item()}")
    assert err.item() <= tol

    #############################################################
    # evaluate BWD pass
    #############################################################
    with torch.no_grad():
        igrad_gather_full = _gather_helper(igrad_local, w_group, h_group)
        err = relative_error(igrad_gather_full, igrad_full)
        if verbose and (world_rank == 0):
            print(f"final relative error of gradients: {err.item()}")
    assert err.item() <= tol


def test_distributed_ifft2():
    verbose = True
    world_rank, world_size, device = setup_test()
    w_group, h_group, e_group, world_rank, world_size = _init_comms()
    # 256, 512, 0, 32,  8, 1e-6
    # nlat, nlon, nalt, batch_size, num_chan, tol,
    tol = 1e-6
    B, C, H, W = 32, 8, 256, 512
    forward_transform_local = RealFFT2(nlat=H, nlon=W).to(device)
    backward_transform_local = InverseRealFFT2(nlat=H, nlon=W).to(device)
    backward_transform_dist = DistributedInverseRealFFT2(nlat=H, nlon=W).to(device)

    # create tensors
    dummy_full = torch.randn((B, C, H, W), dtype=torch.float32, device=device)
    inp_full = forward_transform_local(dummy_full)

    #############################################################
    # local transform
    #############################################################
    # FWD pass
    inp_full.requires_grad = True
    out_full = backward_transform_local(inp_full)

    # create grad for backward
    with torch.no_grad():
        # create full grad
        ograd_full = torch.randn_like(out_full)

    # BWD pass
    out_full.backward(ograd_full)

    # repeat once due to known irfft bug
    inp_full.grad = None
    out_full = backward_transform_local(inp_full)
    out_full.backward(ograd_full)
    igrad_full = inp_full.grad.clone()

    #############################################################
    # distributed transform
    #############################################################
    # FWD pass
    inp_local = _split_helper(inp_full, w_group, h_group)
    inp_local.requires_grad = True
    out_local = backward_transform_dist(inp_local)

    # BWD pass
    ograd_local = _split_helper(ograd_full, w_group, h_group)
    out_local = backward_transform_dist(inp_local)
    out_local.backward(ograd_local)
    igrad_local = inp_local.grad.clone()

    # set eval dims
    dims = (-1, -2, -3)

    #############################################################
    # evaluate FWD pass
    #############################################################
    with torch.no_grad():
        out_gather_full = _gather_helper(out_local, w_group, h_group)
        err = relative_error(out_gather_full, out_full)
        if verbose and (world_rank == 0):
            print(f"final relative error of output: {err.item()}")
    assert err.item() <= tol

    #############################################################
    # evaluate BWD pass
    #############################################################
    with torch.no_grad():
        igrad_gather_full = _gather_helper(igrad_local, w_group, h_group)
        err = relative_error(igrad_gather_full, igrad_full)
        if verbose and (world_rank == 0):
            print(f"final relative error of gradients: {err.item()}")
    assert err.item() <= tol
    comm.cleanup()

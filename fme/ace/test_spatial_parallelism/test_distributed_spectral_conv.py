import os

import torch
import torch_harmonics as th
import torch_harmonics.distributed as thd
from test_helper import gather_helper_conv, init_seed, relative_error, split_helper_conv

from fme.ace.models.makani_mpu.mappings import init_gradient_reduction_hooks
from fme.ace.models.modulus.s2convolutions import SpectralConvS2

# import torch.distributed as dist
from fme.core.device import get_device
from fme.core.distributed import Distributed

DIR = os.path.abspath(os.path.dirname(__file__))


def setup_test():
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD.Dup()
    mpi_comm_rank = mpi_comm.Get_rank()
    mpi_comm_size = mpi_comm.Get_size()
    # if torch.cuda.is_available():
    #   if mpi_comm_rank == 0:
    #     print("Running test on GPU")
    #   local_rank = mpi_comm_rank % torch.cuda.device_count()
    #   device = torch.device(f"cuda:{local_rank}")
    #   torch.cuda.set_device(device)
    #   torch.cuda.manual_seed(333)
    # else:
    #   if mpi_comm_rank == 0:
    #     print("Running test on CPU")
    # device = torch.device("cpu")
    torch.manual_seed(333)
    return mpi_comm, device


def _init_comms():
    # set up distributed
    os.environ["GRID_H"] = "2"
    os.environ["GRID_W"] = "2"
    os.environ["H_PARALLEL_SIZE"] = "2"
    os.environ["W_PARALLEL_SIZE"] = "2"
    grid_size_h = int(os.getenv("GRID_H", 1))
    grid_size_w = int(os.getenv("GRID_W", 1))
    grid_size_e = int(os.getenv("GRID_E", 1))
    world_size = grid_size_h * grid_size_w * grid_size_e

    # init groups
    dist = Distributed.get_instance()
    world_rank = dist.rank

    # store comm group parameters
    wrank = dist.comm_get_rank("w")
    hrank = dist.comm_get_rank("h")
    erank = dist.comm_get_rank("ensemble")
    w_group = dist.comm_get_group("w")
    h_group = dist.comm_get_group("h")
    e_group = dist.comm_get_group("ensemble")

    if world_rank == 0:
        print(
            f"Running distributed tests on grid H x W x E = {grid_size_h} x {grid_size_w} x {grid_size_e}"
        )

    return w_group, h_group, e_group, world_rank, world_size


def test_distributed_spectral_conv():
    tol = 1e-6
    verbose = True
    # mpi_comm, device = setup_test()
    device = get_device()
    w_group, h_group, e_group, world_rank, world_size = _init_comms()
    # set up handles
    B, C, Hi, Wi, Ho, Wo = 32, 8, 256, 512, 256, 512
    print("world_rank", world_rank)
    print("world_size", world_size)

    # input
    init_seed(444)
    inp_full = torch.randn((B, C, Hi, Wi), dtype=torch.float32, device=device)

    init_seed(333)

    ## without domain decomposition
    with Distributed.force_non_distributed():
        forward_transform_local = th.RealSHT(nlat=Hi, nlon=Wi).to(device)
        inverse_transform_local = th.InverseRealSHT(
            nlat=Ho,
            nlon=Wo,
            lmax=forward_transform_local.lmax,
            mmax=forward_transform_local.mmax,
        ).to(device)

        spect_conv_local = SpectralConvS2(
            forward_transform_local,
            inverse_transform_local,
            C,
            C,
            operator_type="dhconv",
            use_tensorly=False,
            bias=True,
        ).to(device)

        # #############################################################
        # # local transform
        # #############################################################
        # # FWD pass
        inp_full.requires_grad = True
        out_full, _ = spect_conv_local(inp_full)
        # create grad for backward
        init_seed(555)
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        wgrad_full = spect_conv_local.weight.grad.clone()
        bgrad_full = spect_conv_local.bias.grad.clone()

    forward_transform_dist = thd.DistributedRealSHT(nlat=Hi, nlon=Wi).to(device)
    inverse_transform_dist = thd.DistributedInverseRealSHT(
        nlat=Ho,
        nlon=Wo,
        lmax=forward_transform_dist.lmax,
        mmax=forward_transform_dist.mmax,
    ).to(device)

    spect_conv_dist = SpectralConvS2(
        forward_transform_dist,
        inverse_transform_dist,
        C,
        C,
        operator_type="dhconv",
        use_tensorly=False,
        bias=True,
    ).to(device)
    # set up wgrad reductions
    spect_conv_dist = init_gradient_reduction_hooks(
        spect_conv_dist,
        device=device,
        reduction_buffer_count=1,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
        verbose=False,
    )
    # make sure weights are the same:
    with torch.no_grad():
        weight = split_helper_conv(
            spect_conv_local.weight,
            hdim=-2,
            wdim=None,
            w_group=w_group,
            h_group=h_group,
        )
        print("spect_conv_local.weight", spect_conv_local.weight.shape)
        print("weight", weight.shape)
        print("spect_conv_dist.module.weight", spect_conv_dist.module.weight.shape)
        spect_conv_dist.module.weight.copy_(weight)
        spect_conv_dist.module.bias.copy_(spect_conv_local.bias)

    #############################################################
    # distributed transform
    #############################################################
    # FWD pass
    inp_local = split_helper_conv(
        inp_full, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
    )
    print("inp_local", inp_local.shape)
    print("inp_full", inp_full.shape)
    inp_local.requires_grad = True
    out_local, _ = spect_conv_dist(inp_local)

    # BWD pass
    ograd_local = split_helper_conv(
        ograd_full, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
    )
    print("ograd_local", ograd_local.shape)
    print("ograd_full", ograd_full.shape)
    out_local, _ = spect_conv_dist(inp_local)
    out_local.backward(ograd_local)
    igrad_local = inp_local.grad.clone()
    wgrad_local = spect_conv_dist.module.weight.grad.clone()
    bgrad_local = spect_conv_dist.module.bias.grad.clone()
    dist.barrier()
    #############################################################
    # evaluate FWD pass
    #############################################################
    with torch.no_grad():
        out_gather_full = gather_helper_conv(
            out_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
        )
        err = relative_error(out_gather_full, out_full)
        if verbose and (world_rank == 0):
            print(f"final relative error of output: {err.item()}")
        # self.assertTrue(err.item() <= tol)
    assert err.item() <= tol
    dist.barrier()
    #############################################################
    # evaluate input grads
    #############################################################
    with torch.no_grad():
        igrad_gather_full = gather_helper_conv(
            igrad_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
        )
        err = relative_error(igrad_gather_full, igrad_full)
        if verbose and (world_rank == 0):
            print(f"final relative error of input gradients: {err.item()}")
    assert err.item() <= tol
    # self.assertTrue(err.item() <= tol)
    dist.barrier()
    #############################################################
    # evaluate Weight grads
    #############################################################
    with torch.no_grad():
        wgrad_gather_full = gather_helper_conv(
            wgrad_local, hdim=-2, wdim=None, w_group=w_group, h_group=h_group
        )
        print("wgrad_gather_full", wgrad_local.shape)
        print("wgrad_gather_full", wgrad_gather_full.shape)
        err = relative_error(wgrad_gather_full, wgrad_full)
        if verbose and (world_rank == 0):
            print(f"final relative error of weight gradients: {err.item()}")
        # self.assertTrue(err.item() <= tol)
    assert err.item() <= tol
    dist.barrier()

    with torch.no_grad():
        bgrad_gather_list = [torch.empty_like(bgrad_local) for _ in range(world_size)]
        bgrad_gather_list[world_rank] = bgrad_local
        dist.all_gather(bgrad_gather_list, bgrad_local, group=None)
        errs = []
        for bgrad_gather_full in bgrad_gather_list:
            errs.append(relative_error(bgrad_gather_full, bgrad_full))
        err = torch.mean(torch.stack(errs, dim=0))
        if verbose and (world_rank == 0):
            print(f"final relative error of bias gradients: {err.item()}")
    assert err.item() <= tol
    dist.shutdown()

import os

import pytest
import torch

from fme.ace.models.modulus.sfnonet import SFNO
from fme.core.device import get_device

DIR = os.path.abspath(os.path.dirname(__file__))

from physicsnemo.distributed.mappings import reduce_from_parallel_region
from test_helper import gather_helper_conv, init_seed, relative_error

from fme.ace.models.makani_mpu.mappings import init_gradient_reduction_hooks
from fme.ace.models.makani_utils.makani_driver import (
    _restore_checkpoint_flexible,
    _save_checkpoint_flexible,
)
from fme.core.distributed import Distributed


def test_sfnonet_without_sp(distributed):
    if distributed:
        pytest.skip("Disable serial tests when distributed tests are enabled")
    init_seed(333)
    ## without domain decomposition
    os.environ["H_PARALLEL_SIZE"] = "1"
    os.environ["W_PARALLEL_SIZE"] = "1"
    verbose = False
    input_channels = 3
    output_channels = 3
    img_shape = (8, 16)
    n_samples = 4
    embed_dim = 16
    num_layers = 2
    with Distributed.force_non_distributed():
        model = SFNO(
            params=None,
            embed_dim=embed_dim,
            num_layers=num_layers,
            # operator_type="dhconv",
            # normalization_layer="layer_norm",
            img_shape=img_shape,
            in_chans=input_channels,
            out_chans=output_channels,
        )
        # must initialize on CPU to get the same results on GPU
        inp_full = torch.randn(n_samples, input_channels, *img_shape)
        inp_full.requires_grad = True
        # with torch.no_grad():
        out_full = model(inp_full)
        loss_full = torch.sum(out_full)

        # perform backward pass
        loss_full.backward()
        igrad_full = inp_full.grad.clone()

        assert out_full.shape == (n_samples, output_channels, *img_shape)
        tmp_path = "testdata"
        os.makedirs(tmp_path, exist_ok=True)
        torch.save(out_full, os.path.join(tmp_path, "out_full.pt"))
        torch.save(inp_full, os.path.join(tmp_path, "inp_full.pt"))
        torch.save(loss_full, os.path.join(tmp_path, "loss_full.pt"))
        torch.save(igrad_full, os.path.join(tmp_path, "igrad_full.pt"))

        _save_checkpoint_flexible(
            checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"), model=model
        )


def test_sfnonet_with_sp(distributed):
    if not distributed:
        pytest.skip("Distributed tests are not enabled")
    init_seed(333)
    tmp_path = "testdata"
    os.environ["H_PARALLEL_SIZE"] = "2"
    os.environ["W_PARALLEL_SIZE"] = "1"
    verbose = False
    input_channels = 3
    output_channels = 3
    img_shape = (8, 16)
    n_samples = 4
    embed_dim = 16
    num_layers = 2
    if not torch.cuda.is_available():
        # physicsnemo DistributedManager assumes that the device_id is a GPU
        # so we override the init_process_group function to not pass in device_id
        import torch.distributed as dist

        orig_init = dist.init_process_group

        def cpu_friendly_init(*args, **kwargs):
            if (
                "device_id" in kwargs
                and getattr(kwargs["device_id"], "type", None) == "cpu"
            ):
                kwargs.pop("device_id")
            return orig_init(*args, **kwargs)

        dist.init_process_group = cpu_friendly_init
    dist = Distributed.get_instance()
    mpi_comm_rank = dist.local_rank

    w_group = dist.comm_get_group("w")
    h_group = dist.comm_get_group("h")
    world_rank = dist.rank

    device = get_device()

    model_dist = SFNO(
        params=None,
        embed_dim=embed_dim,
        num_layers=num_layers,
        # operator_type="dhconv",
        # normalization_layer="layer_norm",
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
    ).to(device)

    # save reduction hooks
    model_dist = init_gradient_reduction_hooks(
        model_dist,
        device=device,
        reduction_buffer_count=1,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
        verbose=True,
    )

    # load checkpoint
    _restore_checkpoint_flexible(
        checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"), model=model_dist
    )

    # must initialize on CPU to get the same results on GPU
    # inp_full = torch.randn(n_samples, input_channels, *img_shape)
    inp_full = torch.load(os.path.join(tmp_path, "inp_full.pt"))
    dist.barrier()

    # split input
    # inputs: ntimes, nsamples, h, w
    this_shape = (inp_full.shape[-2], inp_full.shape[-1])
    ## Create a leaf variable
    inp_local_host = (
        (inp_full[:, :, *dist.get_local_slices(this_shape)]).detach().clone()
    )
    inp_local = inp_local_host.to(device)
    inp_local.requires_grad = True
    if world_rank == 0:
        print("inp_full", inp_full.shape)
        print("inp_local", inp_local.shape)
    dist.barrier()
    out_local = model_dist(inp_local)
    loss_dist = reduce_from_parallel_region(torch.sum(out_local), "model")
    loss_dist.backward()
    igrad_local = inp_local.grad.clone()

    out_full = torch.load(os.path.join(tmp_path, "out_full.pt"))
    dist.barrier()

    with torch.no_grad():
        out_full_device = out_full.to(device)
        out_gather_full = gather_helper_conv(
            out_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
        )
        err = relative_error(out_gather_full, out_full_device)
        if world_rank == 0:
            print(f"final relative error of output: {err.item()}")
    assert err < 1e-3
    dist.barrier()
    loss_full = torch.load(os.path.join(tmp_path, "loss_full.pt"))

    with torch.no_grad():
        loss_full_device = loss_full.to(device)
        err = relative_error(loss_dist, loss_full)
        if world_rank == 0:
            print(f"final relative error of loss: {err.item()}")
    # mpi_comm.Barrier()
    dist.barrier()
    assert err < 1e-3

    #############################################################
    # evaluate BWD pass
    #############################################################
    # dgrad
    igrad_full = torch.load(os.path.join(tmp_path, "igrad_full.pt"))
    with torch.no_grad():
        igrad_full_device = igrad_full.to(device)
        igrad_gather_full = gather_helper_conv(
            igrad_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
        )
        err = relative_error(igrad_gather_full, igrad_full_device)
        if world_rank == 0:
            print(f"final relative error of input gradient: {err.item()}")
        # cleanup
    dist.barrier()
    assert err < 1e-3


def test_sfnonet_spatial_dist_output_is_unchanged(distributed):
    if distributed:
        pytest.skip("Disable serial tests when distributed tests are enabled")
    # torch.manual_seed(0)
    # fix seed
    init_seed(333)
    ## without domain decomposition
    verbose = False
    input_channels = 3
    output_channels = 3
    img_shape = (8, 16)
    n_samples = 4
    embed_dim = 16
    num_layers = 2
    with Distributed.force_non_distributed():
        model = SFNO(
            params=None,
            embed_dim=embed_dim,
            num_layers=num_layers,
            # operator_type="dhconv",
            # normalization_layer="layer_norm",
            img_shape=img_shape,
            in_chans=input_channels,
            out_chans=output_channels,
        )
        # must initialize on CPU to get the same results on GPU
        inp_full = torch.randn(n_samples, input_channels, *img_shape)
        inp_full.requires_grad = True
        # with torch.no_grad():
        out_full = model(inp_full)
        loss_full = torch.sum(out_full)

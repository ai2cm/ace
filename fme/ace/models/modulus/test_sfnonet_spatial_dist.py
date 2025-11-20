import os
import sys
import torch
from fme.core.device import get_device

from .sfnonet import SFNO

DIR = os.path.abspath(os.path.dirname(__file__))

from fme.ace.models.makani_mpu.mappings import init_gradient_reduction_hooks
from fme.core.distributed import Distributed
from fme.core.dataset.test_helper import gather_helper_conv, relative_error, init_seed
from fme.ace.models.makani_utils.makani_driver import _save_checkpoint_flexible, _restore_checkpoint_flexible
from physicsnemo.distributed.mappings import reduce_from_parallel_region

def test_sfnonet_without_sp():
    ## without domain decomposition
    os.environ['H_PARALLEL_SIZE'] = '1'
    verbose=False
    input_channels = 3
    output_channels = 3
    img_shape = (8, 16)
    n_samples = 4
    embed_dim=16
    num_layers=2

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
    tmp_path="testdata"
    torch.save(out_full, os.path.join(tmp_path, "out_full.pt"))
    torch.save(inp_full, os.path.join(tmp_path, "inp_full.pt"))
    torch.save(loss_full, os.path.join(tmp_path, "loss_full.pt"))
    torch.save(igrad_full, os.path.join(tmp_path, "igrad_full.pt"))

    _save_checkpoint_flexible(checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"),
                                          model=model)

def test_sfnonet_with_sp():
    tmp_path="testdata"
    os.environ['H_PARALLEL_SIZE'] = '2'
    verbose=False
    input_channels = 3
    output_channels = 3
    img_shape = (8, 16)
    n_samples = 4
    embed_dim=16
    num_layers=2

    dist = Distributed.get_instance()
    mpi_comm_rank = dist.local_rank

    w_group = dist.comm_get_group("w")
    h_group = dist.comm_get_group("h")
    world_rank = dist.rank

    device=get_device()

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
    _restore_checkpoint_flexible(checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"),
                                model=model_dist)

    # must initialize on CPU to get the same results on GPU
    # inp_full = torch.randn(n_samples, input_channels, *img_shape)
    inp_full = torch.load(os.path.join(tmp_path, "inp_full.pt"))

    # split input
    # inputs: ntimes, nsamples, h, w
    this_shape=(inp_full.shape[-2],inp_full.shape[-1])
    ## Create a leaf variable
    inp_local_host = (inp_full[:,:,*dist.get_local_slices(this_shape)]).detach().clone()
    inp_local=inp_local_host.to(device)
    inp_local.requires_grad = True
    if world_rank == 0:
      print("inp_full", inp_full.shape)
      print("inp_local", inp_local.shape)

    out_local = model_dist(inp_local)
    loss_dist = reduce_from_parallel_region(torch.sum(out_local), "model")
    loss_dist.backward()
    igrad_local = inp_local.grad.clone()

    out_full = torch.load(os.path.join(tmp_path, "out_full.pt"))

    with torch.no_grad():
      out_full_device=out_full.to(device)
      out_gather_full = gather_helper_conv(out_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
      err = relative_error(out_gather_full, out_full_device)
      if world_rank == 0:
        print(f"final relative error of output: {err.item()}")
    assert err < 0.0006

    loss_full=torch.load(os.path.join(tmp_path, "loss_full.pt"))

    with torch.no_grad():
      loss_full_device=loss_full.to(device)
      err = relative_error(loss_dist, loss_full)
      if (world_rank == 0):
        print(f"final relative error of loss: {err.item()}")
    # mpi_comm.Barrier()
    assert err < 1e-3

    #############################################################
    # evaluate BWD pass
    #############################################################
    # dgrad
    igrad_full = torch.load(os.path.join(tmp_path, "igrad_full.pt"))
    with torch.no_grad():
      igrad_full_device=igrad_full.to(device)
      igrad_gather_full = gather_helper_conv(igrad_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
      err = relative_error(igrad_gather_full, igrad_full_device)
      if (world_rank == 0):
        print(f"final relative error of input gradient: {err.item()}")
      # cleanup
    assert err < 1e-3

def test_sfnonet_spatial_dist_output_is_unchanged():
    # torch.manual_seed(0)
    # fix seed
    init_seed(333)
    ## without domain decomposition
    verbose=False
    input_channels = 3
    output_channels = 3
    img_shape = (8, 16)
    n_samples = 4
    embed_dim=16
    num_layers=2
    with Distributed.non_distributed():
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

    # # perform backward pass
    # loss_full.backward()
    # igrad_full = inp_full.grad.clone()

    # assert out_full.shape == (n_samples, output_channels, *img_shape)
    # tmp_path="testdata"
    # torch.save(out_full, "testdata/test_sfnonet_spatial_dist_output_is_unchanged.pt")

    # # get state dict
    # state_dict_full = checkpoint_helpers.gather_model_state_dict(model, grads=False)


    # torch.save(out_full, os.path.join(tmp_path, "out_full.pt"))
    # # torch.save(igrad_full, os.path.join(tmp_path, "igrad_full.pt"))
    # # if mpi_comm_rank == 0:
    # _save_checkpoint_flexible(checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"),
    #                                       model=model)
    # # delete local model
    # del model
    # print("--------------------------------------------------")

    # dist.shutdown()

    # ## with domain decomposition
    # dist = Distributed()
    # mpi_comm_rank = dist.local_rank

    # h_parallel_size=2
    # w_parallel_size=2
    # dist._init_distributed(h_parallel_size =  h_parallel_size, w_parallel_size=w_parallel_size)
    # # thd.init(h_parallel_size, w_parallel_size)


    # comm = dist.get_comm()
    # w_group = comm.get_group("w")
    # h_group = comm.get_group("h")
    # world_rank = comm.get_world_rank()
    # device=get_device()

    # model_dist = SFNO(
    #     params=None,
    #     embed_dim=embed_dim,
    #     num_layers=num_layers,
    #     # operator_type="dhconv",
    #     # normalization_layer="layer_norm",
    #     img_shape=img_shape,
    #     in_chans=input_channels,
    #     out_chans=output_channels,
    # ).to(device)


    # # save reduction hooks
    # model_dist = init_gradient_reduction_hooks(
    #         model_dist,
    #         device=device,
    #         reduction_buffer_count=1,
    #         broadcast_buffers=False,
    #         find_unused_parameters=False,
    #         gradient_as_bucket_view=True,
    #         static_graph=True,
    #         verbose=True,
    # )

    # # load checkpoint
    # _restore_checkpoint_flexible(checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"),
    #                             model=model_dist)

    # # split input
    # inp_full_device=inp_full.to(device)
    # inp_local= split_helper_conv(inp_full_device, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
    # inp_local.requires_grad = True
    # if world_rank == 0:
    #   print("inp_full", inp_full.shape)
    #   print("inp_local", inp_local.shape)

    # # with torch.no_grad():
    # out_local = model_dist(inp_local)
    # loss_dist = reduce_from_parallel_region(torch.sum(out_local), "model")
    # loss_dist.backward()
    # igrad_local = inp_local.grad.clone()

    # # get weights and wgrads
    # state_dict_gather_full = checkpoint_helpers.gather_model_state_dict(model_dist, grads=True)

    # # output
    # # mpi_comm.Barrier()
    # with torch.no_grad():
    #   out_full_device=out_full.to(device)
    #   out_gather_full = gather_helper_conv(out_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
    #   err = relative_error(out_gather_full, out_full_device)
    #   if world_rank == 0:
    #     print(f"final relative error of output: {err.item()}")
    # # mpi_comm.Barrier()
    # assert err < 1e-3
    # # loss
    # with torch.no_grad():
    #   loss_full_device=loss_full.to(device)
    #   err = relative_error(loss_dist, loss_full)
    #   if (world_rank == 0):
    #     print(f"final relative error of loss: {err.item()}")
    # # mpi_comm.Barrier()
    # assert err < 1e-3
    # #############################################################
    # # evaluate BWD pass
    # #############################################################
    # # dgrad
    # with torch.no_grad():
    #   igrad_full_device=igrad_full.to(device)
    #   igrad_gather_full = gather_helper_conv(igrad_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
    #   err = relative_error(igrad_gather_full, igrad_full_device)
    #   if (world_rank == 0):
    #     print(f"final relative error of input gradient: {err.item()}")
    #   # cleanup
    # assert err < 1e-3
    # # mpi_comm.Barrier()

    # comm.cleanup()

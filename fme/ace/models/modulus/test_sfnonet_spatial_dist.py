import os

import torch
import torch.distributed as dist
from fme.core.device import get_device
from fme.core.testing import validate_tensor

from .sfnonet import SphericalFourierNeuralOperatorNet, SFNO

DIR = os.path.abspath(os.path.dirname(__file__))

from .layers import MLP, DropPath, RealFFT2, SpectralAttention2d, InverseRealFFT2
from .s2convolutions import SpectralAttentionS2, SpectralConvS2

from fme.ace.models.makani_mpu.fft import DistributedRealFFT1, DistributedInverseRealFFT1, DistributedRealFFT2, DistributedInverseRealFFT2, DistributedRealFFT3, DistributedInverseRealFFT3
from fme.ace.models.makani_mpu.mappings import init_gradient_reduction_hooks

from fme.ace.utils import comm
import torch_harmonics as th
import torch_harmonics.distributed as thd
from physicsnemo.distributed.utils import split_tensor_along_dim
from fme.ace.models.makani_utils import checkpoint_helpers
from fme.ace.models.makani_utils.makani_driver import _save_checkpoint_flexible, _restore_checkpoint_flexible
from physicsnemo.distributed.mappings import reduce_from_parallel_region


# this computes a relative error compatible with torch.allclose or np.allclose
def relative_error(tensor1, tensor2):
    return torch.sum(torch.abs(tensor1-tensor2)) / torch.sum(torch.abs(tensor2))

# this computes an absolute error compatible with torch.allclose or np.allclose
def absolute_error(tensor1, tensor2):
    return torch.max(torch.abs(tensor1-tensor2))

def setup_test():
  from mpi4py import MPI
  mpi_comm = MPI.COMM_WORLD.Dup()
  mpi_comm_rank = mpi_comm.Get_rank()
  mpi_comm_size = mpi_comm.Get_size()
  if torch.cuda.is_available():
    if mpi_comm_rank == 0:
      print("Running test on GPU")
    local_rank = mpi_comm_rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(333)
  else:
    if mpi_comm_rank == 0:
      print("Running test on CPU")
    device = torch.device("cpu")
    torch.manual_seed(333)
  return mpi_comm, device


def _init_comms():
  # set up distributed
  grid_size_h = int(os.getenv("GRID_H", 1))
  grid_size_w = int(os.getenv("GRID_W", 1))
  grid_size_e = int(os.getenv("GRID_E", 1))
  world_size = grid_size_h * grid_size_w * grid_size_e

  # init groups
  comm.init(
            model_parallel_sizes=[grid_size_h, grid_size_w, 1, 1],
            model_parallel_names=["h", "w", "fin", "fout"],
            data_parallel_sizes=[grid_size_e, -1],
            data_parallel_names=["ensemble", "batch"],
  )
  world_rank = comm.get_world_rank()

  # store comm group parameters
  wrank = comm.get_rank("w")
  hrank = comm.get_rank("h")
  erank = comm.get_rank("ensemble")
  w_group = comm.get_group("w")
  h_group = comm.get_group("h")
  e_group = comm.get_group("ensemble")
  # initializing sht process groups just to be sure
  thd.init(h_group, w_group)

  if world_rank == 0:
    print(f"Running distributed tests on grid H x W x E = {grid_size_h} x {grid_size_w} x {grid_size_e}")

  return w_group, h_group, e_group, world_rank

def _split_helper_conv(tensor, hdim=-2, wdim=-1, w_group=1, h_group=1):
  tensor_local = split_helper(tensor, dim=hdim, group=h_group)
  tensor_local = split_helper(tensor_local, dim=wdim, group=w_group)
  return tensor_local


def _gather_helper_conv(tensor, hdim=-2, wdim=-1, w_group=1, h_group=1):
  tensor_gather = gather_helper(tensor, dim=hdim, group=h_group)
  tensor_gather = gather_helper(tensor_gather, dim=wdim, group=w_group)
  return tensor_gather

def split_helper(tensor, dim=None, group=None):
    with torch.no_grad():
        if (dim is not None) and dist.get_world_size(group=group):
            gsize = dist.get_world_size(group=group)
            grank = dist.get_rank(group=group)
            # split in dim
            tensor_list_local = split_tensor_along_dim(tensor, dim=dim, num_chunks=gsize)
            tensor_local = tensor_list_local[grank]
        else:
            tensor_local = tensor.clone()

    return tensor_local


def gather_helper(tensor, dim=None, group=None):
    # get shapes
    if (dim is not None) and (dist.get_world_size(group=group) > 1):
        gsize = dist.get_world_size(group=group)
        grank = dist.get_rank(group=group)
        shape_loc = torch.tensor([tensor.shape[dim]], dtype=torch.long, device=tensor.device)
        shape_list = [torch.empty_like(shape_loc) for _ in range(dist.get_world_size(group=group))]
        shape_list[grank] = shape_loc
        dist.all_gather(shape_list, shape_loc, group=group)
        tshapes = []
        for ids in range(gsize):
            tshape = list(tensor.shape)
            tshape[dim] = shape_list[ids].item()
            tshapes.append(tuple(tshape))
        tens_gather = [torch.empty(tshapes[ids], dtype=tensor.dtype, device=tensor.device) for ids in range(gsize)]
        tens_gather[grank] = tensor
        dist.all_gather(tens_gather, tensor, group=group)
        tensor_gather = torch.cat(tens_gather, dim=dim)
    else:
        tensor_gather = tensor.clone()

    return tensor_gather
def _init_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
    return

def test_sfnonet_spatial_dist_output_is_unchanged():
    # torch.manual_seed(0)
    # fix seed
    _init_seed(333)
    mpi_comm, device = setup_test()
    mpi_comm_rank = mpi_comm.Get_rank()
    verbose=False
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
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
    ).to(device)
    # must initialize on CPU to get the same results on GPU
    inp_full = torch.randn(n_samples, input_channels, *img_shape).to(device)
    inp_full.requires_grad = True
    # with torch.no_grad():
    out_full = model(inp_full)
    loss_full = torch.sum(out_full)

    # perform backward pass
    loss_full.backward()
    igrad_full = inp_full.grad.clone()

    assert out_full.shape == (n_samples, output_channels, *img_shape)
    tmp_path="testdata"
    torch.save(out_full, "testdata/test_sfnonet_spatial_dist_output_is_unchanged.pt")

    # get state dict
    state_dict_full = checkpoint_helpers.gather_model_state_dict(model, grads=False)


    torch.save(out_full, os.path.join(tmp_path, "out_full.pt"))
    # torch.save(igrad_full, os.path.join(tmp_path, "igrad_full.pt"))
    if mpi_comm_rank == 0:
      _save_checkpoint_flexible(checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"),
                                          model=model)
    # delete local model
    del model
    mpi_comm.Barrier()
    print("--------------------------------------------------")

    w_group, h_group, e_group, world_rank = _init_comms()
    print("comm.get_size(matmul)",comm.get_size("matmul"))

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

    # split input
    inp_local= _split_helper_conv(inp_full, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
    inp_local.requires_grad = True
    if world_rank == 0:
      print("inp_full", inp_full.shape)
      print("inp_local", inp_local.shape)

    # with torch.no_grad():
    out_local = model_dist(inp_local)
    loss_dist = reduce_from_parallel_region(torch.sum(out_local), "model")
    loss_dist.backward()
    igrad_local = inp_local.grad.clone()

    # get weights and wgrads
    state_dict_gather_full = checkpoint_helpers.gather_model_state_dict(model_dist, grads=True)

    # output
    if world_rank == 0:
      print("world_rank",world_rank)
    mpi_comm.Barrier()
    with torch.no_grad():
      out_gather_full = _gather_helper_conv(out_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
      err = relative_error(out_gather_full, out_full)
      if world_rank == 0:
        print(f"final relative error of output: {err.item()}")
    mpi_comm.Barrier()
    assert err < 1e-6
    # loss
    with torch.no_grad():
      err = relative_error(loss_dist, loss_full)
      if verbose and (world_rank == 0):
        print(f"final relative error of loss: {err.item()}")
    mpi_comm.Barrier()
    assert err < 1e-6
    #############################################################
    # evaluate BWD pass
    #############################################################
    # dgrad
    with torch.no_grad():
      igrad_gather_full = _gather_helper_conv(igrad_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
      err = relative_error(igrad_gather_full, igrad_full)
      if verbose and (world_rank == 0):
        print(f"final relative error of input gradient: {err.item()}")
      # cleanup
    assert err < 1e-3
    mpi_comm.Barrier()

    comm.cleanup()

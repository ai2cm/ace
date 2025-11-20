import os

import torch
import torch.distributed as dist
from fme.core.device import get_device
from fme.core.testing import validate_tensor

from .sfnonet import SphericalFourierNeuralOperatorNet, SFNO


from .layers import MLP, DropPath, RealFFT2, SpectralAttention2d, InverseRealFFT2
from .s2convolutions import SpectralAttentionS2, SpectralConvS2

from fme.ace.models.makani_mpu.fft import DistributedRealFFT1, DistributedInverseRealFFT1, DistributedRealFFT2, DistributedInverseRealFFT2, DistributedRealFFT3, DistributedInverseRealFFT3
from fme.ace.models.makani_mpu.mappings import init_gradient_reduction_hooks

from fme.ace.utils import comm
import torch_harmonics as th
import torch_harmonics.distributed as thd
from physicsnemo.distributed.utils import split_tensor_along_dim
DIR = os.path.abspath(os.path.dirname(__file__))

# this computes a relative error compatible with torch.allclose or np.allclose
def relative_error(tensor1, tensor2):
    return torch.sum(torch.abs(tensor1-tensor2)) / torch.sum(torch.abs(tensor2))

# this computes an absolute error compatible with torch.allclose or np.allclose
def absolute_error(tensor1, tensor2):
    return torch.max(torch.abs(tensor1-tensor2))

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

def _split_helper(tensor, w_group, h_group):
  tensor_local = split_helper(tensor, dim=-1, group=w_group)
  tensor_local = split_helper(tensor_local, dim=-2, group=h_group)
  return tensor_local


def _gather_helper(tensor, w_group, h_group):
  tensor_gather = gather_helper(tensor, dim=-2, group=h_group)
  tensor_gather =	gather_helper(tensor_gather, dim=-1, group=w_group)

  return tensor_gather

def _split_helper_conv(tensor, hdim=-2, wdim=-1, w_group=1, h_group=1):
  tensor_local = split_helper(tensor, dim=hdim, group=h_group)
  tensor_local = split_helper(tensor_local, dim=wdim, group=w_group)
  return tensor_local


def _gather_helper_conv(tensor, hdim=-2, wdim=-1, w_group=1, h_group=1):
  tensor_gather = gather_helper(tensor, dim=hdim, group=h_group)
  tensor_gather = gather_helper(tensor_gather, dim=wdim, group=w_group)
  return tensor_gather

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

  return w_group, h_group, e_group, world_rank, world_size

def test_distributed_fft2():
  verbose=True
  mpi_comm, device = setup_test()
  w_group, h_group, e_group, world_rank, world_size = _init_comms()

  # 256, 512, 0, 32,  8, 1e-6
  # nlat, nlon, nalt, batch_size, num_chan, tol,
  tol=1e-6
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
  dims = (-1,-2,-3)

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

def _init_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
  return

def test_distributed_ifft2():
  verbose=True
  mpi_comm, device = setup_test()
  w_group, h_group, e_group, world_rank, world_size = _init_comms()
  # 256, 512, 0, 32,  8, 1e-6
  # nlat, nlon, nalt, batch_size, num_chan, tol,
  tol=1e-6
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
  dims = (-1,-2,-3)

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

def test_distributed_spectral_conv():
  tol=1e-6
  verbose=True
  mpi_comm, device = setup_test()
  w_group, h_group, e_group, world_rank, world_size = _init_comms()
  # set up handles
  B, C, Hi, Wi, Ho, Wo = 32, 8, 256, 512, 256, 512
  print("world_rank", world_rank)
  print("world_size", world_size)

  forward_transform_local = th.RealSHT(nlat=Hi, nlon=Wi).to(device)
  inverse_transform_local = th.InverseRealSHT(nlat=Ho, nlon=Wo, lmax=forward_transform_local.lmax, mmax=forward_transform_local.mmax).to(device)
  forward_transform_dist = thd.DistributedRealSHT(nlat=Hi, nlon=Wi).to(device)
  inverse_transform_dist = thd.DistributedInverseRealSHT(nlat=Ho, nlon=Wo, lmax=forward_transform_dist.lmax, mmax=forward_transform_dist.mmax).to(device)

  _init_seed(333)
  spect_conv_local = SpectralConvS2(
            forward_transform_local,
            inverse_transform_local,
            C,
            C,
            operator_type="dhconv",
            use_tensorly=False,
            bias=True
        ).to(device)

  spect_conv_dist = SpectralConvS2(
	          forward_transform_dist,
            inverse_transform_dist,
            C,
            C,
            operator_type="dhconv",
            use_tensorly=False,
            bias=True
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
    weight = _split_helper_conv(spect_conv_local.weight, hdim=-2, wdim=None, w_group=w_group, h_group=h_group)
    print("spect_conv_local.weight",spect_conv_local.weight.shape)
    print("weight",weight.shape)
    print("spect_conv_dist.module.weight",spect_conv_dist.module.weight.shape)
    spect_conv_dist.module.weight.copy_(weight)
    spect_conv_dist.module.bias.copy_(spect_conv_local.bias)

  # input
  _init_seed(444)
  inp_full = torch.randn((B, C, Hi, Wi), dtype=torch.float32, device=device)
  # #############################################################
  # # local transform
  # #############################################################
  # # FWD pass
  inp_full.requires_grad = True
  out_full, _ = spect_conv_local(inp_full)
  # create grad for backward
  _init_seed(555)
  with torch.no_grad():
    # create full grad
    ograd_full = torch.randn_like(out_full)

  # # BWD pass
  out_full.backward(ograd_full)
  igrad_full = inp_full.grad.clone()
  wgrad_full = spect_conv_local.weight.grad.clone()
  bgrad_full = spect_conv_local.bias.grad.clone()

  #############################################################
  # distributed transform
  #############################################################
  # FWD pass
  inp_local = _split_helper_conv(inp_full, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
  print("inp_local", inp_local.shape)
  print("inp_full", inp_full.shape)
  inp_local.requires_grad = True
  out_local, _ = spect_conv_dist(inp_local)

  # BWD pass
  ograd_local = _split_helper_conv(ograd_full, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
  print("ograd_local", ograd_local.shape)
  print("ograd_full", ograd_full.shape)
  out_local, _ = spect_conv_dist(inp_local)
  out_local.backward(ograd_local)
  igrad_local = inp_local.grad.clone()
  wgrad_local = spect_conv_dist.module.weight.grad.clone()
  bgrad_local = spect_conv_dist.module.bias.grad.clone()
  mpi_comm.Barrier()
  #############################################################
  # evaluate FWD pass
  #############################################################
  with torch.no_grad():
    out_gather_full = _gather_helper_conv(out_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
    err = relative_error(out_gather_full, out_full)
    if verbose and (world_rank == 0):
      print(f"final relative error of output: {err.item()}")
    # self.assertTrue(err.item() <= tol)
  assert err.item() <= tol
  mpi_comm.Barrier()
  #############################################################
  # evaluate input grads
  #############################################################
  with torch.no_grad():
    igrad_gather_full = _gather_helper_conv(igrad_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
    err = relative_error(igrad_gather_full, igrad_full)
    if verbose and (world_rank == 0):
      print(f"final relative error of input gradients: {err.item()}")
  assert err.item() <= tol
    # self.assertTrue(err.item() <= tol)
  mpi_comm.Barrier()
  #############################################################
  # evaluate Weight grads
  #############################################################
  with torch.no_grad():
    wgrad_gather_full = _gather_helper_conv(wgrad_local, hdim=-2, wdim=None, w_group=w_group, h_group=h_group)
    print("wgrad_gather_full", wgrad_local.shape)
    print("wgrad_gather_full", wgrad_gather_full.shape)
    err = relative_error(wgrad_gather_full, wgrad_full)
    if verbose and (world_rank == 0):
      print(f"final relative error of weight gradients: {err.item()}")
    # self.assertTrue(err.item() <= tol)
  assert err.item() <= tol
  mpi_comm.Barrier()

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
  comm.cleanup()

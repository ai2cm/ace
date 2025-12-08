import os
import torch


from fme.ace.models.modulus.layers import RealFFT2, InverseRealFFT2

from fme.ace.models.makani_mpu.fft import DistributedRealFFT2, DistributedInverseRealFFT2
from fme.ace.utils import comm
import torch_harmonics as th
import torch_harmonics.distributed as thd
from test_helper import  relative_error, _split_helper, _gather_helper

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
  os.environ['GRID_H'] = '2'
  os.environ['GRID_W'] = '2'
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

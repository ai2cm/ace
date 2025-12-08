import os
import sys
import torch
from fme.core.device import get_device
from fme.core.distributed import Distributed
import torch_harmonics as th
import torch_harmonics.distributed as thd
from test_helper import split_helper_conv, gather_helper_conv, relative_error, init_seed, create_directory
from fme.ace.models.modulus.s2convolutions import SpectralConvS2
from fme.ace.models.makani_mpu.mappings import init_gradient_reduction_hooks
from fme.ace.models.makani_mpu.fft import DistributedRealFFT2, DistributedInverseRealFFT2

def setup_test():

  tol=1e-6
  # set up handles
  B, C, Hi, Wi, Ho, Wo = 32, 8, 256, 512, 256, 512

  return B, C, Hi, Wi, Ho, Wo, tol

def test_distributed_spectral_conv_without_sp():
  verbose=True
  os.environ['H_PARALLEL_SIZE'] = '1'
  os.environ['W_PARALLEL_SIZE'] = '1'
  B, C, Hi, Wi, Ho, Wo, tol = setup_test()
  device=get_device()
  forward_transform_local = th.RealSHT(nlat=Hi, nlon=Wi).to(device)
  inverse_transform_local = th.InverseRealSHT(nlat=Ho, nlon=Wo, lmax=forward_transform_local.lmax, mmax=forward_transform_local.mmax).to(device)
  init_seed(333)
  spect_conv_local = SpectralConvS2(
            forward_transform_local,
            inverse_transform_local,
            C,
            C,
            operator_type="dhconv",
            use_tensorly=False,
            bias=True
        ).to(device)

  tmp_path="testdata-scs2"
  create_directory(tmp_path)
  torch.save(spect_conv_local.weight, os.path.join(tmp_path, "weight.pt"))
  torch.save(spect_conv_local.bias, os.path.join(tmp_path, "bias.pt"))

    # input
  init_seed(444)
  inp_full = torch.randn((B, C, Hi, Wi), dtype=torch.float32, device=device)
  torch.save(inp_full, os.path.join(tmp_path, "inp_full.pt"))
  # #############################################################
  # # local transform
  # #############################################################
  # # FWD pass
  inp_full.requires_grad = True
  out_full, _ = spect_conv_local(inp_full)
  torch.save(out_full, os.path.join(tmp_path, "out_full.pt"))
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


def test_distributed_spectral_conv_with_sp():
  tmp_path="testdata-scs2"
  spect_conv_local_weight_host=torch.load(os.path.join(tmp_path, "weight.pt"))
  spect_conv_local_bias_host=torch.load(os.path.join(tmp_path, "bias.pt"))
  inp_full_host=torch.load(os.path.join(tmp_path, "inp_full.pt"))
  out_full_host=torch.load(os.path.join(tmp_path, "out_full.pt"))

  verbose=True
  os.environ['H_PARALLEL_SIZE'] = '2'
  os.environ['W_PARALLEL_SIZE'] = '1'
  device=get_device()
  spect_conv_local_weight=spect_conv_local_weight_host.to(device)
  spect_conv_local_bias=spect_conv_local_bias_host.to(device)
  inp_full=inp_full_host.to(device)
  out_full=out_full_host.to(device)

  dist=Distributed.get_instance()
  B, C, Hi, Wi, Ho, Wo, tol = setup_test()
  # forward_transform_dist = thd.DistributedRealSHT(nlat=Hi, nlon=Wi).to(device)
  # inverse_transform_dist = thd.DistributedInverseRealSHT(nlat=Ho, nlon=Wo, lmax=forward_transform_dist.lmax, mmax=forward_transform_dist.mmax).to(device)

  w_group=dist.comm_get_group("w")
  h_group=dist.comm_get_group("h")
  # print("inverse_transform_dist", inverse_transform_dist.l_shapes)
  # print("dist.comm_get_rank(h)", dist.comm_get_rank("h"))

  fft_handle = DistributedRealFFT2
  ifft_handle = DistributedInverseRealFFT2
  scale_factor=1
  hard_thresholding_fraction=1.0
  img_shape=(Hi,Wi)

  # compute the downscaled image size
  h = int(img_shape[0] // scale_factor)
  w = int(img_shape[1] // scale_factor)

  # Compute the maximum frequencies in h and in w
  modes_lat = int(h * hard_thresholding_fraction)
  modes_lon = int((w // 2 + 1) * hard_thresholding_fraction)

  padding = (0, 0)
  # effective image size:
  img_shape_eff = (
                img_shape[0] + padding[0],
                img_shape[1] + padding[1],
            )

  forward_transform_dist = fft_handle(
                *img_shape_eff, lmax=modes_lat, mmax=modes_lon
            ).float()
  inverse_transform_dist = ifft_handle(
                *img_shape_eff, lmax=modes_lat, mmax=modes_lon
            ).float()


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
    weight = split_helper_conv(spect_conv_local_weight, hdim=-2, wdim=None, w_group=w_group, h_group=h_group)
    # print("spect_conv_local.weight",spect_conv_local.weight.shape)
    # print("weight",weight.shape)
    # print("spect_conv_dist.module.weight",spect_conv_dist.module.weight.shape)
    spect_conv_dist.module.weight.copy_(weight)
    spect_conv_dist.module.bias.copy_(spect_conv_local_bias)


  #############################################################
  # distributed transform
  #############################################################
  # FWD pass
  inp_local = split_helper_conv(inp_full, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
  print("inp_local", inp_local.shape)
  print("inp_full", inp_full.shape)
  inp_local.requires_grad = True
  out_local, _ = spect_conv_dist(inp_local)

  # BWD pass
  ograd_local = split_helper_conv(ograd_full, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
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

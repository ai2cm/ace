"""This file contains unit tests of XarrayDataset."""

import os

import torch
from test_helper import gather_helper_conv, init_seed, relative_error

from fme.core.dataset.concat import get_dataset
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.distributed import Distributed
from fme.core.typing_ import Slice


# TODO: Make this run with a Bash script; I am running this test manually.
# 1. Get an interactive node in PM.
# 2. Then srun -n 4 pytest test_xarray_sp_dist.py.
def test_concat_of_XarrayConcat_w_spatial_parallel(mock_monthly_netcdfs):
    # We must use the same random seed because this code will be executed several times.
    init_seed(333)
    mock_data = mock_monthly_netcdfs

    n_timesteps = 5
    names = mock_data.var_names.all_names
    os.environ["H_PARALLEL_SIZE"] = "2"
    os.environ["W_PARALLEL_SIZE"] = "2"

    ## without domain decomposition
    with Distributed.force_non_distributed():
        config_ref = XarrayDataConfig(data_path=mock_data.tmpdir, subset=Slice(None, 4))
        ref, _ = get_dataset([config_ref], names, n_timesteps)
        niters = len(ref)
        tensor_refs = []
        for i in range(niters):
            ref_t, _, _ = ref[i]
            for var in ref_t:
                reft = ref_t[var]
                # NOTE: We need to make a hard copy
                # because the reference gets overwritten.
                tensor_refs.append(reft.clone())

    dist = Distributed.get_instance()
    w_group = dist.comm_get_group("w")
    h_group = dist.comm_get_group("h")
    # print("h_group", h_group)
    config_c1 = XarrayDataConfig(data_path=mock_data.tmpdir, subset=Slice(None, 4))
    c1, _ = get_dataset([config_c1], names, n_timesteps)

    with torch.no_grad():
        niters = len(ref)
        j = 0
        for i in range(niters):
            t1, _, _ = c1[i]
            for var in ref_t:
                reft = tensor_refs[j]
                j += 1
                c1t = t1[var]
                # NOTE: only check variables w time, lat, and lon
                if len(c1t.shape) > 3:
                    # gather_helper_conv assumes that
                    # the distribution is across the GPUs
                    c1t = c1t.to(dist.local_rank)
                    c1t_full = gather_helper_conv(
                        c1t, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
                    )
                    # Get back to the CPU so that it can be compared with the reference.
                    c1t_full_cpu = c1t_full.to("cpu")
                    err = relative_error(c1t_full_cpu, reft)
                    if dist.local_rank == 0:
                        print(var, f"final relative error of output: {err.item()}")
                        this_shape = c1t_full_cpu.shape
                        for f in range(this_shape[0]):
                            for g in range(this_shape[1]):
                                for k in range(this_shape[2]):
                                    diff = abs(c1t_full_cpu[f, g, k] - reft[f, g, k])
                                    if diff > 1e-12:
                                        print(
                                            f,
                                            g,
                                            k,
                                            " : ",
                                            c1t_full_cpu[f, g, k],
                                            reft[f, g, k],
                                        )
                    assert torch.equal(c1t_full_cpu, reft)

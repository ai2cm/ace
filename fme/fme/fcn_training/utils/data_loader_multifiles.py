# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from fme.core.device import using_gpu
from fme.core.distributed import Distributed

# import cv2
from .data_loader_fv3gfs import FV3GFSDataset
from .data_loader_params import DataLoaderParams
from .data_requirements import DataRequirements


def get_data_loader(
    params: DataLoaderParams,
    train: bool,
    requirements: DataRequirements,
):
    dist = Distributed.get_instance()
    # TODO: move this default to the DataLoaderParams init
    if params.data_type is None:
        params.data_type = "ERA5"
    if params.data_type == "ERA5":
        raise NotImplementedError("ERA5 data loader is not implemented. ")
    elif params.data_type in ["FV3GFS", "E3SMV2"]:
        dataset = FV3GFSDataset(params, requirements=requirements)
        if params.num_data_workers > 0:
            # netCDF4 __getitem__ fails with
            # "RuntimeError: Resource temporarily unavailable"
            # if num_data_workers > 0
            # TODO: move this logic to the DataLoaderParams initialization
            logging.warning(
                f"If data_type=={params.data_type}, must use num_data_workers=0. "
                "Got num_data_workers="
                f"{params.num_data_workers}, but it is being set to 0."
            )
            params.num_data_workers = 0
    else:
        raise NotImplementedError(
            f"{params.data_type} does not have an implemented data loader"
        )

    sampler = (
        DistributedSampler(dataset, shuffle=train) if dist.is_distributed() else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=dist.local_batch_size(int(params.batch_size)),
        num_workers=params.num_data_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler if train else None,
        drop_last=True,
        pin_memory=using_gpu(),
    )

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset

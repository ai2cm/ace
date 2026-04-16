# flake8: noqa
# Copied from https://github.com/ai2cm/modulus/commit/22df4a9427f5f12ff6ac891083220e7f2f54d229
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


@torch.jit.script
def compl_mul2d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex-valued multiplication operation between two 2-dimensional
    tensors.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,io->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs complex multiplication of two 2-dimensional tensors 'a' and 'b', and then
    adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(compl_mul2d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script
def compl_exp_mul2d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D complex multiplication operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,xio->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_exp_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D complex multiplication operation between two tensors 'a' and 'b',
    and then adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(compl_exp_mul2d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)

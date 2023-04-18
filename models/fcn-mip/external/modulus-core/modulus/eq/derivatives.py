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
import torch.nn as nn
from torch import Tensor
import numpy as np
import logging
import functorch
import modulus

from typing import Optional, Callable, List, Dict, Union

logger = logging.getLogger(__name__)


# TODO - This needs to be re-written to avoid Key paradigm
# class DerivWrapper(nn.Module):
#     """Base class for all neural networks using functorch functional API.
#     FuncArch perform Jacobian and Hessian calculations during the forward pass.
#
#     TODO: Get rid of Keys
#     TODO: Profile and refactor if needed
#
#     Parameters
#     ----------
#     arch : nn.Module
#         An instantiated nn.Module object
#     input_keys : List[Key]
#         A list of input keys
#     output_keys : List[Key]
#             A list of output keys
#     deriv_keys : List[Key]
#         A list of needed derivative keys
#     detach_keys : List[Key], optional
#         A list of derivative keys to be detached, by default []
#     forward_func : Optional[Callable], optional
#         If provided then it will be used as the forward function instead of the
#         default nn.Module forward` function., by default None
#     """
#
#     def __init__(
#         self,
#         arch: modulus.Module,
#         input_keys: List[Key],
#         output_keys: List[Key],
#         deriv_keys: List[Key],
#         detach_keys: List[Key] = [],
#         forward_func: Optional[Callable] = None,
#     ):
#         super().__init__()
#
#         if "torch.jit" in str(type(arch)):
#             raise RuntimeError(
#                 f"Found {type(arch)}, currently FuncArch does not work with jit."
#             )
#
#         if forward_func is None:
#             forward_func = arch.forward
#
#         self.saveable = True
#         self.input_keys = [Key(k) for k in input_keys]
#         self.output_keys = [Key(k) for k in output_keys]
#         self.deriv_keys = [Key.from_str(k) for k in deriv_keys]
#
#         self.input_key_dim = self._get_key_dim(self.input_keys)
#         self.output_key_dim = self._get_key_dim(self.output_keys)
#
#         self.input_key_dict = {str(var): var.size for var in self.input_keys}
#         self.output_key_dict = {str(var): var.size for var in self.output_keys}
#
#         self.detach_key_dict: Dict[str, int] = {
#             str(var): var.size for var in detach_keys
#         }
#
#         # If no detach keys, add a dummy for TorchScript compilation
#         if not self.detach_key_dict:
#             dummy_str = "_"
#             while dummy_str in self.input_key_dict:
#                 dummy_str += "_"
#             self.detach_key_dict[dummy_str] = 0
#
#         self.deriv_key_dict, self.max_order = self._collect_derivs(
#             self.input_key_dict, self.output_key_dict, self.deriv_keys
#         )
#         # may only need to evaluate the partial hessian or jacobian
#         needed_output_keys = set(
#             [Key(d.name) for d in self.deriv_key_dict[1] + self.deriv_key_dict[2]]
#         )
#         # keep the keys in the original order, so the mapped dims are correct
#         needed_output_keys = [
#             key for key in self.output_keys if key in needed_output_keys
#         ]
#         # needed_output_dims is used to slice I_N to save some computation
#         self.needed_output_dims = torch.tensor(
#             [self.output_key_dim[key.name] for key in needed_output_keys]
#         )
#         # if partial hessian or jacobian, the final output shape has changed and so the
#         # corresponding output key dim mapping
#         self.output_key_dim = {str(var): i for i, var in enumerate(needed_output_keys)}
#
#         in_features = sum(self.input_key_dict.values())
#         out_features = sum(self.output_key_dict.values())
#
#         if self.max_order == 0:
#             self._tensor_forward = forward_func
#         elif self.max_order == 1:
#             I_N = torch.eye(out_features)[self.needed_output_dims]
#             self.register_buffer("I_N", I_N, persistent=False)
#             self._tensor_forward = self._jacobian_impl(forward_func)
#         elif self.max_order == 2:
#             I_N1 = torch.eye(out_features)[self.needed_output_dims]
#             I_N2 = torch.eye(in_features)
#             self.register_buffer("I_N1", I_N1, persistent=False)
#             self.register_buffer("I_N2", I_N2, persistent=False)
#             self._tensor_forward = self._hessian_impl(forward_func)
#         else:
#             raise ValueError(
#                 "FuncArch currently does not support "
#                 f"{self.max_order}th order derivative"
#             )
#
#     def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
#         x = self.concat_input(
#             in_vars,
#             self.input_key_dict.keys(),
#             detach_dict=self.detach_key_dict,
#             dim=-1,
#         )
#         if self.max_order == 0:
#             pred = self._tensor_forward(x)
#             jacobian = None
#             hessian = None
#         elif self.max_order == 1:
#             pred, jacobian = self._tensor_forward(x)
#             hessian = None
#         elif self.max_order == 2:
#             pred, jacobian, hessian = self._tensor_forward(x)
#         else:
#             raise ValueError(
#                 "FuncArch currently does not support "
#                 f"{self.max_order}th order derivative"
#             )
#
#         # prepare output, jacobian and hessian
#         out = self.split_output(
#             pred,
#             self.output_key_dict,
#             dim=-1,
#         )
#         if jacobian is not None:
#             out.update(
#                 self.prepare_jacobian(
#                     jacobian,
#                     self.deriv_key_dict[1],
#                     self.input_key_dim,
#                     self.output_key_dim,
#                 )
#             )
#         if hessian is not None:
#             out.update(
#                 self.prepare_hessian(
#                     hessian,
#                     self.deriv_key_dict[2],
#                     self.input_key_dim,
#                     self.output_key_dim,
#                 )
#             )
#         return out
#
#     @staticmethod
#     def _get_key_dim(keys: List[Key]):
#         """
#         Find the corresponding dims of the keys.
#         For example: Suppose we have the following keys and corresponding size
#         {x: 2, y: 1, z: 1}, the concatenate result has dim 4, and each key map to
#         a dim {x: [0, 1], y: 2, z: 3}.
#
#         TODO Currently, the keys with more than one dim are dropped because they
#         have no use cases.
#         """
#
#         def exclusive_sum(sizes: List):
#             return np.concatenate([[0], np.cumsum(sizes)])
#
#         exclu_sum = exclusive_sum([k.size for k in keys])
#         out = {}
#         for i, k in enumerate(keys):
#             if k.size == 1:
#                 out[str(k)] = exclu_sum[i]
#         return out
#
#     @staticmethod
#     def _collect_derivs(
#         input_key_dict: Dict[str, int],
#         output_key_dict: Dict[str, int],
#         deriv_keys: List[Key],
#     ):
#         deriv_key_dict = {1: [], 2: []}
#         for x in deriv_keys:
#             # check the derivative is computable
#             assert x.name in output_key_dict, f"Cannot calculate {x}"
#             assert output_key_dict[x.name] == 1, f"key ({x.name}) size must be 1"
#             for deriv in x.derivatives:
#                 assert deriv.name in input_key_dict, f"Cannot calculate {x}"
#                 assert (
#                     input_key_dict[deriv.name] == 1
#                 ), f"key ({deriv.name}) size must be 1"
#             # collect each order derivatives
#             order = len(x.derivatives)
#             if order == 0 or order >= 3:
#                 raise ValueError(
#                     "FuncArch currently does not support " f"{order}th order derivative"
#                 )
#             else:
#                 deriv_key_dict[order].append(x)
#         max_order = 0
#         for order, keys in deriv_key_dict.items():
#             if keys:
#                 max_order = order
#         return deriv_key_dict, max_order
#
#     def _jacobian_impl(self, forward_func):
#         def jacobian_func(x, v):
#             pred, vjpfunc = functorch.vjp(forward_func, x)
#             return vjpfunc(v)[0], pred
#
#         def get_jacobian(x):
#             I_N = self.I_N
#             jacobian, pred = functorch.vmap(
#                 functorch.vmap(jacobian_func, in_dims=(None, 0)), in_dims=(0, None)
#             )(x, I_N)
#             pred = pred[:, 0, :]
#             return pred, jacobian
#
#         return get_jacobian
#
#     def _hessian_impl(self, forward_func):
#         def hessian_func(x, v1, v2):
#             def jacobian_func(x):
#                 pred, vjpfunc = functorch.vjp(forward_func, x)
#                 return vjpfunc(v1)[0], pred
#
#             # jvp and vjp
#             (jacobian, hessian, pred) = functorch.jvp(
#                 jacobian_func, (x,), (v2,), has_aux=True
#             )
#             # vjp and vjp is slow
#             # jacobian, hessianfunc, pred = functorch.vjp(jacobian_func, x, has_aux=True)
#             # hessian = hessianfunc(v2)[0]
#             return hessian, jacobian, pred
#
#         def get_hessian(x):
#             I_N1 = self.I_N1  # used to slice hessian rows
#             I_N2 = self.I_N2  # used to slice hessian columns
#             hessian, jacobian, pred = functorch.vmap(
#                 functorch.vmap(
#                     functorch.vmap(hessian_func, in_dims=(None, None, 0)),  # I_N2
#                     in_dims=(None, 0, None),  # I_N1
#                 ),
#                 in_dims=(0, None, None),  # x
#             )(x, I_N1, I_N2)
#             pred = pred[:, 0, 0, :]
#             jacobian = jacobian[:, :, 0, :]
#             return pred, jacobian, hessian
#
#         return get_hessian
#
#     @staticmethod
#     def prepare_jacobian(
#         output_tensor: Tensor,
#         deriv_keys_1st_order: List[Key],
#         input_key_dim: Dict[str, int],
#         output_key_dim: Dict[str, int],
#     ) -> Dict[str, Tensor]:
#         """Prepares for Jacobian computation
#
#         Parameters
#         ----------
#         output_tensor : Tensor
#             output tensor
#         deriv_keys_1st_order : List[Key]
#             list of 1st-order derivative keys
#         input_key_dim : Dict[str, int]
#             Number of dimensions for each input key
#         output_key_dim : Dict[str, int]
#             Number of dimensions for each output key
#
#         Returns
#         -------
#         Dict[str, Tensor]
#             Prepared dictionary for Jacobian computation
#         """
#         output = {}
#         for k in deriv_keys_1st_order:
#             input_dim = input_key_dim[k.derivatives[0].name]
#             out_dim = output_key_dim[k.name]
#             output[str(k)] = output_tensor[:, out_dim, input_dim].reshape(-1, 1)
#         return output
#
#     @staticmethod
#     def prepare_hessian(
#         output_tensor: Tensor,
#         deriv_keys_2nd_order: List[Key],
#         input_key_dim: Dict[str, int],
#         output_key_dim: Dict[str, int],
#     ) -> Dict[str, Tensor]:
#         """Prepares for Jacobian computation
#
#         Parameters
#         ----------
#         output_tensor : Tensor
#             output tensor
#         deriv_keys_2nd_order : List[Key]
#             list of 2st-order derivative keys
#         input_key_dim : Dict[str, int]
#             Number of dimensions for each input key
#         output_key_dim : Dict[str, int]
#            Number of dimensions for each output key
#
#         Returns
#         -------
#         Dict[str, Tensor]
#             Prepared dictionary for Hessian computation
#         """
#         output = {}
#         for k in deriv_keys_2nd_order:
#             input_dim0 = input_key_dim[k.derivatives[0].name]
#             input_dim1 = input_key_dim[k.derivatives[1].name]
#             out_dim = output_key_dim[k.name]
#             output[str(k)] = output_tensor[:, out_dim, input_dim0, input_dim1].reshape(
#                 -1, 1
#             )
#         return output
#
#     @staticmethod
#     def concat_input(
#         input_variables: Dict[str, Tensor],
#         mask: List[str],
#         detach_dict: Union[Dict[str, int], None] = None,
#         dim: int = -1,
#     ) -> Tensor:
#         """Concatenate dictionary elements
#
#         Parameters
#         ----------
#         input_variables : Dict[str, Tensor]
#             Input variables
#         mask : List[str]
#             List of keys that will take part in concat
#         detach_dict : Union[Dict[str, int], None], optional
#             List of keys to be excluded from concat , by default None
#         dim : int, optional
#             Concat dim, by default -1
#
#         Returns
#         -------
#         Tensor
#             Concatenated tensor of inputs
#         """
#
#         output_tensor = []
#         for key in mask:
#             if detach_dict is not None and key in detach_dict:
#                 x = input_variables[key].detach()
#             else:
#                 x = input_variables[key]
#             output_tensor += [x]
#         return torch.cat(output_tensor, dim=dim)
#
#     @staticmethod
#     def split_output(
#         output_tensor: Tensor,
#         output_dict: Dict[str, int],
#         dim: int = -1,
#     ) -> Dict[str, Tensor]:
#         """Splits the output into a dictionary of tensors
#
#         Parameters
#         ----------
#         output_tensor : Tensor
#             The output tensor
#         output_dict : Dict[str, int]
#            The output dictionary
#         dim : int, optional
#             split dim
#
#         Returns
#         -------
#         Dict[str, Tensor]
#             Dictionary of outputs
#         """
#         output = {}
#         for k, v in zip(
#             output_dict,
#             torch.split(output_tensor, list(output_dict.values()), dim=dim),
#         ):
#             output[k] = v
#         return output
#
#
# """ Key
# """
#
# from typing import Union, List
# from functools import reduce
#
# diff_str = "__"
# NO_OP_SCALE = (0, 1)
#
#
# class Key(object):
#     """
#     Class describing keys used for graph unroll.
#     The most basic key is just a simple string
#     however you can also add dimension information
#     and even information on how to scale inputs
#     to networks.
#
#     Parameters
#     ----------
#     name : str
#       String used to refer to the variable (e.g. 'x', 'y'...).
#     size : int=1
#       Dimension of variable.
#     derivatives : List=[]
#       This signifies that this key holds a derivative with
#       respect to that key.
#     scale: (float, float)
#       Characteristic location and scale of quantity: used for normalisation.
#     """
#
#     def __init__(self, name, size=1, derivatives=[], base_unit=None, scale=NO_OP_SCALE):
#         super(Key, self).__init__()
#         self.name = name
#         self.size = size
#         self.derivatives = derivatives
#         self.base_unit = base_unit
#         self.scale = scale
#
#     @classmethod
#     def from_str(cls, name):
#         """Creates key from string"""
#         split_name = name.split(diff_str)
#         var_name = split_name[0]
#         diff_names = Key.convert_list(split_name[1:])
#         return cls(var_name, size=1, derivatives=diff_names)
#
#     @classmethod
#     def from_tuple(cls, name_size):
#         """Creates keys from tuple"""
#         split_name = name_size[0].split(diff_str)
#         var_name = split_name[0]
#         diff_names = Key.convert_list(split_name[1:])
#         return cls(var_name, size=name_size[1], derivatives=diff_names)
#
#     @classmethod
#     def convert(cls, name_or_tuple):
#         """Convert string or tuple to key"""
#         if isinstance(name_or_tuple, str):
#             key = Key.from_str(name_or_tuple)
#         elif isinstance(name_or_tuple, tuple):
#             key = cls.from_tuple(name_or_tuple)
#         elif isinstance(name_or_tuple, cls):
#             key = name_or_tuple
#         else:
#             raise ValueError("can only convert string or tuple to key")
#         return key
#
#     @staticmethod
#     def convert_list(ls):
#         """Converts list into set of keys"""
#         keys = []
#         for name_or_tuple in ls:
#             keys.append(Key.convert(name_or_tuple))
#         return keys
#
#     def __str__(self):
#         diff_str = "".join(["__" + x.name for x in self.derivatives])
#         return self.name + diff_str
#
#     def __repr__(self):
#         return str(self)
#
#     def __eq__(self, obj):
#         return isinstance(obj, Key) and str(self) == str(obj)
#
#     def __lt__(self, obj):
#         assert isinstance(obj, Key)
#         return str(self) < str(obj)
#
#     def __gt__(self, obj):
#         assert isinstance(obj, Key)
#         return str(self) > str(obj)
#
#     def __hash__(self):
#         return hash(str(self))
#

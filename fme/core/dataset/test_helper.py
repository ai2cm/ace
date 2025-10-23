import os

import torch
import torch.distributed as dist

# this computes a relative error compatible with torch.allclose or np.allclose
def relative_error(tensor1, tensor2):
    return torch.sum(torch.abs(tensor1-tensor2)) / torch.sum(torch.abs(tensor2))

# this computes an absolute error compatible with torch.allclose or np.allclose
def absolute_error(tensor1, tensor2):
    return torch.max(torch.abs(tensor1-tensor2))

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

def gather_helper_conv(tensor, hdim=-2, wdim=-1, w_group=1, h_group=1):
  tensor_gather = gather_helper(tensor, dim=hdim, group=h_group)
  tensor_gather = gather_helper(tensor_gather, dim=wdim, group=w_group)
  return tensor_gather

def init_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
  return

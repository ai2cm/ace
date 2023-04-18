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
import dgl.function as fn

from dgl import DGLGraph
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def checkpoint_identity(layer, *args, **kwargs):
    return layer(*args)


def set_checkpoint_fn(do_checkpointing: bool):
    if do_checkpointing:
        return checkpoint
    else:
        return checkpoint_identity


def concat_message_function(edges) -> Tensor:
    # concats src node , dst node, and edge features
    cat_feat = torch.cat((edges.src["x"], edges.dst["x"], edges.data["x"]), dim=1)
    return {"cat_feat": cat_feat}


def concat_efeat_dgl_mesh(efeat: Tensor, nfeat: Tensor, graph: DGLGraph) -> Tensor:
    with graph.local_scope():
        graph.ndata["x"] = nfeat
        graph.edata["x"] = efeat

        graph.apply_edges(concat_message_function)
        return graph.edata["cat_feat"]


def concat_efeat_dgl_m2g_g2m(
    efeat: Tensor, src_feat: Tensor, dst_feat: Tensor, graph: DGLGraph
) -> Tensor:
    with graph.local_scope():
        graph.srcdata["x"] = src_feat
        graph.dstdata["x"] = dst_feat
        graph.edata["x"] = efeat

        graph.apply_edges(concat_message_function)
        return graph.edata["cat_feat"]


@torch.jit.script
def sum_efeat_dgl(
    efeat: Tensor, src_feat: Tensor, dst_feat: Tensor, src_idx: Tensor, dst_idx: Tensor
) -> Tensor:
    return efeat + src_feat[src_idx] + dst_feat[dst_idx]


def agg_concat_dgl(
    efeat: Tensor, nfeat: Tensor, graph: DGLGraph, aggregation: str
) -> Tensor:
    with graph.local_scope():
        # populate features on graph edges
        graph.edata["x"] = efeat

        # aggregate edge features
        if aggregation == "sum":
            graph.update_all(fn.copy_e("x", "m"), fn.sum("m", "h_dest"))
        elif aggregation == "mean":
            graph.update_all(fn.copy_e("x", "m"), fn.mean("m", "h_dest"))
        else:
            raise RuntimeError("Not a valid aggregation!")

        # concat node & edge features
        cat_feat = torch.cat((graph.dstdata["h_dest"], nfeat), -1)
        return cat_feat

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
from dgl import DGLGraph
from .mlp import MLP
from .utils import agg_concat_dgl

try:
    from pylibcugraphops.torch.autograd import agg_concat_e2n
    from pylibcugraphops.typing import FgCsr
except ImportError:
    FgCsr = None
    agg_concat_e2n = None


class NodeBlockDGL(nn.Module):
    def __init__(
        self,
        graph: DGLGraph,
        aggregation: str = "sum",
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.graph = graph
        self.aggregation = aggregation

        self.node_MLP = MLP(
            input_dim=input_dim_nodes + input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
    ) -> Tensor:
        cat_feat = agg_concat_dgl(efeat, nfeat, self.graph, self.aggregation)
        # update node features + residual connection
        nfeat_new = self.node_MLP(cat_feat) + nfeat

        return efeat, nfeat_new

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        self.graph = self.graph.to(device=device, dtype=dtype)
        self = super().to(*args, **kwargs)
        return self


class NodeBlockCUGO(nn.Module):
    def __init__(
        self,
        graph: FgCsr,
        aggregation: str = "sum",
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.graph = graph
        self.aggregation = aggregation

        self.node_MLP = MLP(
            input_dim=input_dim_nodes + input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
    ) -> Tensor:
        # aggregate edge features and concat node features
        cat_feat = agg_concat_e2n(efeat, nfeat, self.graph, self.aggregation)
        # update node features + residual connection
        nfeat_new = self.node_MLP(cat_feat) + nfeat
        return efeat, nfeat_new

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        self.graph = self.graph.to(device=device, dtype=dtype)
        self = super().to(*args, **kwargs)
        return self

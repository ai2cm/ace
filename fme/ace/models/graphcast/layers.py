import torch
from torch import nn

# This has been coded based on the Supplmentary Information
# in the GraphCast paper by Lam et al., 2023
# https://doi.org/10.1126/science.adi2336

# ---------- multi-layer perceptron ----------


def mlp(
    sizes: list[int], act: nn.Module = nn.SiLU, norm: bool = True, bias: bool = True
) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1], bias=bias)]
        if i < len(sizes) - 2:
            layers += [act()]
            if norm:
                layers += [nn.LayerNorm(sizes[i + 1])]
    return nn.Sequential(*layers)


# ---------- Encoder with embedders ----------


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        grid_node_structural_dim: int,
        mesh_node_structural_dim: int,
        edge_features: int,
        use_layernorm: bool = True,
        act: nn.Module = nn.SiLU,
        bias: bool = True,
    ):
        super().__init__()

        # Create 5 MLP embedders (Eq 6 of GraphCast SI)
        # 1. Grid node embedder: input data + structural features
        self.grid_node_embed = mlp(
            [
                input_channels + grid_node_structural_dim,
                output_channels,
                output_channels,
            ],
            norm=use_layernorm,
            act=act,
            bias=bias,
        )
        # 2. Mesh node embedder: just structural features
        self.mesh_node_embed = mlp(
            [mesh_node_structural_dim, output_channels, output_channels],
            norm=use_layernorm,
            act=act,
            bias=bias,
        )
        # 3. Mesh edge embedder M2M structural edge features
        self.M2M_edge_embed = mlp(
            [edge_features, output_channels, output_channels],
            norm=use_layernorm,
            act=act,
            bias=bias,
        )
        # 4. G2M edge embedder: G2M structural edge features
        self.G2M_edge_embed = mlp(
            [edge_features, output_channels, output_channels],
            norm=use_layernorm,
            act=act,
            bias=bias,
        )
        # 5. M2G edge embedder: M2G structural edge features
        self.M2G_edge_embed = mlp(
            [edge_features, output_channels, output_channels],
            norm=use_layernorm,
            act=act,
            bias=bias,
        )

        # Message function: embedded grid and mesh nodes + embedded edge (Eq 7 of SI)
        self.G2M_message = mlp(
            [output_channels * 3, output_channels, output_channels],
            norm=use_layernorm,
            act=act,
            bias=bias,
        )

        # Mesh node update: embedded mesh node + aggregated messages  (Eq 8 of SI)
        self.G2M_node_update = mlp(
            [output_channels * 2, output_channels, output_channels],
            norm=use_layernorm,
            act=act,
            bias=bias,
        )

        # Grid node update: receives self.grid_node_embed (Eq 9 of SI)
        self.G_update = mlp(
            [output_channels, output_channels, output_channels],
            norm=use_layernorm,
            act=act,
            bias=bias,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        grid_structural: torch.Tensor,
        mesh_structural: torch.Tensor,
        M2M_edge_structural: torch.Tensor,
        G2M_edge_structural: torch.Tensor,
        M2G_edge_structural: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        residual: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = inputs.shape[0]
        Nm = mesh_structural.shape[0]
        E = G2M_edge_structural.shape[0]

        # Embeddings (Eq 6 of GraphCast SI)
        grid_combined = torch.cat([inputs, grid_structural.expand(B, -1, -1)], dim=-1)
        vG = self.grid_node_embed(grid_combined)
        vM = self.mesh_node_embed(mesh_structural.expand(B, -1, -1))
        eG2M = self.G2M_edge_embed(G2M_edge_structural.expand(B, -1, -1))

        # Update edges (Eq 7 of GraphCast SI)
        edge_in = torch.cat([eG2M, vG[:, senders], vM[:, receivers]], dim=-1)
        if residual:
            eG2M = eG2M + self.G2M_message(edge_in)  # Eq 10 of SI
        else:
            eG2M = self.G2M_message(edge_in)

        # Aggregate messages
        H = eG2M.size(-1)
        agg = torch.zeros(B, Nm, H, dtype=eG2M.dtype, device=eG2M.device)
        agg.scatter_add_(dim=1, index=receivers.view(1, E, 1).expand(B, E, H), src=eG2M)

        # Update mesh and grid nodes (Eqs 8 & 9 of GraphCast SI)
        vM_in = torch.cat([vM, agg], dim=-1)
        if residual:
            vM = vM + self.G2M_node_update(vM_in)  # Eq 10 of SI
            vG = vG + self.G_update(vG)  # Eq 10 of SI
        else:
            vM = self.G2M_node_update(vM_in)
            vG = self.G_update(vG)

        # Embeddings used later in the processor and decoder
        eM2M = self.M2M_edge_embed(M2M_edge_structural)
        eM2G = self.M2G_edge_embed(M2G_edge_structural)
        return vM, vG, eM2M, eM2G


# ---------- Processor ----------


class InteractionNetwork(nn.Module):
    def __init__(
        self,
        output_channels: int = 512,
        use_layernorm: bool = True,
        act: nn.Module = nn.SiLU,
        bias: bool = True,
    ):
        super().__init__()

        # Message function: embedded mesh edge (Eq 11 of SI)
        self.M2M_message = mlp(
            [output_channels * 3, output_channels, output_channels],
            norm=use_layernorm,
            act=act,
            bias=bias,
        )

        # Mesh node update: embedded mesh node + messages (Eq 12 of SI)
        self.M2M_node_update = mlp(
            [output_channels * 2, output_channels, output_channels],
            norm=use_layernorm,
            act=act,
            bias=bias,
        )

    def forward(
        self,
        vM: torch.Tensor,
        eM: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        residual: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Nm, H = vM.shape
        eM = eM.expand(B, -1, -1)

        eM_in = torch.cat([eM, vM[:, senders], vM[:, receivers]], dim=-1)
        if residual:
            eM = eM + self.M2M_message(eM_in)  # (Eq 13 of SI)
        else:
            eM = self.M2M_message(eM_in)

        E = eM.shape[1]
        agg = torch.zeros((B, Nm, H), device=vM.device, dtype=vM.dtype)
        agg.scatter_add_(dim=1, index=receivers.view(1, E, 1).expand(B, E, H), src=eM)

        # Update nodes
        vM_in = torch.cat([vM, agg], dim=-1)
        if residual:
            vM = vM + self.M2M_node_update(vM_in)  # (Eq 13 of SI)
        else:
            vM = self.M2M_node_update(vM_in)
        return vM, eM


class Processor(nn.Module):
    def __init__(
        self,
        num_layers: int = 16,
        output_channels: int = 512,
        use_layernorm: bool = True,
        act: nn.Module = nn.SiLU,
        bias: bool = True,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                InteractionNetwork(
                    output_channels=output_channels,
                    use_layernorm=use_layernorm,
                    act=act,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        vM: torch.Tensor,
        eM: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        residual: bool = True,
    ) -> torch.Tensor:
        for layer in self.layers:
            vM, eM = layer(vM, eM, senders, receivers, residual=residual)
        return vM


# ---------- Decoder ----------


class Decoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        use_layernorm: bool = True,
        act: nn.Module = nn.SiLU,
        bias: bool = True,
    ):
        super().__init__()

        # Effectively the reverse of G2M_message (Eq 14 of SI)
        self.edge_M2G = mlp(
            [input_channels * 3, input_channels, input_channels],
            act=act,
            norm=use_layernorm,
            bias=bias,
        )
        # Grid node update (Eq 15 of SI)
        self.node_VG = mlp(
            [input_channels * 2, input_channels, input_channels],
            act=act,
            norm=use_layernorm,
            bias=bias,
        )
        # Final output to physical variables (Eq 17 of SI)
        self.out_head = mlp(
            [input_channels, input_channels, output_channels],
            act=act,
            norm=False,
            bias=bias,
        )

    def forward(
        self,
        vM: torch.Tensor,
        vG: torch.Tensor,
        eM2G: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        residual: bool = True,
    ) -> torch.Tensor:
        B, Nm, H = vM.shape
        Ng = vG.shape[1]
        E = eM2G.shape[0]

        # M2G edge update (Eq 14)
        e_in = torch.cat(
            [eM2G.expand(B, -1, -1), vM[:, senders], vG[:, receivers]], dim=-1
        )
        if residual:
            eM2G = eM2G + self.edge_M2G(e_in)
        else:
            eM2G = self.edge_M2G(e_in)

        # Aggregate to grid receivers
        agg = torch.zeros((B, Ng, H), device=vG.device, dtype=vG.dtype)
        agg.scatter_add_(dim=1, index=receivers.view(1, E, 1).expand(B, E, H), src=eM2G)

        # Grid node update (Eq 15)
        vG_in = torch.cat([vG, agg], dim=-1)
        if residual:
            vG = vG + self.node_VG(vG_in)
        else:
            vG = self.node_VG(vG_in)

        return self.out_head(vG)

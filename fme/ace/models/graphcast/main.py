import numpy as np
import torch
import torch.nn as nn

from fme.ace.models.graphcast.layers import Decoder, Encoder, Processor
from fme.ace.models.graphcast.utils import (
    faces_to_edges,
    get_bipartite_graph_spatial_features,
    get_graph_spatial_features,
    get_hierarchy_of_triangular_meshes_for_sphere,
    get_max_edge_distance,
    get_mesh_positions,
    in_mesh_triangle_indices,
    masked_mesh_faces,
    merge_meshes,
    radius_query_indices,
)
from fme.core.dataset_info import DatasetInfo


class GraphCast(torch.nn.Module):
    """
    This is a streamlined version of GraphCast, made compatible for the
    ACE codebase. The only parts which have been directly copied from the
    DeepMind repo are in utils.py, which define the graphs for the data
    and mesh grids. The rest has been more or less coded from scratch.

    Some terminology:
    G2M = Grid-to-Mesh (Encoder)
    M2M = Mesh-to-Mesh (Processor)
    M2G = Mesh-to-Grid (Decoder)

    TODO: GraphCast embeds spatial positions into the grid and mesh nodes, as
    well as their relative positions along the node edges. These are currently
    hard-coded in the functions get_bipartite_graph_spatial_features and
    get_graph_spatial_features (see below). These then set the
    grid_node_structural_dim, mesh_node_structural_dim, and edge_features in
    the Encoder. The value of 3 for the node_structural_dim corresponds to:
    add_node_latitude (1D; cosine of lat), and add_node_longitude (2D; sine and
    cosine of lon). The edge_features is then 4 with add_relative_positions (2D;
    delta lat and delta lon), relative_longitude_local_coordinates (1D), and
    relative_latitude_local_coordinates (1D). In the future, we may want to add
    these as options in the config file.

    Parameters
    ----------
        input_channels (int).   : Number of input features of the model
        output_channels (int)   : Number of output features of the model
        dataset_info (DatasetInfo): Data info (img_shape, coordinates etc)
        latent_dimension (int)  : Number of features output by each MLP
        activation (str)        : Activation function to use in all MLPs
        meshes (int)            : Number of mesh levels (splits of an icosahedron)
        M0 (int)                : Starting mesh (0 includes coarsest. Must be <= meshes)
        bias (bool)             : Whether to use bias term in MLPs
        radius_fraction (float) : Fraction of max edge length of finest mesh
        layernorm (bool)        : Whether to use layernorm in MLPs
        processor_steps (int)   : Number of MLP layers in the processor
        residual (bool)         : Whether to output residual connections
        is_ocean (bool)         : Whether to mask land points from the data and mesh.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dataset_info: DatasetInfo,
        latent_dimension: int = 512,
        activation: str = "SiLU",
        meshes: int = 6,
        M0: int = 0,
        bias: bool = True,
        radius_fraction: float = 0.6,
        layernorm: bool = True,
        processor_steps: int = 16,
        residual: bool = True,
        is_ocean: bool = False,
    ):
        super().__init__()

        self.dataset_info = dataset_info
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.latent_dimension = latent_dimension
        self.activation = getattr(nn, activation)
        self.bias = bias
        self.processor_steps = processor_steps
        self.residual = residual
        self.is_ocean = is_ocean

        # define the multi-mesh
        if M0 > meshes:
            M0 = meshes
        if M0 < 0:
            M0 = 0
        self.meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=meshes)[M0:]
        # radius for G2M connections
        self.radius = (
            get_max_edge_distance(self.meshes[-1].faces, self.meshes[-1].vertices)
            * radius_fraction
        )
        self.G2M_built = False
        self.M2M_built = False
        self.M2G_built = False
        self.lon = None
        self.lat = None
        self.mask = None

        self.encoder = Encoder(
            input_channels=self.input_channels,
            output_channels=self.latent_dimension,
            grid_node_structural_dim=3,
            mesh_node_structural_dim=3,
            edge_features=4,
            use_layernorm=layernorm,
            act=self.activation,
            bias=self.bias,
        )

        self.processor = Processor(
            num_layers=self.processor_steps,
            output_channels=self.latent_dimension,
            use_layernorm=layernorm,
            act=self.activation,
            bias=self.bias,
        )

        self.decoder = Decoder(
            input_channels=self.latent_dimension,
            output_channels=self.output_channels,
            use_layernorm=layernorm,
            act=self.activation,
            bias=self.bias,
        )

    def get_coordinates_and_mask(self):
        if any(x is None for x in (self.lat, self.lon, self.mask)):
            if self.is_ocean:
                mask_provider = self.dataset_info.mask_provider
                mask = mask_provider.get_mask_tensor_for("mask_2d")
                if mask is not None:
                    self.mask = mask.cpu().numpy().astype(bool)
                else:
                    raise RuntimeError("Could not get mask tensor")
            else:
                self.mask = np.ones(self.dataset_info.img_shape, dtype=bool)

            lat, lon = self.dataset_info.horizontal_coordinates.meshgrid
            self.lat = lat.cpu().numpy()
            self.lon = lon.cpu().numpy()

    def init_G2M_graph(self, lat: np.ndarray, lon: np.ndarray, mask: np.ndarray):
        if self.G2M_built:
            return
        else:
            query = radius_query_indices(
                grid_latitude=lat,
                grid_longitude=lon,
                mesh=self.meshes[-1],
                radius=self.radius,
                mask=mask,
                ocean_mesh=self.is_ocean,
                return_positions=True,
            )
            grid_indices, mesh_indices, grid_pos, mesh_pos = query

            ## Grid node coordinates for encoder
            self.grid_nodes_lon = grid_pos[:, 0].astype(np.float32)
            self.grid_nodes_lat = grid_pos[:, 1].astype(np.float32)

            # Mesh node coordinates for encoder
            self.mesh_nodes_lon = mesh_pos[:, 0].astype(np.float32)
            self.mesh_nodes_lat = mesh_pos[:, 1].astype(np.float32)

            # G2M edges and features
            (grid_node_features, mesh_node_features, edge_features) = (
                get_bipartite_graph_spatial_features(
                    senders_node_lat=self.grid_nodes_lat,
                    senders_node_lon=self.grid_nodes_lon,
                    receivers_node_lat=self.mesh_nodes_lat,
                    receivers_node_lon=self.mesh_nodes_lon,
                    senders=grid_indices,
                    receivers=mesh_indices,
                    edge_normalization_factor=None,
                    add_node_positions=False,
                    add_node_latitude=True,
                    add_node_longitude=True,
                    add_relative_positions=True,
                    relative_longitude_local_coordinates=True,
                    relative_latitude_local_coordinates=True,
                )
            )

            self.G2M_grid_features = torch.from_numpy(grid_node_features).float()
            self.G2M_mesh_features = torch.from_numpy(mesh_node_features).float()
            self.G2M_edge_features = torch.from_numpy(edge_features).float()
            self.G2M_senders = torch.from_numpy(grid_indices)
            self.G2M_receivers = torch.from_numpy(mesh_indices)
            self.G2M_built = True

    def init_M2M_graph(self, lat: np.ndarray, lon: np.ndarray, mask: np.ndarray):
        if self.M2M_built:
            return
        else:
            merged_mesh = merge_meshes(self.meshes)
            mesh_mask = get_mesh_positions(
                merged_mesh.vertices,
                grid_latitude=lat,
                grid_longitude=lon,
                mask=mask,
                return_pos_ids=True,
                ocean_mesh=self.is_ocean,
            )[1]

            merged_mesh_faces = masked_mesh_faces(
                merged_mesh.faces, np.where(mesh_mask)[0]
            )
            mesh_senders, mesh_receivers = faces_to_edges(merged_mesh_faces)

            # M2M edges and features
            node_features, edge_features = get_graph_spatial_features(
                node_lat=self.mesh_nodes_lat,
                node_lon=self.mesh_nodes_lon,
                senders=mesh_senders,
                receivers=mesh_receivers,
                edge_normalization_factor=None,
                add_node_positions=False,
                add_node_latitude=True,
                add_node_longitude=True,
                add_relative_positions=True,
                relative_longitude_local_coordinates=True,
                relative_latitude_local_coordinates=True,
            )

            self.M2M_mesh_features = torch.from_numpy(node_features).float()
            self.M2M_edge_features = torch.from_numpy(edge_features).float()
            self.M2M_senders = torch.from_numpy(mesh_senders)
            self.M2M_receivers = torch.from_numpy(mesh_receivers)
            self.M2M_built = True

    def init_M2G_graph(self, lat: np.ndarray, lon: np.ndarray, mask: np.ndarray):
        if self.M2G_built:
            return
        else:
            # get the 3 vertices of the mesh triangle in which each grid point falls
            query = in_mesh_triangle_indices(
                grid_latitude=lat,
                grid_longitude=lon,
                mesh=self.meshes[-1],
                mask=mask,
                ocean_mesh=self.is_ocean,
                return_positions=True,
            )
            grid_indices, mesh_indices, grid_pos, mesh_pos = query

            # M2G edges and features
            (grid_node_features, mesh_node_features, edge_features) = (
                get_bipartite_graph_spatial_features(
                    senders_node_lat=mesh_pos[:, 1].astype(np.float32),
                    senders_node_lon=mesh_pos[:, 0].astype(np.float32),
                    receivers_node_lat=grid_pos[:, 1].astype(np.float32),
                    receivers_node_lon=grid_pos[:, 0].astype(np.float32),
                    senders=mesh_indices,
                    receivers=grid_indices,
                    edge_normalization_factor=None,
                    add_node_positions=False,
                    add_node_latitude=True,
                    add_node_longitude=True,
                    add_relative_positions=True,
                    relative_longitude_local_coordinates=True,
                    relative_latitude_local_coordinates=True,
                )
            )

            self.M2G_grid_features = torch.from_numpy(grid_node_features).float()
            self.M2G_mesh_features = torch.from_numpy(mesh_node_features).float()
            self.M2G_edge_features = torch.from_numpy(edge_features).float()
            self.M2G_senders = torch.from_numpy(mesh_indices)
            self.M2G_receivers = torch.from_numpy(grid_indices)
            self.M2G_built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.get_coordinates_and_mask()
        self.init_G2M_graph(self.lat, self.lon, self.mask)
        self.init_M2M_graph(self.lat, self.lon, self.mask)
        self.init_M2G_graph(self.lat, self.lon, self.mask)
        device = x.device
        dtype = x.dtype

        # Prep inputs
        mask = torch.from_numpy(self.mask).to(device)
        H, W = mask.shape
        x = x[:, :, mask].permute(0, 2, 1).contiguous()

        # Return inputs mapped to latent space on mesh nodes
        vM, vG, eM2M, eM2G = self.encoder(
            inputs=x,
            grid_structural=self.G2M_grid_features.to(device),
            mesh_structural=self.G2M_mesh_features.to(device),
            M2M_edge_structural=self.M2M_edge_features.to(device),
            G2M_edge_structural=self.G2M_edge_features.to(device),
            M2G_edge_structural=self.M2G_edge_features.to(device),
            senders=self.G2M_senders.to(device),
            receivers=self.G2M_receivers.to(device),
            residual=self.residual,
        )

        # Return updated mesh nodes after M2M message passing
        vM = self.processor(
            vM=vM,
            eM=eM2M,
            senders=self.M2M_senders.to(device),
            receivers=self.M2M_receivers.to(device),
            residual=self.residual,
        )

        # Return output predictions at grid nodes
        y = self.decoder(
            vM=vM,
            vG=vG,
            eM2G=eM2G,
            senders=self.M2G_senders.to(device),
            receivers=self.M2G_receivers.to(device),
            residual=self.residual,
        )

        # Prep outputs
        B, N, C = y.shape
        out = torch.zeros((B, C, H, W), dtype=dtype, device=device)
        out[:, :, mask] = y.permute(0, 2, 1).contiguous()
        return out

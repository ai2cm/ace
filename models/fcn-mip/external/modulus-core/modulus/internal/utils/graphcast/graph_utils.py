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

import dgl
from dgl import DGLGraph
import torch
from torch import Tensor, testing
import numpy as np
from torch.nn import functional as F


def create_graph(src, dst, to_bidirected=True, add_self_loop=False, dtype=torch.int32):
    """
    creates a DGL graph from an adj matrix in COO format.
    """
    graph = dgl.graph((src, dst), idtype=dtype)
    if to_bidirected:
        graph = dgl.to_bidirected(graph)
    if add_self_loop:
        graph = dgl.add_self_loop(graph)
    return graph


def create_heterograph(src, dst, labels, dtype=torch.int32):
    """
    creates a heterogeneous DGL graph from an adj matrix in COO format.
    """
    graph = dgl.heterograph({labels: ("coo", (src, dst))}, idtype=dtype)
    return graph


def add_edge_features(graph: DGLGraph, pos: Tensor) -> DGLGraph:
    """
    adds relative displacement & displacement norm as edge features
    """
    if isinstance(pos, tuple):
        src_pos, dst_pos = pos
    else:
        src_pos = dst_pos = pos
    src, dst = graph.edges()

    src_pos, dst_pos = src_pos[src.long()], dst_pos[dst.long()]
    dst_latlon = xyz2latlon(dst_pos, unit="rad")
    dst_lat, dst_lon = dst_latlon[:, 0], dst_latlon[:, 1]

    # azimuthal & polar rotation
    theta_azimuthal = azimuthal_angle(dst_lon)
    theta_polar = polar_angle(dst_lat)

    src_pos = geospatial_rotation(src_pos, theta=theta_azimuthal, axis="z", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_azimuthal, axis="z", unit="rad")
    # y values should be zero
    try:
        testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
    except:
        raise ValueError("Invalid projection of edge nodes to local ccordinate system")
    src_pos = geospatial_rotation(src_pos, theta=theta_polar, axis="y", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_polar, axis="y", unit="rad")
    # x values should be one, y & z values should be zero
    try:
        testing.assert_close(dst_pos[:, 0], torch.ones_like(dst_pos[:, 0]))
        testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
        testing.assert_close(dst_pos[:, 2], torch.zeros_like(dst_pos[:, 2]))
    except:
        raise ValueError("Invalid projection of edge nodes to local ccordinate system")

    # prepare edge features
    disp = src_pos - dst_pos
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)

    # normalize using the longest edge
    max_disp_norm = torch.max(disp_norm)
    graph.edata["x"] = torch.cat(
        (disp / max_disp_norm, disp_norm / max_disp_norm), dim=-1
    )
    return graph


def add_node_features(graph: DGLGraph, pos: Tensor) -> DGLGraph:
    """
    assigns cos(lat), sin(lon), cos(lon) features to the nodes
    """
    latlon = xyz2latlon(pos)
    lat, lon = latlon[:, 0], latlon[:, 1]
    graph.ndata["x"] = torch.stack(
        (torch.cos(lat), torch.sin(lon), torch.cos(lon)), dim=-1
    )
    return graph


def latlon2xyz(latlon: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    """
    Converts latlon in degrees to xyz
    Based on: https://stackoverflow.com/questions/1185408
    - The x-axis goes through long,lat (0,0);
    - The y-axis goes through (0,90);
    - The z-axis goes through the poles.
    """
    if unit == "deg":
        latlon = deg2rad(latlon)
    elif unit == "rad":
        pass
    else:
        raise ValueError("Not a valid unit")
    lat, lon = latlon[:, 0], latlon[:, 1]
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack((x, y, z), dim=1)


def xyz2latlon(xyz: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    """
    Converts xyz to latlon in degrees
    Based on: https://stackoverflow.com/questions/1185408
    - The x-axis goes through long,lat (0,0);
    - The y-axis goes through (0,90);
    - The z-axis goes through the poles.
    """
    lat = torch.arcsin(xyz[:, 2] / radius)
    lon = torch.arctan2(xyz[:, 1], xyz[:, 0])
    if unit == "deg":
        return torch.stack((rad2deg(lat), rad2deg(lon)), dim=1)
    elif unit == "rad":
        return torch.stack((lat, lon), dim=1)
    else:
        raise ValueError("Not a valid unit")


def geospatial_rotation(
    invar: Tensor, theta: Tensor, axis: str, unit: str = "rad"
) -> Tensor:
    """Rotation using right hand rule"""

    # get the right unit
    if unit == "deg":
        invar = rad2deg(invar)
    elif unit == "rad":
        pass
    else:
        raise ValueError("Not a valid unit")

    invar = torch.unsqueeze(invar, -1)
    rotation = torch.zeros((theta.size(0), 3, 3))
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    if axis == "x":
        rotation[:, 0, 0] += 1.0
        rotation[:, 1, 1] += cos
        rotation[:, 1, 2] -= sin
        rotation[:, 2, 1] += sin
        rotation[:, 2, 2] += cos
    elif axis == "y":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 2] += sin
        rotation[:, 1, 1] += 1.0
        rotation[:, 2, 0] -= sin
        rotation[:, 2, 2] += cos
    elif axis == "z":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 1] -= sin
        rotation[:, 1, 0] += sin
        rotation[:, 1, 1] += cos
        rotation[:, 2, 2] += 1.0
    else:
        raise ValueError("Invalid axis")

    outvar = torch.matmul(rotation, invar)
    outvar = outvar.squeeze()
    return outvar


def azimuthal_angle(lon: Tensor) -> Tensor:
    angle = torch.where(lon >= 0.0, 2 * np.pi - lon, -lon)
    return angle


def polar_angle(lat: Tensor) -> Tensor:
    angle = torch.where(lat >= 0.0, lat, 2 * np.pi + lat)
    return angle


def deg2rad(deg):
    return deg * np.pi / 180


def rad2deg(rad):
    return rad * 180 / np.pi


def get_edge_len(edge_src, edge_dst, axis=1):
    return np.linalg.norm(edge_src - edge_dst, axis=axis)


def cell_to_adj(cells):
    """creates adjancy matrix in COO format from mesh cells"""
    num_cells = np.shape(cells)[0]
    src = [cells[i][indx] for i in range(num_cells) for indx in [0, 1, 2]]
    dst = [cells[i][indx] for i in range(num_cells) for indx in [1, 2, 0]]
    return src, dst


# NOTE debug code
# latlon = torch.tensor([[0,0]])
# xyz = latlon2xyz(latlon)
# print(xyz)
# latlon = xyz2latlon(xyz)
# print(latlon)

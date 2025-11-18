# flake8: noqa
# Copied from tag v0.1.1 https://github.com/google-deepmind/graphcast/tree/5e63ba030f6438d0a97521b30360b7c95d28796a
# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from scipy.spatial import cKDTree, transform
from scipy.interpolate import griddata

from fme.ace.models.graphcast import GRAPHCAST_AVAIL


NumpyInterface = Any
TransformInterface = Any

# ----------------------------- Data structures -----------------------------


class TriangularMesh(NamedTuple):
    """Triangular mesh on the unit sphere."""

    vertices: np.ndarray  # [num_vertices, 3], unit-norm (x,y,z)
    faces: np.ndarray  # [num_faces, 3], CCW indices into vertices


# ----------------------------- Mesh construction ---------------------------


def get_icosahedron() -> TriangularMesh:
    """
    Regular icosahedron with circumscribed unit sphere.

    Returns:
      vertices: (12,3) float32, unit-norm Cartesian coords
      faces:    (20,3) int32, CCW indices
    """
    phi = (1 + np.sqrt(5.0)) / 2.0
    verts = []
    for c1 in [1.0, -1.0]:
        for c2 in [phi, -phi]:
            verts.append((c1, c2, 0.0))
            verts.append((0.0, c1, c2))
            verts.append((c2, 0.0, c1))
    vertices = np.asarray(verts, dtype=np.float32)
    vertices /= np.linalg.norm([1.0, phi])

    faces = np.array(
        [
            (0, 1, 2),
            (0, 6, 1),
            (8, 0, 2),
            (8, 4, 0),
            (3, 8, 2),
            (3, 2, 7),
            (7, 2, 1),
            (0, 4, 6),
            (4, 11, 6),
            (6, 11, 5),
            (1, 5, 7),
            (4, 10, 11),
            (4, 8, 10),
            (10, 8, 3),
            (10, 3, 9),
            (11, 10, 9),
            (11, 9, 5),
            (5, 9, 7),
            (9, 3, 7),
            (1, 6, 5),
        ],
        dtype=np.int32,
    )

    # Rotate to match a conventional orientation (optional but consistent)
    angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
    rotation_angle = (np.pi - angle_between_faces) / 2
    R = transform.Rotation.from_euler(seq="y", angles=rotation_angle).as_matrix()
    vertices = (vertices @ R).astype(np.float32)

    return TriangularMesh(vertices=vertices, faces=faces)


def _split_faces_once(mesh: TriangularMesh) -> TriangularMesh:
    """
    Split each triangular face into 4 by adding mid-edge vertices, then project
    onto unit sphere. Keeps CCW orientation.
    """
    parent_V = mesh.vertices
    faces = mesh.faces

    # Deduplicate child vertices at shared edges via a map
    child_index: dict[tuple[int, int], int] = {}
    all_vertices = list(parent_V)

    def midpoint_index(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in child_index:
            return child_index[key]
        p = (parent_V[i] + parent_V[j]) / 2.0
        p /= np.linalg.norm(p)  # project to unit sphere
        idx = len(all_vertices)
        all_vertices.append(p.astype(np.float32))
        child_index[key] = idx
        return idx

    new_faces = []
    for i1, i2, i3 in faces:
        i12 = midpoint_index(i1, i2)
        i23 = midpoint_index(i2, i3)
        i31 = midpoint_index(i3, i1)
        # 4 child triangles, CCW
        new_faces.extend(
            [
                (i1, i12, i31),
                (i12, i2, i23),
                (i31, i23, i3),
                (i12, i23, i31),
            ]
        )

    return TriangularMesh(
        vertices=np.asarray(all_vertices, dtype=np.float32),
        faces=np.asarray(new_faces, dtype=np.int32),
    )


def merge_meshes(mesh_list: Sequence[TriangularMesh]) -> TriangularMesh:
    """Merges all meshes into one. Assumes the last mesh is the finest.

    Args:
       mesh_list: Sequence of meshes, from coarse to fine refinement levels. The
         vertices and faces may contain those from preceding, coarser levels.

    Returns:
       `TriangularMesh` for which the vertices correspond to the highest
       resolution mesh in the hierarchy, and the faces are the join set of the
       faces at all levels of the hierarchy.
    """
    for mesh_i, mesh_ip1 in itertools.pairwise(mesh_list):
        num_nodes_mesh_i = mesh_i.vertices.shape[0]
        assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])

    return TriangularMesh(
        vertices=mesh_list[-1].vertices,
        faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0),
    )


def faces_to_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transforms polygonal faces to sender and receiver indices.

    It does so by transforming every face into N_i edges. Such if the triangular
    face has indices [0, 1, 2], three edges are added 0->1, 1->2, and 2->0.

    If all faces have consistent orientation, and the surface represented by the
    faces is closed, then every edge in a polygon with a certain orientation
    is also part of another polygon with the opposite orientation. In this
    situation, the edges returned by the method are always bidirectional.

    Args:
      faces: Integer array of shape [num_faces, 3]. Contains node indices
          adjacent to each face.

    Returns:
      Tuple with sender/receiver indices, each of shape [num_edges=num_faces*3].

    """
    assert faces.ndim == 2
    assert faces.shape[-1] == 3
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return senders, receivers


def get_hierarchy_of_triangular_meshes_for_sphere(splits: int) -> list[TriangularMesh]:
    """
    Build M0..Msplits by repeatedly quartering faces on a unit sphere.

    Args:
      splits: number of refinement steps (M0 -> M1 -> ... -> Msplits)

    Returns:
      List of meshes [M0, M1, ..., Msplits]
    """
    meshes = [get_icosahedron()]
    for _ in range(splits):
        meshes.append(_split_faces_once(meshes[-1]))
    return meshes


# ----------------------------- Grid<->Mesh edges ---------------------------


def radius_query_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: TriangularMesh,
    radius: float,
    mask: np.ndarray,
    ocean_mesh: bool = False,
    return_positions: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns mesh-grid edge indices for radius query.

    Args:
      grid_latitude: Latitude values for the grid [num_lat_points].
      grid_longitude: Longitude values for the grid [num_lon_points].
      mesh: Mesh object.
      radius: Radius of connectivity in R3. for a sphere of unit radius.
      mask: Boolean array of locations to mask from data (e.g land points).
      ocean_mesh: Whether to mask land points from the mesh.
      return_positions: Return lat/lon of grid and mesh nodes.

    Returns:
      tuple with `grid_indices` and `mesh_indices` indicating edges between the
      grid and the mesh such that the distances in a straight line (not geodesic)
      are smaller than or equal to `radius`.
      * grid_indices: Indices of shape [num_edges], that index into a
        [num_lat_points, num_lon_points] grid, after flattening the leading axes.
      * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
    """
    # [num_grid_points=num_lat_points * num_lon_points, 3]
    grid_positions = _grid_lat_lon_to_coordinates(grid_latitude, grid_longitude, mask)

    # [num_mesh_points, 3]
    mesh_positions, mesh_latlon = get_mesh_positions(
        mesh.vertices,
        grid_latitude=grid_latitude,
        grid_longitude=grid_longitude,
        mask=mask,
        return_pos=True,
        ocean_mesh=ocean_mesh,
    )
    kd_tree = cKDTree(mesh_positions)

    # [num_grid_points, num_mesh_points_per_grid_point]
    # Note `num_mesh_points_per_grid_point` is not constant, so this is a list
    # of arrays, rather than a 2d array.
    query_indices = kd_tree.query_ball_point(x=grid_positions, r=radius)

    grid_edge_indices = []
    mesh_edge_indices = []
    for grid_index, mesh_neighbors in enumerate(query_indices):
        grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
        mesh_edge_indices.append(mesh_neighbors)

    # [num_edges]
    grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(int)
    mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(int)

    if return_positions:
        grid_pos = np.array([grid_longitude[mask], grid_latitude[mask]]).T
        return grid_edge_indices, mesh_edge_indices, grid_pos, mesh_latlon
    else:
        return grid_edge_indices, mesh_edge_indices, np.array([]), np.array([])


def _grid_lat_lon_to_coordinates(
    grid_latitude: np.ndarray, grid_longitude: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Lat [num_lat] lon [num_lon] to 3d coordinates [num_lat, num_lon, 3]."""
    # Convert to spherical coordinates phi and theta defined in the grid.
    # Each [num_latitude_points, num_longitude_points]
    phi_grid = np.deg2rad(grid_longitude)
    theta_grid = np.deg2rad(90 - grid_latitude)

    # [num_latitude_points, num_longitude_points, 3]
    # Note this assumes unit radius, since for now we model the earth as a
    # sphere of unit radius, and keep any vertical dimension as a regular grid.
    return np.stack(
        [
            (np.cos(phi_grid) * np.sin(theta_grid))[mask],
            (np.sin(phi_grid) * np.sin(theta_grid))[mask],
            np.cos(theta_grid)[mask],
        ],
        axis=-1,
    )


# ----------------------------- Mesh<->Grid edges ---------------------------


def in_mesh_triangle_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: TriangularMesh,
    mask: np.ndarray,
    ocean_mesh: bool = False,
    return_positions: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns mesh-grid edge indices for grid points contained in mesh triangles.

    Args:
      grid_latitude: Latitude values for the grid [num_lat_points].
      grid_longitude: Longitude values for the grid [num_lon_points].
      mesh: Mesh object.
      mask: Mask for ocean grid points.
      ocean_mesh: Whether to mask land points from the mesh.
      return_positions: Return lat/lon of grid and mesh nodes.

    Returns:
      tuple with `grid_indices` and `mesh_indices` indicating edges between the
      grid and the mesh vertices of the triangle that contain each grid point.
      The number of edges is always num_lat_points * num_lon_points * 3
      * grid_indices: Indices of shape [num_edges], that index into a
        [num_lat_points, num_lon_points] grid, after flattening the leading axes.
      * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
    """
    if not GRAPHCAST_AVAIL:
        raise ImportError("GraphCast dependencies (trimesh, rtree) not available.")
    else:
        import trimesh

    # [num_grid_points=num_lat_points * num_lon_points, 3]
    grid_positions = _grid_lat_lon_to_coordinates(grid_latitude, grid_longitude, mask)

    mesh_attrs = get_mesh_positions(
        mesh.vertices,
        grid_latitude=grid_latitude,
        grid_longitude=grid_longitude,
        mask=mask,
        return_pos=True,
        return_pos_ids=True,
        ocean_mesh=ocean_mesh,
    )
    mesh_vertices, mesh_latlon, mesh_mask = mesh_attrs
    mesh_faces = masked_mesh_faces(mesh.faces, np.where(mesh_mask)[0])
    mesh_trimesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)

    # [num_grid_points] with mesh face indices for each grid point.
    _, _, query_face_indices = trimesh.proximity.closest_point(
        mesh_trimesh, grid_positions
    )

    # [num_grid_points, 3] with mesh node indices for each grid point.
    mesh_edge_indices = mesh_faces[query_face_indices]

    # [num_grid_points, 3] with grid node indices, where every row simply contains
    # the row (grid_point) index.
    grid_indices = np.arange(grid_positions.shape[0])
    grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])

    # Flatten to get a regular list.
    # [num_edges=num_grid_points*3]
    mesh_edge_indices = mesh_edge_indices.reshape([-1])
    grid_edge_indices = grid_edge_indices.reshape([-1])

    if return_positions:
        grid_pos = np.array([grid_longitude[mask], grid_latitude[mask]]).T
        return grid_edge_indices, mesh_edge_indices, grid_pos, mesh_latlon
    else:
        return grid_edge_indices, mesh_edge_indices, np.array([]), np.array([])


# ----------------------------- Model Utils -----------------------------


def get_graph_spatial_features(
    *,
    node_lat: np.ndarray,
    node_lon: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    add_node_positions: bool,
    add_node_latitude: bool,
    add_node_longitude: bool,
    add_relative_positions: bool,
    edge_normalization_factor: float | None = None,
    relative_longitude_local_coordinates: bool,
    relative_latitude_local_coordinates: bool,
    sine_cosine_encoding: bool = False,
    encoding_num_freqs: int = 10,
    encoding_multiplicative_factor: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes spatial features for the nodes.

    Args:
      node_lat: Latitudes in the [-90, 90] interval of shape [num_nodes]
      node_lon: Longitudes in the [0, 360] interval of shape [num_nodes]
      senders: Sender indices of shape [num_edges]
      receivers: Receiver indices of shape [num_edges]
      add_node_positions: Add unit norm absolute positions.
      add_node_latitude: Add a feature for latitude (cos(90 - lat))
          Note even if this is set to False, the model may be able to infer the
          longitude from relative features, unless
          `relative_latitude_local_coordinates` is also True, or if there is any
          bias on the relative edge sizes for different longitudes.
      add_node_longitude: Add features for longitude (cos(lon), sin(lon)).
          Note even if this is set to False, the model may be able to infer the
          longitude from relative features, unless
          `relative_longitude_local_coordinates` is also True, or if there is any
          bias on the relative edge sizes for different longitudes.
      add_relative_positions: Whether to relative positions in R3 to the edges.
      edge_normalization_factor: Allows explicitly controlling edge normalization.
          If None, defaults to max edge length. This supports using pre-trained
          model weights with a different graph structure to what it was trained.
      relative_longitude_local_coordinates: If True, relative positions are
          computed in a local space where the receiver is at 0 longitude.
      relative_latitude_local_coordinates: If True, relative positions are
          computed in a local space where the receiver is at 0 latitude.
      sine_cosine_encoding: If True, we will transform the node/edge features
          with sine and cosine functions, similar to NERF.
      encoding_num_freqs: frequency parameter
      encoding_multiplicative_factor: used for calculating the frequency.

    Returns:
      Arrays of shape: [num_nodes, num_features] and [num_edges, num_features].
      with node and edge features.

    """
    num_nodes = node_lat.shape[0]
    num_edges = senders.shape[0]
    dtype = node_lat.dtype
    node_phi, node_theta = lat_lon_deg_to_spherical(node_lat, node_lon)

    # Computing some node features.
    node_features: list[np.ndarray] = []
    if add_node_positions:
        # Already in [-1, 1.] range.
        node_features.extend(spherical_to_cartesian(node_phi, node_theta))

    if add_node_latitude:
        # Using the cos of theta.
        # From 1. (north pole) to -1 (south pole).
        node_features.append(np.cos(node_theta))

    if add_node_longitude:
        # Using the cos and sin, which is already normalized.
        node_features.append(np.cos(node_phi))
        node_features.append(np.sin(node_phi))

    if not node_features:
        node_features = np.zeros([num_nodes, 0], dtype=dtype)
    else:
        node_features = np.stack(node_features, axis=-1)

    # Computing some edge features.
    edge_features = []

    if add_relative_positions:
        relative_position = get_relative_position_in_receiver_local_coordinates(
            node_phi=node_phi,
            node_theta=node_theta,
            senders=senders,
            receivers=receivers,
            latitude_local_coordinates=relative_latitude_local_coordinates,
            longitude_local_coordinates=relative_longitude_local_coordinates,
        )

        # Note this is L2 distance in 3d space, rather than geodesic distance.
        relative_edge_distances = np.linalg.norm(
            relative_position, axis=-1, keepdims=True
        )

        if edge_normalization_factor is None:
            # Normalize to the maximum edge distance. Note that we expect to always
            # have an edge that goes in the opposite direction of any given edge
            # so the distribution of relative positions should be symmetric around
            # zero. So by scaling by the maximum length, we expect all relative
            # positions to fall in the [-1., 1.] interval, and all relative distances
            # to fall in the [0., 1.] interval.
            edge_normalization_factor = relative_edge_distances.max()
        edge_features.append(relative_edge_distances / edge_normalization_factor)
        edge_features.append(relative_position / edge_normalization_factor)

    if not edge_features:
        edge_features = np.zeros([num_edges, 0], dtype=dtype)
    else:
        edge_features = np.concatenate(edge_features, axis=-1)

    if sine_cosine_encoding:

        def sine_cosine_transform(x: np.ndarray) -> np.ndarray:
            freqs = encoding_multiplicative_factor ** np.arange(encoding_num_freqs)
            phases = freqs * x[..., None]
            x_sin = np.sin(phases)
            x_cos = np.cos(phases)
            x_cat = np.concatenate([x_sin, x_cos], axis=-1)
            return x_cat.reshape([x.shape[0], -1])

        node_features = sine_cosine_transform(node_features)
        edge_features = sine_cosine_transform(edge_features)

    return node_features, edge_features


def lat_lon_deg_to_spherical(
    node_lat: np.ndarray,
    node_lon: np.ndarray,
    np_: NumpyInterface = np,
) -> tuple[np.ndarray, np.ndarray]:
    phi = np_.deg2rad(node_lon)
    theta = np_.deg2rad(90 - node_lat)
    return phi, theta


def spherical_to_lat_lon(
    phi: np.ndarray,
    theta: np.ndarray,
    np_: NumpyInterface = np,
) -> tuple[np.ndarray, np.ndarray]:
    lon = np_.mod(np_.rad2deg(phi), 360)
    lat = 90 - np_.rad2deg(theta)
    return lat, lon


def cartesian_to_spherical(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    np_: NumpyInterface = np,
) -> tuple[np.ndarray, np.ndarray]:
    phi = np_.arctan2(y, x)
    with np.errstate(invalid="ignore"):  # circumventing b/253179568
        theta = np_.arccos(z)  # Assuming unit radius.
    return phi, theta


def spherical_to_cartesian(
    phi: np.ndarray,
    theta: np.ndarray,
    np_: NumpyInterface = np,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Assuming unit radius.
    return (
        np_.cos(phi) * np_.sin(theta),
        np_.sin(phi) * np_.sin(theta),
        np_.cos(theta),
    )


def lat_lon_to_cartesian(
    lat: np.ndarray,
    lon: np.ndarray,
    np_: NumpyInterface = np,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return spherical_to_cartesian(*lat_lon_deg_to_spherical(lat, lon, np_=np_), np_=np_)


def cartesian_to_lat_lon(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    np_: NumpyInterface = np,
) -> tuple[np.ndarray, np.ndarray]:
    return spherical_to_lat_lon(*cartesian_to_spherical(x, y, z, np_=np_), np_=np_)


def get_mesh_positions(
    mesh_vertices: np.ndarray,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mask: np.ndarray,
    return_pos: bool = False,
    return_pos_ids: bool = False,
    ocean_mesh: bool = True,
) -> np.ndarray:
    """Returns mesh vertices."""
    mesh_phi, mesh_theta = cartesian_to_spherical(
        mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2]
    )
    mesh_lat, mesh_lon = spherical_to_lat_lon(mesh_phi, mesh_theta)
    if ocean_mesh:
        grid_cart = lat_lon_to_cartesian(
            grid_latitude.flatten(), grid_longitude.flatten()
        )
        mesh_cart = lat_lon_to_cartesian(mesh_lat, mesh_lon)
        grid_points = np.stack(grid_cart, axis=-1)
        mesh_points = np.stack(mesh_cart, axis=-1)
        mask = griddata(
            grid_points, mask.flatten().astype(float), mesh_points, method="nearest"
        ).astype(bool)
    else:
        mask = np.ones_like(mesh_lat, dtype=bool)
    if return_pos:
        a = mesh_vertices[mask]
        b = np.array([mesh_lon[mask], mesh_lat[mask]]).T
        if return_pos_ids:
            return a, b, mask
        else:
            return a, b
    else:
        if return_pos_ids:
            return mesh_vertices[mask], mask
        else:
            return mesh_vertices[mask]


def masked_mesh_faces(faces, mask):
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(mask)}
    valid_faces = []
    for face in faces:
        if all(vertex in old_to_new for vertex in face):
            new_face = [old_to_new[vertex] for vertex in face]
            valid_faces.append(new_face)
    return np.array(valid_faces)


def get_relative_position_in_receiver_local_coordinates(
    node_phi: np.ndarray,
    node_theta: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    latitude_local_coordinates: bool,
    longitude_local_coordinates: bool,
    np_: NumpyInterface = np,
    transform_: TransformInterface = transform,
) -> np.ndarray:
    """Returns relative position features for the edges.

    The relative positions will be computed in a rotated space for a local
    coordinate system as defined by the receiver. The relative positions are
    simply obtained by subtracting sender position minues receiver position in
    that local coordinate system after the rotation in R^3.

    Args:
      node_phi: [num_nodes] with polar angles.
      node_theta: [num_nodes] with azimuthal angles.
      senders: [num_edges] with indices.
      receivers: [num_edges] with indices.
      latitude_local_coordinates: Whether to rotate edges such that in the
          positions are computed such that the receiver is always at latitude 0.
      longitude_local_coordinates: Whether to rotate edges such that in the
          positions are computed such that the receiver is always at longitude 0.
      np_: Numpy library interface.
      transform_: scipy.transform library interface.

    Returns:
      Array of relative positions in R3 [num_edges, 3]
    """
    node_pos = np_.stack(spherical_to_cartesian(node_phi, node_theta, np_=np_), axis=-1)

    # No rotation in this case.
    if not (latitude_local_coordinates or longitude_local_coordinates):
        return node_pos[senders] - node_pos[receivers]

    # Get rotation matrices for the local space space for every node.
    rotation_matrices = get_rotation_matrices_to_local_coordinates(
        reference_phi=node_phi,
        reference_theta=node_theta,
        rotate_latitude=latitude_local_coordinates,
        rotate_longitude=longitude_local_coordinates,
        np_=np_,
        transform_=transform_,
    )

    # Each edge will be rotated according to the rotation matrix of its receiver
    # node.
    edge_rotation_matrices = rotation_matrices[receivers]

    # Rotate all nodes to the rotated space of the corresponding edge.
    # Note for receivers we can also do the matmul first and the gather second:
    # ```
    # receiver_pos_in_rotated_space = rotate_with_matrices(
    #    rotation_matrices, node_pos)[receivers]
    # ```
    # which is more efficient, however, we do gather first to keep it more
    # symmetric with the sender computation.
    receiver_pos_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, node_pos[receivers], np_=np_
    )
    sender_pos_in_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, node_pos[senders], np_=np_
    )
    # Note, here, that because the rotated space is chosen according to the
    # receiver, if:
    # * latitude_local_coordinates = True: latitude for the receivers will be
    #   0, that is the z coordinate will always be 0.
    # * longitude_local_coordinates = True: longitude for the receivers will be
    #   0, that is the y coordinate will be 0.

    # Now we can just subtract.
    # Note we are rotating to a local coordinate system, where the y-z axes are
    # parallel to a tangent plane to the sphere, but still remain in a 3d space.
    # Note that if both `latitude_local_coordinates` and
    # `longitude_local_coordinates` are True, and edges are short,
    # then the difference in x coordinate between sender and receiver
    # should be small, so we could consider dropping the new x coordinate if
    # we wanted to the tangent plane, however in doing so
    # we would lose information about the curvature of the mesh, which may be
    # important for very coarse meshes.
    return sender_pos_in_in_rotated_space - receiver_pos_in_rotated_space


def get_rotation_matrices_to_local_coordinates(
    reference_phi: np.ndarray,
    reference_theta: np.ndarray,
    rotate_latitude: bool,
    rotate_longitude: bool,
    np_: NumpyInterface = np,
    transform_: TransformInterface = transform,
) -> np.ndarray:
    """Returns a rotation matrix to rotate to a point based on a reference vector.

    The rotation matrix is build such that, a vector in the
    same coordinate system at the reference point that points towards the pole
    before the rotation, continues to point towards the pole after the rotation.

    Args:
      reference_phi: [leading_axis] Polar angles of the reference.
      reference_theta: [leading_axis] Azimuthal angles of the reference.
      rotate_latitude: Whether to produce a rotation matrix that would rotate
          R^3 vectors to zero latitude.
      rotate_longitude: Whether to produce a rotation matrix that would rotate
          R^3 vectors to zero longitude.
      np_: Numpy library interface.
      transform_: scipy.transform library interface.

    Returns:
      Matrices of shape [leading_axis] such that when applied to the reference
          position with `rotate_with_matrices(rotation_matrices, reference_pos)`

          * phi goes to 0. if "rotate_longitude" is True.

          * theta goes to np.pi / 2 if "rotate_latitude" is True.

          The rotation consists of:
          * rotate_latitude = False, rotate_longitude = True:
              Latitude preserving rotation.
          * rotate_latitude = True, rotate_longitude = True:
              Latitude preserving rotation, followed by longitude preserving
              rotation.
          * rotate_latitude = True, rotate_longitude = False:
              Latitude preserving rotation, followed by longitude preserving
              rotation, and the inverse of the latitude preserving rotation. Note
              this is computationally different from rotating the longitude only
              and is. We do it like this, so the polar geodesic curve, continues
              to be aligned with one of the axis after the rotation.

    """
    # Azimuthal angle we need to apply to move to zero longitude.
    azimuthal_rotation = -reference_phi

    # Polar angle we need to apply to move from "theta" to zero latitude.
    polar_rotation = -reference_theta + np.pi / 2

    if rotate_longitude and rotate_latitude:
        # We first rotate to zero longitude around the z axis, and then, when the
        # point is at x=0 we can simply apply the polar rotation around the y axis.
        return transform_.Rotation.from_euler(
            "zy", np_.stack([azimuthal_rotation, polar_rotation], axis=1)
        ).as_matrix()
    elif rotate_longitude:
        # Just like the previous case, but applying only the azimuthal rotation,
        # leaving the latitude unchanged.
        return transform_.Rotation.from_euler("z", azimuthal_rotation).as_matrix()
    elif rotate_latitude:
        # We want to apply the polar rotation only, but we don't know the rotation
        # axis to apply a polar rotation. The simplest way to achieve this is to
        # first rotate all the way to longitude 0, then apply the polar rotation
        # arond the y axis, and then rotate back to the original longitude.
        return transform_.Rotation.from_euler(
            "zyz",
            np_.stack(
                [azimuthal_rotation, polar_rotation, -azimuthal_rotation], axis=1
            ),
        ).as_matrix()
    else:
        raise ValueError("At least one of longitude and latitude should be rotated.")


def rotate_with_matrices(
    rotation_matrices: np.ndarray, positions: np.ndarray, np_: NumpyInterface = np
) -> np.ndarray:
    return np_.einsum("...ji,...i->...j", rotation_matrices, positions)


def get_bipartite_graph_spatial_features(
    *,
    senders_node_lat: np.ndarray,
    senders_node_lon: np.ndarray,
    senders: np.ndarray,
    receivers_node_lat: np.ndarray,
    receivers_node_lon: np.ndarray,
    receivers: np.ndarray,
    add_node_positions: bool,
    add_node_latitude: bool,
    add_node_longitude: bool,
    add_relative_positions: bool,
    edge_normalization_factor: float | None = None,
    relative_longitude_local_coordinates: bool,
    relative_latitude_local_coordinates: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes spatial features for the nodes.

    This function is almost identical to `get_graph_spatial_features`. The only
    difference is that sender nodes and receiver nodes can be in different arrays.
    This is necessary to enable combination with typed Graph.

    Args:
      senders_node_lat: Latitudes in the [-90, 90] interval of shape
        [num_sender_nodes]
      senders_node_lon: Longitudes in the [0, 360] interval of shape
        [num_sender_nodes]
      senders: Sender indices of shape [num_edges], indices in [0,
        num_sender_nodes)
      receivers_node_lat: Latitudes in the [-90, 90] interval of shape
        [num_receiver_nodes]
      receivers_node_lon: Longitudes in the [0, 360] interval of shape
        [num_receiver_nodes]
      receivers: Receiver indices of shape [num_edges], indices in [0,
        num_receiver_nodes)
      add_node_positions: Add unit norm absolute positions.
      add_node_latitude: Add a feature for latitude (cos(90 - lat)) Note even if
        this is set to False, the model may be able to infer the longitude from
        relative features, unless `relative_latitude_local_coordinates` is also
        True, or if there is any bias on the relative edge sizes for different
        longitudes.
      add_node_longitude: Add features for longitude (cos(lon), sin(lon)). Note
        even if this is set to False, the model may be able to infer the longitude
        from relative features, unless `relative_longitude_local_coordinates` is
        also True, or if there is any bias on the relative edge sizes for
        different longitudes.
      add_relative_positions: Whether to relative positions in R3 to the edges.
      edge_normalization_factor: Allows explicitly controlling edge normalization.
        If None, defaults to max edge length. This supports using pre-trained
        model weights with a different graph structure to what it was trained on.
      relative_longitude_local_coordinates: If True, relative positions are
        computed in a local space where the receiver is at 0 longitude.
      relative_latitude_local_coordinates: If True, relative positions are
        computed in a local space where the receiver is at 0 latitude.

    Returns:
      Arrays of shape: [num_nodes, num_features] and [num_edges, num_features].
      with node and edge features.

    """
    num_senders = senders_node_lat.shape[0]
    num_receivers = receivers_node_lat.shape[0]
    num_edges = senders.shape[0]
    dtype = senders_node_lat.dtype
    assert receivers_node_lat.dtype == dtype
    senders_node_phi, senders_node_theta = lat_lon_deg_to_spherical(
        senders_node_lat, senders_node_lon
    )
    receivers_node_phi, receivers_node_theta = lat_lon_deg_to_spherical(
        receivers_node_lat, receivers_node_lon
    )

    # Computing some node features.
    senders_node_features: list[np.ndarray] = []
    receivers_node_features: list[np.ndarray] = []
    if add_node_positions:
        # Already in [-1, 1.] range.
        senders_node_features.extend(
            spherical_to_cartesian(senders_node_phi, senders_node_theta)
        )
        receivers_node_features.extend(
            spherical_to_cartesian(receivers_node_phi, receivers_node_theta)
        )

    if add_node_latitude:
        # Using the cos of theta.
        # From 1. (north pole) to -1 (south pole).
        senders_node_features.append(np.cos(senders_node_theta))
        receivers_node_features.append(np.cos(receivers_node_theta))

    if add_node_longitude:
        # Using the cos and sin, which is already normalized.
        senders_node_features.append(np.cos(senders_node_phi))
        senders_node_features.append(np.sin(senders_node_phi))

        receivers_node_features.append(np.cos(receivers_node_phi))
        receivers_node_features.append(np.sin(receivers_node_phi))

    if not senders_node_features:
        senders_node_features = np.zeros([num_senders, 0], dtype=dtype)
        receivers_node_features = np.zeros([num_receivers, 0], dtype=dtype)
    else:
        senders_node_features = np.stack(senders_node_features, axis=-1)
        receivers_node_features = np.stack(receivers_node_features, axis=-1)

    # Computing some edge features.
    edge_features = []

    if add_relative_positions:
        relative_position = (
            get_bipartite_relative_position_in_receiver_local_coordinates(  # pylint: disable=line-too-long
                senders_node_phi=senders_node_phi,
                senders_node_theta=senders_node_theta,
                receivers_node_phi=receivers_node_phi,
                receivers_node_theta=receivers_node_theta,
                senders=senders,
                receivers=receivers,
                latitude_local_coordinates=relative_latitude_local_coordinates,
                longitude_local_coordinates=relative_longitude_local_coordinates,
            )
        )

        # Note this is L2 distance in 3d space, rather than geodesic distance.
        relative_edge_distances = np.linalg.norm(
            relative_position, axis=-1, keepdims=True
        )

        if edge_normalization_factor is None:
            # Normalize to the maximum edge distance. Note that we expect to always
            # have an edge that goes in the opposite direction of any given edge
            # so the distribution of relative positions should be symmetric around
            # zero. So by scaling by the maximum length, we expect all relative
            # positions to fall in the [-1., 1.] interval, and all relative distances
            # to fall in the [0., 1.] interval.
            edge_normalization_factor = relative_edge_distances.max()

        edge_features.append(relative_edge_distances / edge_normalization_factor)
        edge_features.append(relative_position / edge_normalization_factor)

    if not edge_features:
        edge_features = np.zeros([num_edges, 0], dtype=dtype)
    else:
        edge_features = np.concatenate(edge_features, axis=-1)

    return senders_node_features, receivers_node_features, edge_features


def get_bipartite_relative_position_in_receiver_local_coordinates(
    senders_node_phi: np.ndarray,
    senders_node_theta: np.ndarray,
    senders: np.ndarray,
    receivers_node_phi: np.ndarray,
    receivers_node_theta: np.ndarray,
    receivers: np.ndarray,
    latitude_local_coordinates: bool,
    longitude_local_coordinates: bool,
    np_: NumpyInterface = np,
    transform_: TransformInterface = transform,
) -> np.ndarray:
    """Returns relative position features for the edges.

    This function is equivalent to
    `get_relative_position_in_receiver_local_coordinates`, but adapted to work
    with bipartite typed graphs.

    The relative positions will be computed in a rotated space for a local
    coordinate system as defined by the receiver. The relative positions are
    simply obtained by subtracting sender position minues receiver position in
    that local coordinate system after the rotation in R^3.

    Args:
      senders_node_phi: [num_sender_nodes] with polar angles.
      senders_node_theta: [num_sender_nodes] with azimuthal angles.
      senders: [num_edges] with indices into sender nodes.
      receivers_node_phi: [num_sender_nodes] with polar angles.
      receivers_node_theta: [num_sender_nodes] with azimuthal angles.
      receivers: [num_edges] with indices into receiver nodes.
      latitude_local_coordinates: Whether to rotate edges such that in the
        positions are computed such that the receiver is always at latitude 0.
      longitude_local_coordinates: Whether to rotate edges such that in the
        positions are computed such that the receiver is always at longitude 0.
      np_: Numpy library interface.
      transform_: scipy.transform library interface.

    Returns:
      Array of relative positions in R3 [num_edges, 3]
    """
    senders_node_pos = np_.stack(
        spherical_to_cartesian(senders_node_phi, senders_node_theta, np_=np_), axis=-1
    )

    receivers_node_pos = np_.stack(
        spherical_to_cartesian(receivers_node_phi, receivers_node_theta, np_=np_),
        axis=-1,
    )

    # No rotation in this case.
    if not (latitude_local_coordinates or longitude_local_coordinates):
        return senders_node_pos[senders] - receivers_node_pos[receivers]

    # Get rotation matrices for the local space space for every receiver node.
    receiver_rotation_matrices = get_rotation_matrices_to_local_coordinates(
        reference_phi=receivers_node_phi,
        reference_theta=receivers_node_theta,
        rotate_latitude=latitude_local_coordinates,
        rotate_longitude=longitude_local_coordinates,
        np_=np_,
        transform_=transform_,
    )

    # Each edge will be rotated according to the rotation matrix of its receiver
    # node.
    edge_rotation_matrices = receiver_rotation_matrices[receivers]

    # Rotate all nodes to the rotated space of the corresponding edge.
    # Note for receivers we can also do the matmul first and the gather second:
    # ```
    # receiver_pos_in_rotated_space = rotate_with_matrices(
    #    rotation_matrices, node_pos)[receivers]
    # ```
    # which is more efficient, however, we do gather first to keep it more
    # symmetric with the sender computation.
    receiver_pos_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, receivers_node_pos[receivers], np_=np_
    )
    sender_pos_in_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, senders_node_pos[senders], np_=np_
    )
    # Note, here, that because the rotated space is chosen according to the
    # receiver, if:
    # * latitude_local_coordinates = True: latitude for the receivers will be
    #   0, that is the z coordinate will always be 0.
    # * longitude_local_coordinates = True: longitude for the receivers will be
    #   0, that is the y coordinate will be 0.

    # Now we can just subtract.
    # Note we are rotating to a local coordinate system, where the y-z axes are
    # parallel to a tangent plane to the sphere, but still remain in a 3d space.
    # Note that if both `latitude_local_coordinates` and
    # `longitude_local_coordinates` are True, and edges are short,
    # then the difference in x coordinate between sender and receiver
    # should be small, so we could consider dropping the new x coordinate if
    # we wanted to the tangent plane, however in doing so
    # we would lose information about the curvature of the mesh, which may be
    # important for very coarse meshes.
    return sender_pos_in_in_rotated_space - receiver_pos_in_rotated_space


# ----------------------------- Convenience helpers -------------------------


def get_max_edge_distance(faces: np.ndarray, vertices: np.ndarray) -> float:
    senders, receivers = faces_to_edges(faces)
    edge_distances = np.linalg.norm(vertices[senders] - vertices[receivers], axis=-1)
    return edge_distances.max()

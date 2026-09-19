"""
EarthMesh Representation Framework for SIH-26078.
Provides extensible spherical geometry representations for atmospheric neural operators:
1. EarthMesh (Abstract Base Class)
2. SphericalLatLonMesh: Regular regional lat/lon grid mapped to 3D Cartesian coordinates with Haversine edges.
3. IcosahedralGeodesicMesh: Subdivided regular icosahedron projected onto the sphere with geodesic great-circle connectivity.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from backend.app.config import domain_config, DomainConfig

EARTH_RADIUS_KM = 6371.0


class EarthMesh(ABC):
    """Abstract Base Class for spherical and geodesic atmospheric graph representations."""

    @abstractmethod
    def grid_to_node_features(self, field_tensor: torch.Tensor) -> torch.Tensor:
        """Converts a grid tensor (Batch, Channels, H, W) to (Batch, Num_Nodes, Channels + 3)."""
        pass

    @abstractmethod
    def node_features_to_grid(self, node_features: torch.Tensor, n_channels: Optional[int] = None) -> torch.Tensor:
        """Converts node features (Batch, Num_Nodes, Channels) to (Batch, Channels, H, W)."""
        pass


class SphericalLatLonMesh(EarthMesh):
    """
    Constructs a graph representation of a regular atmospheric grid on a sphere.
    Nodes represent geographical grid points with (x, y, z) 3D Cartesian coordinates and atmospheric features.
    Edges connect spatial neighbors (8-connectivity) with Haversine distance-based weights and bearings.
    """
    def __init__(self, config: DomainConfig = domain_config, connectivity: str = "8-conn"):
        self.config = config
        self.mesh_type = "SphericalLatLonMesh"
        self.lats = np.arange(config.lat_min, config.lat_max + 1e-5, config.grid_res_deg)
        self.lons = np.arange(config.lon_min, config.lon_max + 1e-5, config.grid_res_deg)
        self.n_lats = len(self.lats)
        self.n_lons = len(self.lons)
        self.n_nodes = self.n_lats * self.n_lons
        self.connectivity = connectivity

        self.lat_grid, self.lon_grid = np.meshgrid(self.lats, self.lons, indexing="ij")
        self.coords_3d = self._compute_3d_coordinates(self.lat_grid.flatten(), self.lon_grid.flatten())
        self.edge_index, self.edge_attr = self._build_graph_topology()

    def _compute_3d_coordinates(self, lats_deg: np.ndarray, lons_deg: np.ndarray) -> np.ndarray:
        phi = np.radians(lats_deg)
        lam = np.radians(lons_deg)
        x = np.cos(phi) * np.cos(lam)
        y = np.cos(phi) * np.sin(lam)
        z = np.sin(phi)
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp = np.radians(lat2 - lat1)
        dl = np.radians(lon2 - lon1)
        a = np.sin(dp / 2.0)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0)**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return EARTH_RADIUS_KM * c

    def _build_graph_topology(self) -> Tuple[torch.Tensor, torch.Tensor]:
        src_list, dst_list = [], []
        attr_list = []

        neighbor_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ] if self.connectivity == "8-conn" else [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for i in range(self.n_lats):
            for j in range(self.n_lons):
                node_idx = i * self.n_lons + j
                lat_i = self.lats[i]
                lon_j = self.lons[j]

                for di, dj in neighbor_offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.n_lats and 0 <= nj < self.n_lons:
                        neighbor_idx = ni * self.n_lons + nj
                        dist = self._haversine_distance(lat_i, lon_j, self.lats[ni], self.lons[nj])
                        d_lat = self.lats[ni] - lat_i
                        d_lon = self.lons[nj] - lon_j
                        bearing = np.arctan2(d_lon, d_lat)

                        src_list.append(node_idx)
                        dst_list.append(neighbor_idx)
                        attr_list.append([dist / 1000.0, bearing / np.pi])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(attr_list, dtype=torch.float32)
        return edge_index, edge_attr

    def grid_to_node_features(self, field_tensor: torch.Tensor) -> torch.Tensor:
        if field_tensor.dim() == 3:
            field_tensor = field_tensor.unsqueeze(0)

        batch_size, channels, H, W = field_tensor.shape
        assert H == self.n_lats and W == self.n_lons, f"Grid dimensions ({H}, {W}) do not match graph ({self.n_lats}, {self.n_lons})"

        flat_fields = field_tensor.view(batch_size, channels, -1).permute(0, 2, 1)
        coords_3d_tensor = torch.from_numpy(self.coords_3d).to(field_tensor.device).unsqueeze(0).expand(batch_size, -1, -1)
        node_features = torch.cat([flat_fields, coords_3d_tensor], dim=2)
        return node_features

    def node_features_to_grid(self, node_features: torch.Tensor, n_channels: Optional[int] = None) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)

        batch_size, n_nodes, channels = node_features.shape
        if n_channels is not None:
            node_features = node_features[:, :, :n_channels]
            channels = n_channels

        grid_tensor = node_features.permute(0, 2, 1).view(batch_size, channels, self.n_lats, self.n_lons)
        return grid_tensor


class IcosahedralGeodesicMesh(EarthMesh):
    """
    Subdivided regular icosahedron projected onto the sphere with geodesic great-circle connectivity.
    Provides isotropic, quasi-uniform spatial sampling without polar singularities.
    Includes fast KDTree spatial interpolation to/from regional lat-lon grids.
    """
    def __init__(self, subdivision_level: int = 3, config: DomainConfig = domain_config):
        self.config = config
        self.subdivision_level = subdivision_level
        self.mesh_type = "IcosahedralGeodesicMesh"
        self.lats = np.arange(config.lat_min, config.lat_max + 1e-5, config.grid_res_deg)
        self.lons = np.arange(config.lon_min, config.lon_max + 1e-5, config.grid_res_deg)
        self.n_lats = len(self.lats)
        self.n_lons = len(self.lons)

        # Generate icosahedral vertices & faces
        self.coords_3d, self.faces = self._generate_subdivided_icosahedron(subdivision_level)
        
        # Filter vertices to regional domain of interest with margin
        self._filter_regional_mesh()
        self.n_nodes = len(self.coords_3d)
        self.edge_index, self.edge_attr = self._build_icosahedral_topology()

        # Precompute KD-Tree for spatial interpolation
        self.lat_grid, self.lon_grid = np.meshgrid(self.lats, self.lons, indexing="ij")
        grid_coords = self._latlon_to_cartesian(self.lat_grid.flatten(), self.lon_grid.flatten())
        
        self.mesh_kdtree = cKDTree(self.coords_3d)
        self.grid_kdtree = cKDTree(grid_coords)
        
        # Grid -> Node nearest indices
        _, self.grid_to_node_idx = self.mesh_kdtree.query(grid_coords, k=1)
        _, self.node_to_grid_idx = self.grid_kdtree.query(self.coords_3d, k=1)

    def _latlon_to_cartesian(self, lats_deg: np.ndarray, lons_deg: np.ndarray) -> np.ndarray:
        phi = np.radians(lats_deg)
        lam = np.radians(lons_deg)
        x = np.cos(phi) * np.cos(lam)
        y = np.cos(phi) * np.sin(lam)
        z = np.sin(phi)
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def _cartesian_to_latlon(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        lats = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
        lons = np.degrees(np.arctan2(y, x))
        return lats, lons

    def _generate_subdivided_icosahedron(self, level: int) -> Tuple[np.ndarray, List[List[int]]]:
        # 12 base vertices of a regular icosahedron
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        verts = [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ]
        verts = np.array(verts, dtype=np.float32)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

        # Recursive midpoint subdivision & spherical projection
        vert_list = verts.tolist()
        midpoint_cache = {}

        def get_midpoint(i1: int, i2: int) -> int:
            edge = tuple(sorted((i1, i2)))
            if edge in midpoint_cache:
                return midpoint_cache[edge]
            v1, v2 = np.array(vert_list[i1]), np.array(vert_list[i2])
            mid = (v1 + v2) / 2.0
            mid = (mid / np.linalg.norm(mid)).tolist()
            vert_list.append(mid)
            idx = len(vert_list) - 1
            midpoint_cache[edge] = idx
            return idx

        curr_faces = faces
        for _ in range(level):
            new_faces = []
            for tri in curr_faces:
                v1, v2, v3 = tri[0], tri[1], tri[2]
                a = get_midpoint(v1, v2)
                b = get_midpoint(v2, v3)
                c = get_midpoint(v3, v1)
                new_faces.extend([[v1, a, c], [v2, b, a], [v3, c, b], [a, b, c]])
            curr_faces = new_faces

        return np.array(vert_list, dtype=np.float32), curr_faces

    def _filter_regional_mesh(self):
        lats, lons = self._cartesian_to_latlon(self.coords_3d)
        margin = 3.0
        mask = (
            (lats >= self.config.lat_min - margin) & (lats <= self.config.lat_max + margin) &
            (lons >= self.config.lon_min - margin) & (lons <= self.config.lon_max + margin)
        )
        if np.sum(mask) >= 64:
            self.coords_3d = self.coords_3d[mask]
        # Normalize to unit sphere
        self.coords_3d = self.coords_3d / np.linalg.norm(self.coords_3d, axis=1, keepdims=True)

    def _build_icosahedral_topology(self, k_neighbors: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
        tree = cKDTree(self.coords_3d)
        distances, indices = tree.query(self.coords_3d, k=min(k_neighbors + 1, len(self.coords_3d)))
        
        src_list, dst_list = [], []
        attr_list = []
        lats, lons = self._cartesian_to_latlon(self.coords_3d)

        for i in range(len(self.coords_3d)):
            for n_idx in range(1, indices.shape[1]):
                j = indices[i, n_idx]
                dist_km = float(distances[i, n_idx] * EARTH_RADIUS_KM)
                d_lat = lats[j] - lats[i]
                d_lon = lons[j] - lons[i]
                bearing = float(np.arctan2(d_lon, d_lat))

                src_list.append(i)
                dst_list.append(j)
                attr_list.append([dist_km / 1000.0, bearing / np.pi])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(attr_list, dtype=torch.float32)
        return edge_index, edge_attr

    def grid_to_node_features(self, field_tensor: torch.Tensor) -> torch.Tensor:
        if field_tensor.dim() == 3:
            field_tensor = field_tensor.unsqueeze(0)

        batch_size, channels, H, W = field_tensor.shape
        flat_fields = field_tensor.view(batch_size, channels, -1) # (B, C, H*W)
        
        # Sample grid values at node locations
        node_vals = flat_fields[:, :, self.node_to_grid_idx].permute(0, 2, 1) # (B, N, C)
        coords_tensor = torch.from_numpy(self.coords_3d).to(field_tensor.device).unsqueeze(0).expand(batch_size, -1, -1)
        node_features = torch.cat([node_vals, coords_tensor], dim=2) # (B, N, C+3)
        return node_features

    def node_features_to_grid(self, node_features: torch.Tensor, n_channels: Optional[int] = None) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)

        batch_size, n_nodes, channels = node_features.shape
        if n_channels is not None:
            node_features = node_features[:, :, :n_channels]
            channels = n_channels

        # Interpolate back from nodes to 2D grid
        grid_flat = node_features[:, self.grid_to_node_idx, :] # (B, H*W, C)
        grid_tensor = grid_flat.permute(0, 2, 1).view(batch_size, channels, self.n_lats, self.n_lons)
        return grid_tensor


# Backwards compatibility aliases & default singleton
SphericalGraphBuilder = SphericalLatLonMesh
SphericalAtmosphericGraph = SphericalLatLonMesh
SphericalAtmosphericMesh = SphericalLatLonMesh
spherical_graph = SphericalLatLonMesh()
icosahedral_mesh = IcosahedralGeodesicMesh(subdivision_level=3)

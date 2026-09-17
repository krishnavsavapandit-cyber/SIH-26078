"""
Spherical and Spatial Graph Mesh Construction for Atmospheric Grids.
Maps 2D Regional Lat/Lon Weather Grids into 3D Geodesic / Spherical Graph Networks with Haversine Edge Weighting.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from backend.app.config import domain_config, DomainConfig

EARTH_RADIUS_KM = 6371.0

class SphericalAtmosphericGraph:
    """
    Constructs a graph representation of a regular atmospheric grid on a sphere.
    Nodes represent geographical grid points with (x, y, z) 3D Cartesian coordinates and atmospheric features.
    Edges connect spatial neighbors (4-connectivity or 8-connectivity or k-NN) with Haversine distance-based weights.
    """
    def __init__(self, config: DomainConfig = domain_config, connectivity: str = "8-conn", k_neighbors: int = 8):
        self.config = config
        self.lats = np.arange(config.lat_min, config.lat_max + 1e-5, config.grid_res_deg)
        self.lons = np.arange(config.lon_min, config.lon_max + 1e-5, config.grid_res_deg)
        self.n_lats = len(self.lats)
        self.n_lons = len(self.lons)
        self.n_nodes = self.n_lats * self.n_lons
        self.connectivity = connectivity
        self.k_neighbors = k_neighbors

        # Precompute 3D coordinates and graph topology
        self.lat_grid, self.lon_grid = np.meshgrid(self.lats, self.lons, indexing="ij")
        self.coords_3d = self._compute_3d_coordinates(self.lat_grid.flatten(), self.lon_grid.flatten())
        self.edge_index, self.edge_attr = self._build_graph_topology()

    def _compute_3d_coordinates(self, lats_deg: np.ndarray, lons_deg: np.ndarray) -> np.ndarray:
        """
        Converts (lat, lon) degrees to 3D Cartesian coordinates on unit sphere.
        x = cos(lat) * cos(lon)
        y = cos(lat) * sin(lon)
        z = sin(lat)
        """
        phi = np.radians(lats_deg)
        lam = np.radians(lons_deg)
        x = np.cos(phi) * np.cos(lam)
        y = np.cos(phi) * np.sin(lam)
        z = np.sin(phi)
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Computes great-circle distance in kilometers between two points on Earth."""
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp = np.radians(lat2 - lat1)
        dl = np.radians(lon2 - lon1)
        a = np.sin(dp / 2.0)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0)**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return EARTH_RADIUS_KM * c

    def _build_graph_topology(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds edge_index (2, num_edges) and edge_attr (num_edges, 2) [distance, relative bearing].
        """
        src_list, dst_list = [], []
        attr_list = []

        # 8-connectivity grid offsets
        if self.connectivity == "8-conn":
            neighbor_offsets = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
        else: # 4-conn
            neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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
                        # Relative bearing / orientation
                        d_lat = self.lats[ni] - lat_i
                        d_lon = self.lons[nj] - lon_j
                        bearing = np.arctan2(d_lon, d_lat)

                        src_list.append(node_idx)
                        dst_list.append(neighbor_idx)
                        # Normalize distance (scale by typical regional radius ~ 1000km)
                        attr_list.append([dist / 1000.0, bearing / np.pi])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(attr_list, dtype=torch.float32)
        return edge_index, edge_attr

    def grid_to_node_features(self, field_tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts a 2D or 3D atmospheric grid tensor (Batch, Channels, H, W) or (Channels, H, W)
        into Graph Node features (Batch, Num_Nodes, Channels + 3) including 3D coordinates.
        """
        if field_tensor.dim() == 3:
            # (Channels, H, W) -> (1, Channels, H, W)
            field_tensor = field_tensor.unsqueeze(0)

        batch_size, channels, H, W = field_tensor.shape
        assert H == self.n_lats and W == self.n_lons, f"Grid dimensions ({H}, {W}) do not match graph ({self.n_lats}, {self.n_lons})"

        # Reshape to (Batch, H*W, Channels)
        flat_fields = field_tensor.view(batch_size, channels, -1).permute(0, 2, 1) # (B, N, C)

        # Append 3D coordinates (x, y, z)
        coords_3d_tensor = torch.from_numpy(self.coords_3d).to(field_tensor.device).unsqueeze(0).expand(batch_size, -1, -1)
        node_features = torch.cat([flat_fields, coords_3d_tensor], dim=2) # (B, N, C + 3)
        return node_features

    def node_features_to_grid(self, node_features: torch.Tensor, n_channels: Optional[int] = None) -> torch.Tensor:
        """
        Converts Graph Node features (Batch, Num_Nodes, Out_Channels)
        back into 2D Grid format (Batch, Out_Channels, H, W).
        """
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)

        batch_size, n_nodes, channels = node_features.shape
        if n_channels is not None:
            node_features = node_features[:, :, :n_channels]
            channels = n_channels

        # (B, N, C) -> (B, C, N) -> (B, C, H, W)
        grid_tensor = node_features.permute(0, 2, 1).view(batch_size, channels, self.n_lats, self.n_lons)
        return grid_tensor

# Singleton instance
spherical_graph = SphericalAtmosphericGraph()

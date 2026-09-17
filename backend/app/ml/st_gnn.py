"""
Spatio-Temporal Graph Neural Network (ST-GNN) for Atmospheric Extreme Weather Tracking.
Integrates Spherical Graph Message Passing with Recurrent Temporal Dynamics across Forecast Lead Times.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from backend.app.config import domain_config, DomainConfig
from backend.app.ml.spherical_graph import spherical_graph, SphericalAtmosphericGraph
from backend.app.core.synthetic_engine import SyntheticWeatherEngine

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "storage" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class SphericalGraphConv(nn.Module):
    """
    Spherical Spatial Message Passing Layer.
    Aggregates neighbor features weighted by spherical Haversine distance and directional bearings.
    """
    def __init__(self, in_features: int, out_features: int, edge_dim: int = 2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Edge-conditioned message MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_features * 2 + edge_dim, out_features),
            nn.LeakyReLU(0.1),
            nn.Linear(out_features, out_features)
        )
        
        # Node update layer
        self.node_update = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.LayerNorm(out_features),
            nn.LeakyReLU(0.1)
        )
        
        # Self-loop residual linear
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Num_Nodes, In_Features)
        edge_index: (2, Num_Edges)
        edge_attr: (Num_Edges, Edge_Dim)
        """
        batch_size, n_nodes, in_feat = x.shape
        src_idx = edge_index[0] # (E,)
        dst_idx = edge_index[1] # (E,)
        n_edges = edge_index.shape[1]

        # Gather node features for edges
        # x_src: (B, E, in_feat), x_dst: (B, E, in_feat)
        x_src = x[:, src_idx, :]
        x_dst = x[:, dst_idx, :]
        
        # Expand edge_attr: (B, E, edge_dim)
        edge_attr_exp = edge_attr.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute messages: (B, E, out_features)
        edge_inputs = torch.cat([x_src, x_dst, edge_attr_exp], dim=2)
        messages = self.msg_mlp(edge_inputs)

        # Aggregate messages at destination nodes using scatter_add
        aggregated = torch.zeros(batch_size, n_nodes, self.out_features, device=x.device, dtype=x.dtype)
        # Expand dst_idx for scatter_add: (B, E, out_features)
        dst_idx_exp = dst_idx.view(1, -1, 1).expand(batch_size, -1, self.out_features)
        aggregated.scatter_add_(1, dst_idx_exp, messages)

        # Degree normalization
        degrees = torch.zeros(n_nodes, device=x.device, dtype=torch.float32)
        degrees.scatter_add_(0, dst_idx, torch.ones(n_edges, device=x.device))
        degrees = torch.clamp(degrees, min=1.0).view(1, -1, 1)
        aggregated = aggregated / degrees

        # Update node states
        updated = self.node_update(torch.cat([x, aggregated], dim=2))
        return updated + self.residual(x)

class SpatioTemporalGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network architecture.
    Applies spherical spatial graph convolutions followed by a temporal GRU over forecast lead times.
    """
    def __init__(
        self,
        in_channels: int = 8, # 5 atmospheric channels + 3 (x,y,z) coords
        hidden_dim: int = 32,
        num_spatial_layers: int = 2,
        edge_dim: int = 2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Spatial message passing layers
        self.spatial_layers = nn.ModuleList([
            SphericalGraphConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
            for _ in range(num_spatial_layers)
        ])

        # Temporal GRU across forecast sequence
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Output prediction heads
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1)
        )

        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 2) # (u_vel, v_vel)
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_seq: (Batch, Seq_Len, Num_Nodes, In_Channels)
        returns:
            anomaly_logits: (Batch, Seq_Len, Num_Nodes, 1)
            velocities: (Batch, Seq_Len, 2) global centroid shift dynamics
        """
        batch_size, seq_len, n_nodes, in_channels = x_seq.shape

        # Process each timestep spatially
        spatial_states = []
        for t in range(seq_len):
            xt = x_seq[:, t, :, :] # (B, N, C)
            h = self.input_proj(xt)
            for layer in self.spatial_layers:
                h = layer(h, edge_index, edge_attr)
            spatial_states.append(h)

        # Stack temporal sequence: (B, T, N, H)
        spatial_seq = torch.stack(spatial_states, dim=1)

        # Reshape for GRU: Treat (B * N) as batch
        # (B, T, N, H) -> (B, N, T, H) -> (B*N, T, H)
        spatial_seq_perm = spatial_seq.permute(0, 2, 1, 3).contiguous().view(batch_size * n_nodes, seq_len, self.hidden_dim)
        gru_out, _ = self.temporal_gru(spatial_seq_perm) # (B*N, T, H)

        # Reshape back: (B, N, T, H) -> (B, T, N, H)
        temporal_features = gru_out.view(batch_size, n_nodes, seq_len, self.hidden_dim).permute(0, 2, 1, 3)

        # Anomaly segmentation logits
        anomaly_logits = self.anomaly_head(temporal_features) # (B, T, N, 1)

        # Global spatial pooling for velocity dynamics
        global_pool = temporal_features.mean(dim=2) # (B, T, H)
        velocities = self.velocity_head(global_pool) # (B, T, 2)

        return anomaly_logits, velocities

class STGNNManager:
    """
    Manages execution, training, and robust fallback for Spatio-Temporal GNN.
    """
    def __init__(self, config: DomainConfig = domain_config):
        self.config = config
        self.graph = spherical_graph
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpatioTemporalGNN(in_channels=8, hidden_dim=32, num_spatial_layers=2).to(self.device)
        self.weights_path = MODEL_DIR / "st_gnn.pt"
        self._load_or_init_weights()

    def _load_or_init_weights(self):
        if self.weights_path.exists():
            try:
                state_dict = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
            except Exception:
                pass

    def train_st_gnn(self, epochs: int = 3, batch_size: int = 1) -> Dict[str, Any]:
        """
        Trains the ST-GNN on synthetic meteorological sequence trajectories.
        """
        engine = SyntheticWeatherEngine(self.config)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        edge_index = self.graph.edge_index.to(self.device)
        edge_attr = self.graph.edge_attr.to(self.device)

        # Generate sequence training data
        train_scenarios = [("monsoon_depression", 401), ("coastal_cyclone", 402), ("convective_cluster", 403)]
        dataset_samples = []

        for sc_type, seed in train_scenarios:
            ds, _ = engine.generate_scenario(run_id=f"st_train_{sc_type}_{seed}", scenario_type=sc_type, seed=seed)
            # Normalize channels
            p = ds["precipitation"].values[:, 0] / 100.0
            m = (ds["mslp"].values[:, 0] - 1000.0) / 15.0
            t = (ds["temperature_2m"].values[:, 0] - 300.0) / 10.0
            u = ds["u_wind_850"].values[:, 0] / 20.0
            v = ds["v_wind_850"].values[:, 0] / 20.0
            gt = ds["gt_event_mask"].values # (leads, lats, lons)

            stacked = np.stack([p, m, t, u, v], axis=1) # (T, 5, H, W)
            # Convert each timestep to graph node features
            node_feats = []
            for t_idx in range(stacked.shape[0]):
                t_tensor = torch.from_numpy(stacked[t_idx:t_idx+1]).float()
                nf = self.graph.grid_to_node_features(t_tensor) # (1, N, 8)
                node_feats.append(nf.squeeze(0))
            
            node_seq = torch.stack(node_feats, dim=0) # (T, N, 8)
            gt_flat = torch.from_numpy(gt.reshape(gt.shape[0], -1, 1)).float() # (T, N, 1)
            dataset_samples.append((node_seq, gt_flat))

        loss_history = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for node_seq, gt_flat in dataset_samples:
                x_in = node_seq.unsqueeze(0).to(self.device) # (1, T, N, 8)
                y_in = gt_flat.unsqueeze(0).to(self.device)  # (1, T, N, 1)

                optimizer.zero_grad()
                logits, _ = self.model(x_in, edge_index, edge_attr)
                loss = criterion(logits, y_in)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            loss_history.append(epoch_loss / len(dataset_samples))

        # Save weights
        torch.save(self.model.state_dict(), self.weights_path)
        self.model.eval()

        return {
            "status": "trained",
            "epochs": epochs,
            "final_loss": float(loss_history[-1]),
            "loss_history": [float(l) for l in loss_history]
        }

    def predict_spatiotemporal_anomalies(
        self,
        precip_seq: np.ndarray,
        mslp_seq: np.ndarray,
        temp_seq: np.ndarray,
        u_seq: np.ndarray,
        v_seq: np.ndarray
    ) -> Dict[str, Any]:
        """
        Executes ST-GNN inference across lead times with safe fallback.
        Inputs are sequences of shape (Leads, H, W).
        Returns predicted anomaly probability grids (Leads, H, W) and spatio-temporal dynamics.
        """
        self.model.eval()
        try:
            seq_len, H, W = precip_seq.shape
            p = precip_seq / 100.0
            m = (mslp_seq - 1000.0) / 15.0
            t = (temp_seq - 300.0) / 10.0
            u = u_seq / 20.0
            v = v_seq / 20.0

            stacked = np.stack([p, m, t, u, v], axis=1) # (T, 5, H, W)
            node_feats = []
            for t_idx in range(seq_len):
                t_tensor = torch.from_numpy(stacked[t_idx:t_idx+1]).float()
                nf = self.graph.grid_to_node_features(t_tensor) # (1, N, 8)
                node_feats.append(nf.squeeze(0))

            x_seq = torch.stack(node_feats, dim=0).unsqueeze(0).to(self.device) # (1, T, N, 8)
            edge_index = self.graph.edge_index.to(self.device)
            edge_attr = self.graph.edge_attr.to(self.device)

            with torch.no_grad():
                logits, vels = self.model(x_seq, edge_index, edge_attr)
                probs = torch.sigmoid(logits).squeeze(0) # (T, N, 1)
                
                # Convert back to 2D grid sequence
                prob_grids = []
                for t_idx in range(seq_len):
                    grid_t = self.graph.node_features_to_grid(probs[t_idx:t_idx+1, :, :], n_channels=1)
                    prob_grids.append(grid_t.squeeze().cpu().numpy())

            prob_array = np.stack(prob_grids, axis=0) # (T, H, W)
            vel_array = vels.squeeze(0).cpu().numpy()  # (T, 2)

            return {
                "success": True,
                "model": "SpatioTemporalGNN",
                "probabilities": prob_array,
                "velocities": vel_array,
                "fallback_triggered": False
            }
        except Exception as e:
            # Safe Fallback to standard heuristic probability
            prob_fallback = np.clip(precip_seq / 100.0, 0.0, 1.0)
            return {
                "success": False,
                "model": "Fallback_PrecipHeuristic",
                "probabilities": prob_fallback,
                "velocities": np.zeros((precip_seq.shape[0], 2)),
                "fallback_triggered": True,
                "error": str(e)
            }

# Singleton instance
st_gnn_manager = STGNNManager()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional


class GraphBlock(nn.Module):
    """
    Graph encoder block:
    - Adds sensor embeddings
    - Adds previous activity embeddings
    - Runs GNN encoder
    - Pools node embeddings → graph embedding
    """

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        embed_dim: int,
        hidden: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()

        self.num_sensors = num_sensors

        
        # Embeddings
        self.sensor_emb = SensorEmbedding(num_sensors, embed_dim)
        self.prev_act_emb = nn.Embedding(num_classes + 1, embed_dim)

        self.input_drop = nn.Dropout(dropout)

        
        # Input feature dimension
        # MUST match window_to_graph node features
        node_feature_dim = 15
        in_dim = node_feature_dim + (2 * embed_dim)

        
        # Graph encoder (GNN)
        self.encoder = GraphEncoder(
            in_dim=in_dim,
            hidden=hidden,
            heads=heads,
            dropout=dropout,
        )

        
        # Pooling
        self.pooling = GraphPooling(hidden)

    
    # Forward
    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyG Data object

        Returns:
            Graph embeddings [num_graphs, hidden]
        """

        x = data.x
        edge_index = data.edge_index.to(x.device)
        edge_attr = data.edge_attr.to(x.device)

        
        # Batch handling
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                x.size(0), dtype=torch.long, device=x.device
            )

        
        # Sensor embedding
        num_nodes = x.size(0)

        sensor_ids = torch.arange(
            self.num_sensors, device=x.device
        )
        sensor_ids = sensor_ids.repeat(
            (num_nodes // self.num_sensors) + 1
        )[:num_nodes]

        sensor_emb = self.sensor_emb(sensor_ids)

        
        # Previous activity embedding
        prev_ids = data.prev_activity  # (num_graphs,)
        prev_ids_expanded = prev_ids[batch]  # (num_nodes,)

        prev_emb = self.prev_act_emb(prev_ids_expanded)

        
        # Combine features
        x = torch.cat([x, sensor_emb, prev_emb], dim=-1)

        x = F.layer_norm(x, x.shape[-1:])
        x = self.input_drop(x)

        
        # Graph encoding
        x = self.encoder(x, edge_index, edge_attr)

        
        # Pooling → graph embedding
        g = self.pooling(x, batch)

        return g

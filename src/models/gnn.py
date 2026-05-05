import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention



# Sensor Embedding
class SensorEmbedding(nn.Module):
    """Learnable embedding for each sensor."""

    def __init__(self, num_sensors: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_sensors, embed_dim)

    def forward(self, sensor_ids: torch.Tensor):
        return self.emb(sensor_ids)  # (N, embed_dim)


# Graph Encoder (GAT)
class GraphEncoder(nn.Module):
    """
    Two-layer Graph Attention Network (GAT).
    Uses edge weights for attention.
    """

    def __init__(self, in_dim: int, hidden: int, heads: int, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # First GAT layer (multi-head)
        self.gat1 = GATConv(
            in_dim,
            hidden,
            heads=heads,
            edge_dim=1
        )

        # Second GAT layer (single head)
        self.gat2 = GATConv(
            hidden * heads,
            hidden,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=1
        )

        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr):

        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.norm1(x)

        x = self.dropout(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.norm2(x)

        return x

# Graph Pooling
class GraphPooling(nn.Module):
    """
    Attention-based global pooling.
    Converts node embeddings → graph embedding.
    """

    def __init__(self, hidden: int):
        super().__init__()

        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
        )

    def forward(self, x, batch):
        return self.pool(x, batch)  # (B, hidden)


# Graph Block
class GraphBlock(nn.Module):
    """
    Full graph processing block:
    - Adds embeddings
    - Applies GNN
    - Pools to graph-level representation
    """

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        embed_dim: int,
        hidden: int,
        heads: int,
        dropout: float
    ):
        super().__init__()

        self.num_sensors = num_sensors

        # Embeddings
        self.sensor_emb = SensorEmbedding(num_sensors, embed_dim)
        self.prev_act_emb = nn.Embedding(num_classes + 1, embed_dim)

        self.input_drop = nn.Dropout(dropout)

        # Node feature size (must match window_to_graph)
        node_feature_dim = 15
        in_dim = node_feature_dim + (2 * embed_dim)

        # Graph encoder
        self.encoder = GraphEncoder(
            in_dim=in_dim,
            hidden=hidden,
            heads=heads,
            dropout=dropout
        )

        # Pooling
        self.pooling = GraphPooling(hidden)

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index.to(x.device)
        edge_attr = data.edge_attr.to(x.device)

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
        prev_ids = data.prev_activity
        prev_ids_expanded = prev_ids[batch]
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

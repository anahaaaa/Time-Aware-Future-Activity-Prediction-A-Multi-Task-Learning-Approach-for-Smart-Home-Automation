import torch
import torch.nn as nn
from torch_geometric.data import Batch


class ActivityModel(nn.Module):
    """
    Hybrid GNN + BiLSTM model for:
    - Activity classification
    - Time-to-next-activity prediction
    """

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        embed_dim: int = 8,
        hidden: int = 128,
        heads: int = 4,
        dropout: float = 0.4,
        num_time_bins: int = 10,
    ):
        super().__init__()

        self.hidden = hidden
        self.num_classes = num_classes


        # Duration prior table (per activity)    
        self.register_buffer(
            "act_duration_table",
            torch.zeros(num_classes, dtype=torch.float32)
        )

       
        # Graph encoder    
        self.graph_block = GraphBlock(
            num_sensors=num_sensors,
            num_classes=num_classes,
            embed_dim=embed_dim,
            hidden=hidden,
            heads=heads,
            dropout=dropout,
        )

       
        # Sequence model (BiLSTM)        
        self.bilstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

       
        # Activity prediction head    
        self.activity_head = nn.Linear(hidden, num_classes)

   
        # Activity embedding → time prediction       
        self.activity_time_emb = nn.Embedding(num_classes, hidden // 2)

       
        # Time prediction head
        self.per_activity_time = PerActivityTimeHead(
            input_dim=hidden + hidden // 2 + 2,
            num_classes=num_classes,
            num_outputs=num_time_bins,
        )

    
    # Duration table setup
    def set_duration_table(self, act_time_stats, default_mean):
        """
        Initialize duration priors from training statistics.
        """
        for act_id in range(self.num_classes):
            if act_id in act_time_stats:
                self.act_duration_table[act_id] = torch.tensor(
                    act_time_stats[act_id]["mean"]
                )
            else:
                self.act_duration_table[act_id] = torch.tensor(default_mean)

    
    # Dynamic time features
    def _dynamic_time_features(self, batch, B, S, activity_logits):

        pred_act_id = activity_logits.argmax(dim=1)

        typical = self.act_duration_table[pred_act_id].clamp(1.0, 120.0)

        num_nodes = batch.x.size(0) // (B * S)

        last_idx = (
            torch.arange(B, device=batch.x.device) * S * num_nodes
            + (S - 1) * num_nodes
        )

        elapsed = batch.x[last_idx, 9].clamp(0, 10)

        dynamic_progress = (elapsed / (typical + 1e-6)).clamp(0, 5)
        dynamic_remaining = (typical - elapsed).clamp(0, 120)

        return dynamic_progress, dynamic_remaining

    
    # Forward
    def forward(self, graph_sequences):

        device = next(self.parameters()).device

        B = len(graph_sequences)
        S = len(graph_sequences[0])

        # Flatten sequences → batch graphs
        all_graphs = [g for seq in graph_sequences for g in seq]
        batch = Batch.from_data_list(all_graphs).to(device)

        
        # Graph encoding
        g_emb = self.graph_block(batch)
        g_emb = g_emb.view(B, S, -1)

        
        # Temporal modeling
        lstm_out, _ = self.bilstm(g_emb)
        final = lstm_out[:, -1, :]

        
        # Activity prediction
        activity_logits = self.activity_head(final)

        
        # Activity signal (soft + hard fusion)
        probs = torch.softmax(activity_logits, dim=1)

        soft_emb = probs @ self.activity_time_emb.weight
        hard_emb = self.activity_time_emb(
            activity_logits.argmax(dim=1).detach()
        )

        act_signal = 0.5 * soft_emb + 0.5 * hard_emb

        
        # Dynamic time features
        dynamic_progress, dynamic_remaining = self._dynamic_time_features(
            batch, B, S, activity_logits
        )

        
        # Time prediction input
        time_input = torch.cat([
            final,
            act_signal,
            dynamic_progress.unsqueeze(1),
            dynamic_remaining.unsqueeze(1),
        ], dim=-1)

        pred_act = activity_logits.argmax(dim=1)

        time_pred = self.per_activity_time(time_input, pred_act)

        return activity_logits, time_pred

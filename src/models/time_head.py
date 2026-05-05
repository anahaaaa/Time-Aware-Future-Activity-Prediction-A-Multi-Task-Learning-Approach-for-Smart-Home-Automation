import torch
import torch.nn as nn


class PerActivityTimeHead(nn.Module):
    """
    Activity-conditioned time prediction head.

    For each predicted activity, a separate MLP head is used
    to predict time (or time bins).

    This allows the model to learn activity-specific temporal patterns.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_outputs: int = 10,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_outputs = num_outputs
        self.num_classes = num_classes

        # One head per activity
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(input_dim // 2, input_dim // 4),
                nn.ReLU(),

                nn.Linear(input_dim // 4, num_outputs)
            )
            for _ in range(num_classes)
        ])

    def forward(
        self,
        x: torch.Tensor,
        activity_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] features
            activity_ids: [B] predicted activity indices

        Returns:
            out: [B, num_outputs] time predictions
        """

        B = x.size(0)

        # Initialize output tensor
        out = torch.zeros(
            B,
            self.num_outputs,
            device=x.device,
            dtype=x.dtype
        )

        # Apply correct head per activity
        for act_id in activity_ids.unique():
            act_id_int = int(act_id.item())

            mask = (activity_ids == act_id)

            if mask.any():
                out[mask] = self.heads[act_id_int](x[mask])

        return out

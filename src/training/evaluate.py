import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error


@torch.no_grad()
def eval_epoch(
    model,
    dataloader,
    class_weights,
    time_bin_edges,
    device
):
    """
    Evaluation loop.

    Computes:
    - Activity classification accuracy
    - Time prediction accuracy (bin-based)
    - MAE and normalized MAE

    Args:
        model: trained model
        dataloader: validation/test loader
        class_weights: tensor
        time_bin_edges: bin boundaries
        device: torch device

    Returns:
        avg_loss, acc, preds, labels_all,
        time_preds, time_true, mae, nmae, time_acc
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    preds = []
    labels_all = []

    time_preds = []
    time_true = []

    time_correct = 0
    time_total = 0

    
    # Bin centers (for MAE computation)
    bin_centers = torch.tensor(
        [
            (time_bin_edges[i] + time_bin_edges[i + 1]) / 2.0
            for i in range(len(time_bin_edges) - 1)
        ],
        dtype=torch.float32,
        device=device
    )

    
    # Evaluation loop
    for sequences, labels, time_targets in dataloader:

        target = torch.as_tensor(labels, dtype=torch.long, device=device)
        time_target = torch.as_tensor(time_targets, dtype=torch.long, device=device)

        activity_logits, time_pred = model(sequences)

        
        # Loss
        loss_activity = F.cross_entropy(
            activity_logits,
            target,
            weight=class_weights,
            label_smoothing=0.1
        )

        loss_time = F.cross_entropy(time_pred, time_target)

        total_loss += (loss_activity + loss_time).item()
        num_batches += 1

        
        # Activity metrics
        batch_preds = activity_logits.argmax(dim=1)

        correct += (batch_preds == target).sum().item()
        total += target.size(0)

        preds.extend(batch_preds.cpu().tolist())
        labels_all.extend(target.cpu().tolist())

        
        # Time metrics
        pred_bins = time_pred.argmax(dim=1)

        time_correct += (pred_bins == time_target).sum().item()
        time_total += target.size(0)

        pred_minutes = bin_centers[pred_bins]
        true_minutes = bin_centers[time_target]

        time_preds.extend(pred_minutes.cpu().tolist())
        time_true.extend(true_minutes.cpu().tolist())

  
    # Final metrics
    pred_arr = np.array(time_preds, dtype=np.float64)
    true_arr = np.array(time_true, dtype=np.float64)

    mae = mean_absolute_error(true_arr, pred_arr)
    nmae = mae / (np.mean(true_arr) + 1e-6)

    time_acc = time_correct / max(time_total, 1)
    acc = correct / max(total, 1)
    avg_loss = total_loss / max(num_batches, 1)

    print("\n Evaluation Summary")
    print("-" * 40)
    print(f"Loss          : {avg_loss:.4f}")
    print(f"Activity Acc  : {acc:.4f}")
    print(f"Time Acc      : {time_acc:.4f}")
    print(f"MAE (min)     : {mae:.2f}")
    print(f"NMAE          : {nmae:.4f}")

    return (
        avg_loss,
        acc,
        preds,
        labels_all,
        time_preds,
        time_true,
        mae,
        nmae,
        time_acc
    )

import torch
import torch.nn.functional as F
import torch.nn as nn


def train_epoch(
    model,
    dataloader,
    optimizer,
    class_weights,
    device,
    epoch: int
):
    """
    One training epoch (2-stage training):
    1. Activity prediction
    2. Time prediction (activity-conditioned)

    Args:
        model: ActivityModel
        dataloader: DataLoader
        optimizer: optimizer
        class_weights: tensor for class imbalance
        device: torch device
        epoch: current epoch number

    Returns:
        avg_loss, accuracy
    """

    model.train()

    total_loss = 0.0
    total_act_loss = 0.0
    total_time_loss = 0.0

    correct = 0
    total = 0
    num_batches = 0

    
    # Print learning rates (debug)
    for i, pg in enumerate(optimizer.param_groups):
        print(f"[Epoch {epoch}] Group {i}: lr={pg['lr']:.2e}")

    
    # Training loop
    for batch_idx, (sequences, labels, time_targets) in enumerate(dataloader):

        target = torch.as_tensor(labels, dtype=torch.long, device=device)
        time_target = torch.as_tensor(time_targets, dtype=torch.long, device=device)

        
        # STAGE 1 - Activity prediction
        optimizer.zero_grad()

        activity_logits, _ = model(sequences)

        loss_activity = F.cross_entropy(
            activity_logits,
            target,
            weight=class_weights,
            label_smoothing=0.1
        )

        loss_activity.backward()

        nn.utils.clip_grad_norm_(
            list(model.graph_block.parameters()) +
            list(model.bilstm.parameters()) +
            list(model.activity_head.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        
        # STAGE 2 - Time prediction
        optimizer.zero_grad()

        activity_logits_frozen = activity_logits.detach()

        # Freeze graph encoder (optional but intentional)
        for p in model.graph_block.parameters():
            p.requires_grad_(False)

        time_pred = model.forward_time_only(
            sequences,
            activity_logits_frozen
        )

        loss_time = F.cross_entropy(time_pred, time_target)
        loss_time.backward()

        nn.utils.clip_grad_norm_(
            list(model.bilstm.parameters()) +
            list(model.activity_time_emb.parameters()) +
            list(model.per_activity_time.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        # Unfreeze graph encoder
        for p in model.graph_block.parameters():
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad = None

      
        # Stats
        total_act_loss += loss_activity.item()
        total_time_loss += loss_time.item()
        total_loss += loss_activity.item() + loss_time.item()

        preds = activity_logits.argmax(dim=1)

        correct += (preds == target).sum().item()
        total += target.size(0)
        num_batches += 1

        # Debug (first batch only)
        if batch_idx == 0:
            print(f"[Batch 0] time_pred bins: {time_pred[:5].argmax(dim=1).cpu().numpy()}")
            print(f"[Batch 0] time_target:    {time_target[:5].cpu().numpy()}")

    
    # Epoch summary
    avg_loss = total_loss / num_batches
    avg_act_loss = total_act_loss / num_batches
    avg_time_loss = total_time_loss / num_batches
    accuracy = correct / total

    print(
        f"\n  Epoch {epoch} Summary"
        f"\n  Activity Loss : {avg_act_loss:.4f}"
        f"\n  Time Loss     : {avg_time_loss:.4f}"
        f"\n  Total Loss    : {avg_loss:.4f}"
        f"\n  Accuracy      : {accuracy:.4f}"
    )

    return avg_loss, accuracy

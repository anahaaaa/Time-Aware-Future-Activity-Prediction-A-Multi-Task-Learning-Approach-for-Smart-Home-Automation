import torch
import numpy as np
from collections import defaultdict
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

from src.training.train import train_epoch
from src.training.evaluate import eval_epoch


def train(
    model,
    train_loader,
    val_loader,
    cfg,
    device,
    valid_class_names=None,
    time_bin_edges=None
):
    model = model.to(device)

    
    # Class weights
    train_labels = np.array([label for _, label, _ in train_loader.dataset])
    num_classes = len(np.unique(train_labels))

    class_counts = np.bincount(train_labels, minlength=num_classes)

    class_weights = np.log1p(len(train_labels) / (class_counts + 1e-6))
    class_weights = np.clip(class_weights, 0.3, 3.0)

    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    print("\nClass Distribution")
    print("Counts :", class_counts)
    print("Weights:", class_weights)

    
    # Duration table 
    if time_bin_edges is not None:

        act_groups = defaultdict(list)

       
        for _, label, bin_id in train_loader.dataset:
            act_groups[label].append(int(bin_id))

        act_time_stats = {}

        print("\n⏱️ Per-activity time stats (bin index):")

        for act_id in sorted(act_groups):
            bins = np.array(act_groups[act_id], dtype=np.float32)

            name = (
                valid_class_names[act_id]
                if valid_class_names and act_id < len(valid_class_names)
                else f"class_{act_id}"
            )

            mean_bin = float(np.mean(bins))

            act_time_stats[act_id] = {
                "mean": mean_bin  
            }

            print(f"[{act_id:2d}] {name:20s} → mean_bin={mean_bin:.2f}")

       
        model.set_duration_table(act_time_stats, mean_time=0.0)

        print("✔ Duration table set (bin-based)")

    
    # Optimizer
    optimizer = Adam([
        {"params": model.graph_block.parameters(),       "lr": cfg.LR,     "weight_decay": 1e-4},
        {"params": model.bilstm.parameters(),            "lr": cfg.LR,     "weight_decay": 1e-4},
        {"params": model.activity_head.parameters(),     "lr": cfg.LR,     "weight_decay": 1e-4},
        {"params": model.activity_time_emb.parameters(), "lr": cfg.LR * 3, "weight_decay": 1e-4},
        {"params": model.per_activity_time.parameters(), "lr": cfg.LR * 3, "weight_decay": 1e-4},
    ])

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=3,
        factor=0.5,
        min_lr=1e-5
    )

    
    # Tracking
    best_score = 0.0
    best_state = None
    patience_counter = 0
    early_stop_patience = 18

    history = defaultdict(list)
    f1_history = []

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for epoch in range(1, cfg.EPOCHS + 1):

        print(f"\n Epoch {epoch}/{cfg.EPOCHS}")

        # Train
        tr_loss, tr_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            class_weights,
            device,
            epoch
        )

        # Validation
        (
            vl_loss, vl_acc, val_preds, val_labels,
            time_preds, time_true, mae, nmae, time_acc
        ) = eval_epoch(
            model,
            val_loader,
            class_weights,
            time_bin_edges,
            device
        )

        # Metrics
        macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)

        # Smooth F1
        f1_history.append(macro_f1)
        smoothed_f1 = (
            0.7 * macro_f1 +
            0.3 * (f1_history[-2] if len(f1_history) > 1 else macro_f1)
        )

        # Combined score
        score = 0.7 * smoothed_f1 + 0.3 * time_acc

        scheduler.step(score)

        # Logging
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_f1"].append(macro_f1)

        print(
            f"Train Loss={tr_loss:.4f} | Val Loss={vl_loss:.4f} | "
            f"F1={macro_f1:.3f} | TimeAcc={time_acc:.3f} | MAE={mae:.1f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Best model tracking
        if score > best_score:
            best_score = score
            patience_counter = 0

            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

            print(" New best model")

        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{early_stop_patience})")

            if patience_counter >= early_stop_patience:
                print(" Early stopping")
                break

    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
        print("Best model restored")

    return model, history, class_weights

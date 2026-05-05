import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def show_results(
    y_true,
    y_pred,
    class_names=None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    save_path: str = None
):
    """
    Display classification report and confusion matrix.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        class_names: list of class names (optional)
        title: plot title
        normalize: whether to normalize confusion matrix
        save_path: if provided, saves the figure
    """

    
    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names if class_names else None,
            digits=3,
            zero_division=0
        )
    )

    
    # Confusion matrix
    if normalize:
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        fmt = ".2f"
        print("\n(Showing normalized confusion matrix)")
    else:
        cm = confusion_matrix(y_true, y_pred)
        fmt = "d"
        print("\n(Showing raw confusion matrix)")

    
    # Plot
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto"
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()

    
    # Save 
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✔ Confusion matrix saved to: {save_path}")

    plt.show()

# eval/metrics.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import logging

logger = logging.getLogger("Q-GUARD")

def optimize_threshold(y_true: torch.Tensor, probs: torch.Tensor) -> tuple[float, float]:
    y_true_np = y_true.detach().cpu().numpy()
    probs_np = probs[:, 1].detach().cpu().numpy()
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_true_np, (probs_np >= t).astype(int), average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)
    logger.info(f"Optimal Threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
    return best_thresh, best_f1

def plot_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, class_names: list, path: str):
    cm = confusion_matrix(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_roc_pr_curves(y_true: torch.Tensor, probs: torch.Tensor, num_classes: int, class_names: list, path_roc: str, path_pr: str):
    y_true_np = y_true.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true_np, probs_np[:, 1])
        plt.plot(fpr, tpr, label=f"Class {class_names[1]} (AUC: {roc_auc_score(y_true_np, probs_np[:, 1]):.2f})")
    else:
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_np == i, probs_np[:, i])
            plt.plot(fpr, tpr, label=f"{class_names[i]}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(path_roc)
    plt.close()

    # PR Curve
    plt.figure(figsize=(8, 6))
    if num_classes == 2:
        prec, rec, _ = precision_recall_curve(y_true_np, probs_np[:, 1])
        plt.plot(rec, prec, label=f"Class {class_names[1]}")
    else:
        for i in range(num_classes):
            prec, rec, _ = precision_recall_curve(y_true_np == i, probs_np[:, i])
            plt.plot(rec, prec, label=f"{class_names[i]}")
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(path_pr)
    plt.close()
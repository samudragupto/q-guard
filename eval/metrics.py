# eval/metrics.py
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

def robust_evaluation(y_true: torch.Tensor, probs: torch.Tensor, class_names: list) -> dict:
    y_true_np = y_true.cpu().numpy()
    preds_np = probs.argmax(dim=-1).cpu().numpy()
    probs_np = probs.cpu().numpy()
    
    report = classification_report(y_true_np, preds_np, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true_np, preds_np)
    
    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    
    try:
        if len(class_names) == 2:
            auc = roc_auc_score(y_true_np, probs_np[:, 1])
            fpr, tpr, roc_thresh = roc_curve(y_true_np, probs_np[:, 1])
            precision, recall, pr_thresh = precision_recall_curve(y_true_np, probs_np[:, 1])
            results["roc_auc"] = auc
            results["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            results["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}
        else:
            auc = roc_auc_score(y_true_np, probs_np, multi_class='ovr')
            results["roc_auc"] = auc
    except ValueError:
        results["roc_auc"] = None
        
    return results
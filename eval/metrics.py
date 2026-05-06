# eval/metrics.py
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def robust_evaluation(y_true: torch.Tensor, probs: torch.Tensor, class_names: list) -> dict:
    y_true_np = y_true.cpu().numpy()
    preds_np = probs.argmax(dim=-1).cpu().numpy()
    
    report = classification_report(y_true_np, preds_np, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true_np, preds_np)
    
    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    
    try:
        if len(class_names) == 2:
            auc = roc_auc_score(y_true_np, probs[:, 1].cpu().numpy())
        else:
            auc = roc_auc_score(y_true_np, probs.cpu().numpy(), multi_class='ovr')
        results["roc_auc"] = auc
    except ValueError:
        results["roc_auc"] = None
        
    return results
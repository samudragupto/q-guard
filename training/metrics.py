# training/metrics.py
import torch
from sklearn.metrics import f1_score, accuracy_score

def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    acc = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np, average='macro')
    
    return {"accuracy": acc, "macro_f1": f1}
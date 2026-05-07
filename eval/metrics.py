# eval/metrics.py
import torch
import numpy as np
import logging
from sklearn.metrics import f1_score, precision_recall_curve

logger = logging.getLogger(__name__)

def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Safely convert CUDA/CPU tensor to NumPy array without gradient leakage."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)

def optimize_threshold(
    y_true: torch.Tensor, 
    probs: torch.Tensor, 
    start: float = 0.1, 
    end: float = 0.9, 
    step: float = 0.01
) -> tuple[float, float]:
    """
    Find the optimal decision threshold maximizing the F1-score.
    Uses vectorized operations and handles edge cases safely.
    
    Returns:
        tuple[float, float]: (optimal_threshold, best_f1_score)
    """
    y_true_np = _to_numpy(y_true)
    probs_np = _to_numpy(probs)
    
    # Validation: Empty inputs
    if len(y_true_np) == 0 or len(probs_np) == 0:
        logger.warning("Empty inputs provided to optimize_threshold. Returning default (0.5, 0.0).")
        return 0.5, 0.0
        
    assert len(y_true_np) == len(probs_np), "y_true and probs length mismatch"
    
    # Handle binary classification (extract probability of class 1)
    if probs_np.ndim == 2 and probs_np.shape[1] == 2:
        probs_np = probs_np[:, 1]
    elif probs_np.ndim == 2:
        logger.warning("Multi-class probabilities detected. Threshold optimization skipped.")
        return 0.5, 0.0
        
    # Validation: NaN probabilities
    if np.isnan(probs_np).any():
        logger.warning("NaN probabilities detected. Replacing with 0.5.")
        probs_np = np.nan_to_num(probs_np, nan=0.5)

    # Vectorized threshold search
    thresholds = np.arange(start, end, step)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for t in thresholds:
        preds = (probs_np >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_true_np, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    # Validation: NaN threshold
    if np.isnan(best_threshold):
        logger.warning("NaN threshold detected. Defaulting to 0.5.")
        best_threshold = 0.5
            
    logger.info(f"Optimal threshold found: {best_threshold:.4f} (F1: {best_f1:.4f})")
    return best_threshold, best_f1
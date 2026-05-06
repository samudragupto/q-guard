# eval/calibration.py
import torch
import numpy as np
from torch.nn import functional as F

def expected_calibration_error(y_true: torch.Tensor, probs: torch.Tensor, n_bins: int = 10) -> float:
    confidences, predictions = probs.max(dim=-1)
    
    y_true_np = y_true.cpu().numpy()
    confidences_np = confidences.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences_np > bin_boundaries[i]) & (confidences_np <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions_np[in_bin] == y_true_np[in_bin])
            avg_confidence_in_bin = np.mean(confidences_np[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def brier_score(y_true: torch.Tensor, probs: torch.Tensor) -> float:
    y_true_one_hot = F.one_hot(y_true, num_classes=probs.shape[-1]).float()
    return torch.mean((probs - y_true_one_hot) ** 2).item()
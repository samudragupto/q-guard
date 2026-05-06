# eval/calibration.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01):
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=20)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(max_iter):
            def closure():
                optimizer.zero_grad()
                loss = criterion(logits / self.temperature, labels)
                loss.backward()
                return loss
            optimizer.step(closure)

def expected_calibration_error(y_true: torch.Tensor, probs: torch.Tensor, n_bins: int = 10) -> float:
    confidences, predictions = probs.max(dim=-1)
    y_true_np, conf_np, pred_np = y_true.cpu().numpy(), confidences.cpu().numpy(), predictions.cpu().numpy()
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (conf_np > bin_boundaries[i]) & (conf_np <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            ece += np.abs(np.mean(conf_np[in_bin]) - np.mean(pred_np[in_bin] == y_true_np[in_bin])) * prop_in_bin
    return ece

def brier_score(y_true: torch.Tensor, probs: torch.Tensor) -> float:
    y_true_one_hot = F.one_hot(y_true, num_classes=probs.shape[-1]).float()
    return torch.mean((probs - y_true_one_hot) ** 2).item()
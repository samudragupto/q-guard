# eval/calibration.py
import torch
import torch.nn as nn
import numpy as np

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(min=0.5, max=10.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=20)
        
        for _ in range(max_iter):
            def closure():
                optimizer.zero_grad()
                loss = criterion(logits / self.temperature, labels)
                loss.backward()
                return loss
            optimizer.step(closure)

    def get_calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(logits), dim=-1)

class VectorScaling(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.W = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.W + self.bias

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=20)
        
        for _ in range(max_iter):
            def closure():
                optimizer.zero_grad()
                loss = criterion(self.forward(logits), labels)
                loss.backward()
                return loss
            optimizer.step(closure)

    def get_calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(logits), dim=-1)

def expected_calibration_error(y_true: torch.Tensor, probs: torch.Tensor, n_bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=-1)
    y_true_np = y_true.detach().cpu().numpy()
    conf_np = confidences.detach().cpu().numpy()
    pred_np = predictions.detach().cpu().numpy()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (conf_np > bin_boundaries[i]) & (conf_np <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            ece += np.abs(np.mean(conf_np[in_bin]) - np.mean(pred_np[in_bin] == y_true_np[in_bin])) * prop_in_bin
    return ece

def brier_score(y_true: torch.Tensor, probs: torch.Tensor) -> float:
    import torch.nn.functional as F
    y_true_one_hot = F.one_hot(y_true, num_classes=probs.shape[-1]).float()
    return torch.mean((probs.detach().cpu() - y_true_one_hot.detach().cpu()) ** 2).item()

def reliability_diagram_data(y_true: torch.Tensor, probs: torch.Tensor, n_bins: int = 15) -> dict:
    confidences, predictions = probs.max(dim=-1)
    y_true_np = y_true.detach().cpu().numpy()
    conf_np = confidences.detach().cpu().numpy()
    pred_np = predictions.detach().cpu().numpy()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracy, bin_confidence = [], []
    
    for i in range(n_bins):
        in_bin = (conf_np > bin_boundaries[i]) & (conf_np <= bin_boundaries[i+1])
        bin_accuracy.append(np.mean(pred_np[in_bin] == y_true_np[in_bin]) if in_bin.sum() > 0 else 0.0)
        bin_confidence.append(np.mean(conf_np[in_bin]) if in_bin.sum() > 0 else 0.0)
        
    return {"bin_accuracy": np.array(bin_accuracy), "bin_confidence": np.array(bin_confidence)}
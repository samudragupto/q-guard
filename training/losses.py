# training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, label_smoothing: float = 0.05, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            ce_loss = alpha_t * ce_loss
            
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def get_criterion(use_focal: bool, class_weights: torch.Tensor = None, device: torch.device = torch.device("cpu")) -> nn.Module:
    weights = class_weights.to(device) if class_weights is not None else None
    if use_focal:
        return FocalLoss(alpha=weights, gamma=2.0, label_smoothing=0.05).to(device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05).to(device)
# eval/adversarial.py
import torch
import torch.nn as nn
from utils import move_to_device

def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module, device: torch.device, epsilon: float = 0.05) -> torch.Tensor:
    x_adv = x.clone().detach().requires_grad_(True)
    loss = criterion(model(x_adv)["logits"], y)
    loss.backward()
    return torch.clamp(x_adv + epsilon * x_adv.grad.sign(), 0, 1).detach()

def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module, device: torch.device, eps: float = 0.05, alpha: float = 0.01, steps: int = 3) -> torch.Tensor:
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = criterion(model(x_adv)["logits"], y)
        loss.backward()
        perturbation = alpha * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv + perturbation, min=x - eps, max=x + eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv

def evaluate_robustness(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device, criterion: nn.Module, eps: float = 0.1, num_classes: int = 2, threshold: float = 0.5) -> dict:
    model.eval()
    X_dev, y_dev = move_to_device(X, device), move_to_device(y, device)
    
    with torch.no_grad():
        clean_out = model(X_dev)
        if num_classes == 2:
            clean_preds = (clean_out["probs"][:, 1] > threshold).long()
        else:
            clean_preds = clean_out["logits"].argmax(dim=-1)
        clean_acc = (clean_preds == y_dev).float().mean().item()
        
    X_adv = pgd_attack(model, X_dev, y_dev, criterion, device, eps=eps, alpha=eps/4, steps=5)
    with torch.no_grad():
        adv_out = model(X_adv)
        if num_classes == 2:
            adv_preds = (adv_out["probs"][:, 1] > threshold).long()
        else:
            adv_preds = adv_out["logits"].argmax(dim=-1)
        adv_acc = (adv_preds == y_dev).float().mean().item()
        
    return {"clean_accuracy": clean_acc, "adversarial_accuracy": adv_acc, "accuracy_drop": clean_acc - adv_acc}
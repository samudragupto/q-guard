# eval/adversarial.py
import torch
import torch.nn.functional as F
from utils import move_to_device

def fgsm_attack(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, epsilon: float = 0.1) -> torch.Tensor:
    x_adv = move_to_device(x.clone().detach(), device).requires_grad_(True)
    output = model(x_adv)
    loss = F.cross_entropy(output["logits"], move_to_device(y, device))
    loss.backward()
    return torch.clamp(x_adv + epsilon * x_adv.grad.sign(), 0, 1).detach()

def evaluate_robustness(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device, eps: float = 0.1) -> dict:
    model.eval()
    X_dev, y_dev = move_to_device(X, device), move_to_device(y, device)
    
    with torch.no_grad():
        clean_out = model(X_dev)
        clean_preds = clean_out["logits"].argmax(dim=-1)
        clean_acc = (clean_preds == y_dev).float().mean().item()
        
    X_adv = fgsm_attack(model, X, y, device, eps)
    with torch.no_grad():
        adv_out = model(X_adv)
        adv_preds = adv_out["logits"].argmax(dim=-1)
        adv_acc = (adv_preds == y_dev).float().mean().item()
        
    return {"clean_accuracy": clean_acc, "adversarial_accuracy": adv_acc, "accuracy_drop": clean_acc - adv_acc}
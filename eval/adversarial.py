# eval/adversarial.py
import torch
import torch.nn.functional as F
from training.metrics import calculate_metrics
from utils import move_to_device

def pgd_attack(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, 
               device: torch.device, eps: float = 0.1, alpha: float = 0.01, steps: int = 10) -> torch.Tensor:
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad_(True)
    
    for _ in range(steps):
        output = model(x_adv)
        loss = F.cross_entropy(output["logits"], y)
        loss.backward()
        
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x, -eps, eps)
            x_adv = torch.clamp(x + delta, 0, 1)
            
        x_adv.requires_grad_(True)
        
    return x_adv.detach()

def evaluate_robustness(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, 
                        device: torch.device, eps: float = 0.1) -> dict:
    model.eval()
    X_dev = move_to_device(X, device)
    y_dev = move_to_device(y, device)
    
    with torch.no_grad():
        clean_out = model(X_dev)
        clean_preds = clean_out["logits"].argmax(dim=-1)
        clean_metrics = calculate_metrics(y_dev, clean_preds)
        
    X_adv = pgd_attack(model, X_dev, y_dev, device, eps=eps)
    with torch.no_grad():
        adv_out = model(X_adv)
        adv_preds = adv_out["logits"].argmax(dim=-1)
        adv_metrics = calculate_metrics(y_dev, adv_preds)
        
    return {
        "clean_accuracy": clean_metrics["accuracy"],
        "clean_f1": clean_metrics["macro_f1"],
        "adversarial_accuracy": adv_metrics["accuracy"],
        "adversarial_f1": adv_metrics["macro_f1"],
        "accuracy_drop": clean_metrics["accuracy"] - adv_metrics["accuracy"]
    }
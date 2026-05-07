import torch
import torch.nn as nn

def pgd_attack(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module, eps: float = 0.03, alpha: float = 0.007, iters: int = 3) -> torch.Tensor:
    model.eval() # Ensure batchnorm/dropout are stable during attack generation
    X_adv = X.clone().detach()
    
    for _ in range(iters):
        X_adv.requires_grad_(True)
        outputs = model(X_adv)
        loss = criterion(outputs["logits"], y)
        loss.backward()
        
        # Use data.grad to avoid graph retention issues
        X_adv = X_adv + alpha * X_adv.grad.sign()
        # Projection step
        X_adv = torch.max(torch.min(X_adv, X + eps), X - eps)
        X_adv = X_adv.detach()
        
    model.train() # Reset to train mode for continued training
    return X_adv

def evaluate_robustness(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device, criterion: nn.Module, threshold: float = 0.5, eps: float = 0.1) -> dict:
    model.eval()
    X_dev, y_dev = X.to(device), y.to(device)
    
    with torch.no_grad():
        clean_probs = model(X_dev)["probs"]
        clean_preds = (clean_probs[:, 1] >= threshold).long()
        clean_acc = (clean_preds == y_dev).float().mean().item()
        
    # Generate PGD attack using criterion to maintain graph properly
    X_adv = pgd_attack(model, X_dev, y_dev, criterion, eps=eps, alpha=eps/4, iters=5)
    
    with torch.no_grad():
        adv_probs = model(X_adv)["probs"]
        adv_preds = (adv_probs[:, 1] >= threshold).long()
        adv_acc = (adv_preds == y_dev).float().mean().item()
        
    return {"clean_accuracy": clean_acc, "adversarial_accuracy": adv_acc}
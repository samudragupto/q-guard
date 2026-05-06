# eval/adversarial.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_attack(model: nn.Module, X: torch.Tensor, y: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    X_adv = X.clone().detach().requires_grad_(True)

    outputs = model(X_adv)
    loss = F.cross_entropy(outputs["logits"], y)

    model.zero_grad()
    loss.backward()

    X_adv = X_adv + eps * X_adv.grad.sign()
    return X_adv.detach()


def pgd_attack(model: nn.Module, X: torch.Tensor, y: torch.Tensor, eps: float = 0.1, alpha: float = 0.01, iters: int = 10) -> torch.Tensor:
    X_adv = X.clone().detach()

    for _ in range(iters):
        X_adv.requires_grad_(True)

        outputs = model(X_adv)
        loss = F.cross_entropy(outputs["logits"], y)

        model.zero_grad()
        loss.backward()

        X_adv = X_adv + alpha * X_adv.grad.sign()
        X_adv = torch.max(torch.min(X_adv, X + eps), X - eps)
        X_adv = X_adv.detach()

    return X_adv


def evaluate_robustness(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device, threshold: float = 0.5, eps: float = 0.1) -> dict:
    model.eval()
    X_dev = X.to(device)
    y_dev = y.to(device)

    with torch.no_grad():
        clean_probs = model(X_dev)["probs"]
        clean_preds = (clean_probs[:, 1] >= threshold).long()
        clean_acc = (clean_preds == y_dev).float().mean().item()

    X_adv = pgd_attack(model, X_dev, y_dev, eps=eps, alpha=eps / 4, iters=5)
    with torch.no_grad():
        adv_probs = model(X_adv)["probs"]
        adv_preds = (adv_probs[:, 1] >= threshold).long()
        adv_acc = (adv_preds == y_dev).float().mean().item()

    return {"clean_accuracy": clean_acc, "adversarial_accuracy": adv_acc, "accuracy_drop": clean_acc - adv_acc}
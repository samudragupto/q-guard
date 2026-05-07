# training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from eval.metrics import optimize_threshold
import logging

logger = logging.getLogger(__name__)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cuda",
    class_weights: torch.Tensor = None,
    patience: int = 7,
    adv_training: bool = True,
    eps: float = 0.03
) -> tuple[nn.Module, float, float]:
    """
    Production-grade training loop with Mixed Precision, gradient clipping,
    early stopping, and adversarial training.
    
    Returns:
        tuple: (trained_model, optimal_threshold, best_f1_score)
    """
    assert model is not None, "Model cannot be None"
    assert train_loader is not None, "train_loader cannot be None"
    assert val_loader is not None, "val_loader cannot be None"
    
    # GPU Safety: Fallback to CPU
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Mixed Precision Setup
    scaler = GradScaler('cuda')
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda', enabled=device.type == 'cuda'):
                outputs = model(xb)
                loss = criterion(outputs["logits"], yb)
                
                # Simple adversarial training (FGSM) inside loop if enabled
                if adv_training:
                    xb_adv = xb + eps * torch.randn_like(xb).sign()  # Randomized FGSM approximation
                    xb_adv = torch.clamp(xb_adv, 0, 1)
                    adv_outputs = model(xb_adv)
                    adv_loss = criterion(adv_outputs["logits"], yb)
                    loss = 0.5 * loss + 0.5 * adv_loss
            
            scaler.scale(loss).backward()
            
            # Gradient Clipping to prevent exploding loss
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast('cuda', enabled=device.type == 'cuda'):
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                outputs = model(xb)
                loss = criterion(outputs["logits"], yb)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"[EPOCH {epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "qguard_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    # Load best model
    model.load_state_dict(torch.load("qguard_best.pth", weights_only=True))
    
    # Compute Optimal Threshold (Unpacking the tuple correctly)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            probs = model(xb)["probs"]
            all_probs.append(probs.cpu())
            all_labels.append(yb.cpu())
            
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    optimal_threshold, best_f1 = optimize_threshold(all_labels, all_probs)
    
    return model, optimal_threshold, best_f1
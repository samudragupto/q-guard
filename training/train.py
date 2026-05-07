# training/train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from models.qguard_model import QGuardModel
from training.losses import FocalLoss
from eval.adversarial import pgd_attack
from eval.metrics import optimize_threshold
import logging

logger = logging.getLogger("Q-GUARD")

def train_model(
    model: QGuardModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    config,
    class_weights: torch.Tensor = None
) -> tuple[QGuardModel, float, float]:
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=class_weights.to(device) if class_weights is not None else None, gamma=2.0, label_smoothing=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler('cuda')
    
    best_val_loss, patience_counter = float('inf'), 0
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda', enabled=device.type == 'cuda'):
                clean_out = model(xb)
                clean_loss = criterion(clean_out["logits"], yb)
                
                if config.adv_training:
                    # Adversarial Defense: PGD Generation
                    xb_adv = pgd_attack(model, xb, yb, criterion, device, eps=config.eps_adv, alpha=config.eps_adv/4, iters=2)
                    adv_out = model(xb_adv)
                    adv_loss = criterion(adv_out["logits"], yb)
                    loss = 0.5 * clean_loss + 0.5 * adv_loss
                else:
                    loss = clean_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast('cuda', enabled=device.type == 'cuda'):
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                val_loss += criterion(model(xb)["logits"], yb).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        logger.info(f"[EPOCH {epoch+1}/{config.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "qguard_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    model.load_state_dict(torch.load("qguard_best.pth", weights_only=True))
    
    # Threshold Optimization (Binary only)
    optimal_threshold, best_f1 = 0.5, 0.0
    if num_classes == 2:
        all_probs, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                all_probs.append(model(xb.to(device, non_blocking=True))["probs"].cpu())
                all_labels.append(yb)
        optimal_threshold, best_f1 = optimize_threshold(torch.cat(all_labels), torch.cat(all_probs))
    
    return model, optimal_threshold, best_f1
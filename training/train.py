# training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.qguard_model import QGuardModel
from training.losses import FocalLoss
from utils import move_to_device
from sklearn.metrics import f1_score, accuracy_score

def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    return {"accuracy": accuracy_score(y_true_np, y_pred_np), "macro_f1": f1_score(y_true_np, y_pred_np, average='macro')}

def train_model(X_train: torch.Tensor, y_train: torch.Tensor, 
                X_val: torch.Tensor, y_val: torch.Tensor, 
                epochs: int = 10, batch_size: int = 64, lr: float = 1e-3, 
                patience: int = 5, device: torch.device = torch.device("cpu"),
                class_weights: torch.Tensor = None, adv_training: bool = False) -> QGuardModel:
    
    print(f"[INFO] Training started on device: {device}", flush=True)
    print(f"[DEBUG] Initializing model with input_dim={X_train.shape[1]}", flush=True)
    
    model = QGuardModel(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(device)
    
    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (xb, yb) in enumerate(train_dl):
            xb, yb = move_to_device(xb, device), move_to_device(yb, device)
            optimizer.zero_grad()
            
            # Forward
            output = model(xb)
            loss = criterion(output["logits"], yb)
            
            # Optional Adversarial Training Step
            if adv_training:
                xb_adv = fgsm_train_attack(model, xb, yb, criterion, epsilon=0.05)
                adv_output = model(xb_adv)
                loss = (loss + criterion(adv_output["logits"], yb)) / 2.0
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if batch_idx == 0:
                print(f"  [BATCH 0] Loss: {loss.item():.4f}", flush=True)
                
        avg_epoch_loss = epoch_loss / len(train_dl)
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_dev, y_val_dev = move_to_device(X_val, device), move_to_device(y_val, device)
            val_output = model(X_val_dev)
            val_loss = criterion(val_output["logits"], y_val_dev).item()
            val_preds = val_output["logits"].argmax(dim=-1)
            metrics = calculate_metrics(y_val_dev, val_preds)
            
        print(f"[EPOCH {epoch+1}/{epochs}] Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val F1: {metrics['macro_f1']:.4f}", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_tmp.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[DEBUG] Early stopping triggered at epoch {epoch+1}", flush=True)
                break
                
    model.load_state_dict(torch.load("best_model_tmp.pth", weights_only=True))
    return model

def fgsm_train_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module, epsilon: float) -> torch.Tensor:
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = criterion(output["logits"], y)
    loss.backward()
    return torch.clamp(x_adv + epsilon * x_adv.grad.sign(), 0, 1).detach()
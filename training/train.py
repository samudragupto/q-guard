# training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.qguard_model import QGuardModel
from training.metrics import calculate_metrics
from utils import move_to_device

def train_model(X_train: torch.Tensor, y_train: torch.Tensor, 
                X_val: torch.Tensor, y_val: torch.Tensor, 
                epochs: int = 10, batch_size: int = 32, lr: float = 1e-3, 
                patience: int = 3, device: torch.device = torch.device("cpu")) -> QGuardModel:
    
    print(f"[DEBUG] Initializing model with input_dim={X_train.shape[1]}")
    model = QGuardModel(input_dim=X_train.shape[1]).to(device)
    
    classical_params = list(model.embedding.parameters()) + list(model.complexity_gate.parameters())
    quantum_params = list(model.q_attention.parameters()) + list(model.q_classifier.parameters())
    
    optimizer = optim.AdamW([
        {"params": classical_params, "lr": lr},
        {"params": quantum_params, "lr": lr * 0.5}
    ], weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (xb, yb) in enumerate(train_dl):
            xb = move_to_device(xb, device)
            yb = move_to_device(yb, device)
            
            optimizer.zero_grad()
            
            output = model(xb)
            loss = criterion(output["logits"], yb)
            
            depth_reg = output["depth"].mean() / model.complexity_gate.max_depth
            loss = loss + 0.1 * depth_reg
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_epoch_loss = epoch_loss / len(train_dl)
        
        model.eval()
        with torch.no_grad():
            X_val_dev = move_to_device(X_val, device)
            y_val_dev = move_to_device(y_val, device)
            
            val_output = model(X_val_dev)
            val_loss = criterion(val_output["logits"], y_val_dev).item()
            val_preds = val_output["logits"].argmax(dim=-1)
            metrics = calculate_metrics(y_val_dev, val_preds)
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val F1: {metrics['macro_f1']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[DEBUG] Early stopping triggered at epoch {epoch+1}")
                break
                
    return model
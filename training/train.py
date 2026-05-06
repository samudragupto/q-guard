# training/train.py
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.qguard_model import QGuardModel
from training.losses import get_criterion
from utils import move_to_device
from sklearn.metrics import f1_score, accuracy_score

try:
    from eval.adversarial import fgsm_attack, pgd_attack
except ImportError:
    print("[WARNING] Adversarial attacks not available. Skipping robustness.", flush=True)
    fgsm_attack = None
    pgd_attack = None


def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    return {
        "accuracy": accuracy_score(y_true_np, y_pred_np),
        "f1": f1_score(y_true_np, y_pred_np, average="binary")
    }


def optimize_threshold(y_true: torch.Tensor, probs: torch.Tensor) -> float:
    y_true_np = y_true.cpu().numpy()
    probs_np = probs[:, 1].cpu().numpy()

    best_thresh = 0.5
    best_f1 = 0.0

    for t in np.linspace(0.1, 0.9, 50):
        preds = (probs_np >= t).astype(int)
        f1 = f1_score(y_true_np, preds, average="binary")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh


def train_model(X_train: torch.Tensor, y_train: torch.Tensor,
                X_val: torch.Tensor, y_val: torch.Tensor,
                num_classes: int, epochs: int = 30, batch_size: int = 64, lr: float = 1e-3,
                patience: int = 7, device: torch.device = torch.device("cpu"),
                class_weights: torch.Tensor = None, use_focal: bool = True,
                adv_training: bool = True) -> tuple[QGuardModel, float]:

    print(f"[INFO] Training started on device: {device}", flush=True)
    model = QGuardModel(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = get_criterion(use_focal, class_weights, device, label_smoothing=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    best_val_f1 = 0.0
    patience_counter = 0
    optimal_threshold = 0.5

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_dl:
            xb, yb = move_to_device(xb, device), move_to_device(yb, device)
            optimizer.zero_grad()

            if adv_training and pgd_attack is not None:
                xb_adv = pgd_attack(model, xb, yb, eps=0.05, alpha=0.01, iters=3)
                x_combined = torch.cat([xb, xb_adv], dim=0)
                y_combined = torch.cat([yb, yb], dim=0)
                output = model(x_combined)
                loss = criterion(output["logits"], y_combined)
            elif adv_training and fgsm_attack is not None:
                xb_adv = fgsm_attack(model, xb, yb, eps=0.05)
                x_combined = torch.cat([xb, xb_adv], dim=0)
                y_combined = torch.cat([yb, yb], dim=0)
                output = model(x_combined)
                loss = criterion(output["logits"], y_combined)
            else:
                output = model(xb)
                loss = criterion(output["logits"], yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_dl)
        current_lr = optimizer.param_groups[0]['lr']

        model.eval()
        with torch.no_grad():
            X_val_dev, y_val_dev = move_to_device(X_val, device), move_to_device(y_val, device)
            val_output = model(X_val_dev)
            val_loss = criterion(val_output["logits"], y_val_dev).item()

            optimal_threshold = optimize_threshold(y_val_dev, val_output["probs"])
            val_preds = (val_output["probs"][:, 1] >= optimal_threshold).long()
            metrics = calculate_metrics(y_val_dev, val_preds)

        scheduler.step(metrics["f1"])

        print(f"[EPOCH {epoch+1}/{epochs}]", flush=True)
        print(f"  Train Loss: {avg_epoch_loss:.4f}", flush=True)
        print(f"  Val Loss:   {val_loss:.4f}", flush=True)
        print(f"  Val Acc:    {metrics['accuracy']:.4f}", flush=True)
        print(f"  Val F1:     {metrics['f1']:.4f} (Threshold: {optimal_threshold:.2f})", flush=True)
        print(f"  LR:         {current_lr:.6f}", flush=True)

        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), "qguard_best.pth")
            print("  [CHECKPOINT] Best model saved.", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[DEBUG] Early stopping triggered at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(torch.load("qguard_best.pth", weights_only=True))
    return model, optimal_threshold
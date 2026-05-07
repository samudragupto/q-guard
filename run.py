# run.py
import torch
import random
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

from data.data_loader import load_data, split_data
from data.preprocess import clean_data, normalize_features, encode_target
from data.feature_engineering import engineer_features, select_top_features
from data.rebalance import rebalance_dataset
from models.qguard_model import QGuardModel
from training.train import train_model

# Production Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    force=True
)
logger = logging.getLogger("Q-Guard")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True  # GPU Throughput Optimization

def main():
    set_seed(42)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Loading data...")
    df = load_data("data.csv", sample_size=100000)
    df = clean_data(df)
    df = engineer_features(df)
    
    target_col = next((c for c in ['Attack Type', 'Label', 'Attack_Type'] if c in df.columns), df.columns[-1])
    df, le = encode_target(df, target_col)
    num_classes = len(le.classes_)
    
    top_features = select_top_features(df, target_col, n_features=8)
    df, scaler = normalize_features(df, top_features)
    df = rebalance_dataset(df, target_col, ratio=5)
    
    train_df, val_df, test_df = split_data(df, target_col)
    
    X_train = torch.tensor(train_df[top_features].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[target_col].values, dtype=torch.long)
    X_val = torch.tensor(val_df[top_features].values, dtype=torch.float32)
    y_val = torch.tensor(val_df[target_col].values, dtype=torch.long)

    # GPU-Optimized DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=256, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=256, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # Class Weights
    classes = np.unique(y_train.numpy())
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train.numpy())
    class_weights = torch.tensor(np.clip(weights, 0.5, 5.0), dtype=torch.float32)

    # Model
    model = QGuardModel(input_dim=len(top_features), num_classes=num_classes)

    # Train (Unpacking 3 return values)
    logger.info("Starting training...")
    model, optimal_threshold, best_f1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        lr=1e-3,
        device=device_str,
        class_weights=class_weights,
        patience=7,
        adv_training=True,
        eps=0.03
    )

    # Fixed Logging: threshold and f1 are now separate floats
    logger.info(f"Training complete. Optimal Threshold: {optimal_threshold:.4f} | Best F1: {best_f1:.4f}")
    
    # Save Artifacts
    torch.save(model.state_dict(), "qguard_best.pth")
    torch.save(le.classes_, "label_classes.pth")
    torch.save(top_features, "feature_names.pth")
    torch.save(optimal_threshold, "optimal_threshold.pth")
    logger.info("Artifacts saved successfully.")

if __name__ == "__main__":
    main()
# run.py
import torch
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from data.data_loader import load_data, split_data
from data.preprocess import clean_data, normalize_features
from data.feature_engineering import engineer_features, select_top_features
from data.rebalance import rebalance_dataset
from training.train import train_model
from eval.calibration import expected_calibration_error, brier_score, TemperatureScaling
from eval.adversarial import evaluate_robustness
from utils import get_device, move_to_device

def main():
    device = get_device()

    print("Loading data...", flush=True)
    df = load_data("data.csv", sample_size=50000)
    
    print("Cleaning data...", flush=True)
    df = clean_data(df)
    
    print("Engineering features...", flush=True)
    df = engineer_features(df)
    
    possible_targets = ['Attack Type', 'Label', 'Attack_Type']
    target_col = next((col for col in possible_targets if col in df.columns), df.columns[-1])
    print(f"[DEBUG] Target column found: '{target_col}'", flush=True)
    
    BENIGN_LABELS = {'BENIGN', 'NORMAL', 'NORMAL TRAFFIC'}
    normalized_col = df[target_col].astype(str).str.strip().str.upper()
    print(f"[DEBUG] Unique values BEFORE encoding: {normalized_col.unique()[:10]}", flush=True)

    if pd.api.types.is_numeric_dtype(df[target_col]) and set(df[target_col].unique()) <= {0, 1}:
        pass
    elif pd.api.types.is_numeric_dtype(df[target_col]):
        df[target_col] = df[target_col].apply(lambda x: 0 if x == 0 else 1)
    else:
        df[target_col] = normalized_col.apply(lambda x: 0 if x in BENIGN_LABELS else 1)

    print(f"[DEBUG] Unique values AFTER encoding: {df[target_col].unique()}", flush=True)
    print(f"[DEBUG] Class distribution:\n{df[target_col].value_counts()}", flush=True)

    if df[target_col].nunique() < 2:
        raise ValueError("Dataset must contain both benign and attack samples")
    
    print("\nRebalancing dataset...", flush=True)
    df = rebalance_dataset(df, target_col, ratio=5)
    
    print("Selecting top 8 features...", flush=True)
    top_features = select_top_features(df, target_col, n_features=8)
    df, scaler = normalize_features(df, top_features)
    
    train_df, val_df, test_df = split_data(df, target_col)
    
    X_train = move_to_device(torch.tensor(train_df[top_features].values, dtype=torch.float32), device)
    y_train = move_to_device(torch.tensor(train_df[target_col].values, dtype=torch.long), device)
    X_val = move_to_device(torch.tensor(val_df[top_features].values, dtype=torch.float32), device)
    y_val = move_to_device(torch.tensor(val_df[target_col].values, dtype=torch.long), device)

    # Compute Class Weights
    classes = np.unique(y_train.cpu().numpy())
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train.cpu().numpy())
    class_weights = move_to_device(torch.tensor(weights, dtype=torch.float32), device)
    print(f"[DEBUG] Class weights: {class_weights}", flush=True)

    print("\nStarting training...", flush=True)
    model = train_model(
        X_train, y_train, 
        X_val, y_val, 
        epochs=15,
        batch_size=64,  
        lr=1e-3,
        patience=5,
        device=device,
        class_weights=class_weights,
        adv_training=True
    )

    print("\nCalibrating model...", flush=True)
    model.eval()
    with torch.no_grad():
        logits = model(X_val)["logits"]
        
    temp_scaler = TemperatureScaling().to(device)
    temp_scaler.fit(logits, y_val)
    
    with torch.no_grad():
        calibrated_probs = torch.softmax(temp_scaler(logits), dim=-1)
        ece = expected_calibration_error(y_val, calibrated_probs)
        brier = brier_score(y_val, calibrated_probs)
    print(f"Calibration - ECE: {ece:.4f}, Brier Score: {brier:.4f}", flush=True)
    
    print("\nRunning adversarial robustness test...", flush=True)
    robustness_results = evaluate_robustness(model, X_val[:200], y_val[:200], device, eps=0.1)
    print(f"Robustness - Clean Acc: {robustness_results['clean_accuracy']:.4f}, Adv Acc: {robustness_results['adversarial_accuracy']:.4f}", flush=True)
    
    torch.save(model.state_dict(), "qguard_weights.pth")
    print("\nModel saved to qguard_weights.pth", flush=True)

if __name__ == "__main__":
    main()
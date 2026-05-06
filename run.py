# run.py
import torch
import pandas as pd
from data.data_loader import load_data, split_data
from data.preprocess import clean_data, normalize_features
from data.feature_engineering import engineer_features, select_top_features
from training.train import train_model
from eval.calibration import expected_calibration_error, brier_score
from eval.adversarial import evaluate_robustness
from utils import get_device, move_to_device

def main():
    # Initialize device
    device = get_device()

    # PHASE 1: Data Pipeline
    print("Loading data...")
    df = load_data("data.csv", sample_size=50000)
    
    print("Cleaning data...")
    df = clean_data(df)
    
    print("Engineering features...")
    df = engineer_features(df)
    
    # Dynamic target column detection
    possible_targets = ['Attack Type', 'Label', 'Attack_Type', 'Label ']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[-1]
        print(f"[DEBUG] Standard target columns not found. Assuming last column '{target_col}' is target.")
    else:
        print(f"[DEBUG] Target column found: '{target_col}'")
    
    # Robust label normalization and encoding
    BENIGN_LABELS = {'BENIGN', 'NORMAL', 'NORMAL TRAFFIC'}

    # Print unique values BEFORE encoding (normalized for display)
    normalized_col = df[target_col].astype(str).str.strip().str.upper()
    print(f"[DEBUG] Unique values BEFORE encoding: {normalized_col.unique()[:10]}")

    # Convert to binary target (0: Benign, 1: Attack)
    if pd.api.types.is_numeric_dtype(df[target_col]):
        if set(df[target_col].unique()) <= {0, 1}:
            pass  # Already correctly encoded as 0 and 1
        else:
            df[target_col] = df[target_col].apply(lambda x: 0 if x == 0 else 1)
    else:
        df[target_col] = normalized_col.apply(lambda x: 0 if x in BENIGN_LABELS else 1)

    # Print unique values AFTER encoding and class distribution
    print(f"[DEBUG] Unique values AFTER encoding: {df[target_col].unique()}")
    print(f"[DEBUG] Class distribution:\n{df[target_col].value_counts()}")

    # Safety check for single class dataset
    if df[target_col].nunique() < 2:
        raise ValueError("Dataset must contain both benign and attack samples")
    
    print("Selecting top 8 features...")
    top_features = select_top_features(df, target_col, n_features=8)
    
    df, scaler = normalize_features(df, top_features)
    
    train_df, val_df, test_df = split_data(df, target_col)
    
    # Move tensors to device
    X_train = move_to_device(torch.tensor(train_df[top_features].values, dtype=torch.float32), device)
    y_train = move_to_device(torch.tensor(train_df[target_col].values, dtype=torch.long), device)
    
    X_val = move_to_device(torch.tensor(val_df[top_features].values, dtype=torch.float32), device)
    y_val = move_to_device(torch.tensor(val_df[target_col].values, dtype=torch.long), device)

    # PHASE 6: Training Pipeline
    print("\nStarting training...")
    model = train_model(
        X_train, y_train, 
        X_val, y_val, 
        epochs=10,
        batch_size=32,  
        lr=1e-3,
        patience=3,
        device=device
    )

    # PHASE 7: Evaluation + Robustness
    print("\nEvaluating model...")
    model.eval()
    
    with torch.no_grad():
        test_out = model(X_val)
        ece = expected_calibration_error(y_val, test_out["probs"])
        brier = brier_score(y_val, test_out["probs"])
        
    print(f"Calibration - ECE: {ece:.4f}, Brier Score: {brier:.4f}")
    
    print("\nRunning adversarial robustness test on 100 samples...")
    robustness_results = evaluate_robustness(model, X_val[:100], y_val[:100], eps=0.1, device=device)
    print(f"Robustness - Clean Acc: {robustness_results['clean_accuracy']:.4f}, Adv Acc: {robustness_results['adversarial_accuracy']:.4f}")
    
    # Save model for API
    torch.save(model.state_dict(), "qguard_weights.pth")
    print("\nModel saved to qguard_weights.pth")

if __name__ == "__main__":
    main()
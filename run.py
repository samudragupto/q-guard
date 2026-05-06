# run.py
import torch
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from data.data_loader import load_data, split_data
from data.preprocess import clean_data, normalize_features, encode_target
from data.feature_engineering import engineer_features, select_top_features
from data.rebalance import rebalance_dataset
from training.train import train_model
from training.losses import get_criterion
from eval.calibration import expected_calibration_error, brier_score, TemperatureScaling
from eval.adversarial import evaluate_robustness
from eval.metrics import robust_evaluation
from utils import get_device, move_to_device

def main():
    device = get_device()

    print("Loading data...", flush=True)
    df = load_data("data.csv", sample_size=100000)
    
    print("Cleaning data...", flush=True)
    df = clean_data(df)
    
    print("Engineering features...", flush=True)
    df = engineer_features(df)
    
    possible_targets = ['Attack Type', 'Label', 'Attack_Type']
    target_col = next((col for col in possible_targets if col in df.columns), df.columns[-1])
    print(f"[DEBUG] Target column found: '{target_col}'", flush=True)
    
    print("Encoding target labels...", flush=True)
    df, le = encode_target(df, target_col)
    num_classes = len(le.classes_)
    
    print("Selecting top features...", flush=True)
    top_features = select_top_features(df, target_col, n_features=8)
    df, scaler = normalize_features(df, top_features)
    
    print("\nRebalancing dataset...", flush=True)
    df = rebalance_dataset(df, target_col, ratio=5)
    
    train_df, val_df, test_df = split_data(df, target_col)
    
    X_train = move_to_device(torch.tensor(train_df[top_features].values, dtype=torch.float32), device)
    y_train = move_to_device(torch.tensor(train_df[target_col].values, dtype=torch.long), device)
    X_val = move_to_device(torch.tensor(val_df[top_features].values, dtype=torch.float32), device)
    y_val = move_to_device(torch.tensor(val_df[target_col].values, dtype=torch.long), device)
    X_test = move_to_device(torch.tensor(test_df[top_features].values, dtype=torch.float32), device)
    y_test = move_to_device(torch.tensor(test_df[target_col].values, dtype=torch.long), device)

    classes = np.unique(y_train.cpu().numpy())
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train.cpu().numpy())
    class_weights = move_to_device(torch.tensor(weights, dtype=torch.float32), device)

    print("\nStarting training...", flush=True)
    model, optimal_threshold = train_model(
        X_train, y_train, 
        X_val, y_val, 
        num_classes=num_classes,
        epochs=30,
        batch_size=64,  
        lr=1e-3,
        patience=7,
        device=device,
        class_weights=class_weights,
        use_focal=True,
        adv_training=True,
        adv_method="pgd"
    )

    print("\nCalibrating model...", flush=True)
    model.eval()
    criterion = get_criterion(True, class_weights, device)
    
    with torch.no_grad():
        logits = model(X_val)["logits"]
        
    temp_scaler = TemperatureScaling().to(device)
    temp_scaler.fit(logits, y_val)
    print(f"[DEBUG] Optimal Temperature: {temp_scaler.temperature.item():.4f}", flush=True)
    
    with torch.no_grad():
        calibrated_probs = temp_scaler.get_calibrated_probs(logits)
        ece = expected_calibration_error(y_val, calibrated_probs)
        brier = brier_score(y_val, calibrated_probs)
    print(f"Calibration - ECE: {ece:.4f}, Brier Score: {brier:.4f}", flush=True)
    
    print("\nRunning adversarial robustness test (PGD)...", flush=True)
    robustness_results = evaluate_robustness(model, X_val[:200], y_val[:200], device, criterion, eps=0.1, num_classes=num_classes, threshold=optimal_threshold)
    print(f"Robustness - Clean Acc: {robustness_results['clean_accuracy']:.4f}, Adv Acc: {robustness_results['adversarial_accuracy']:.4f}", flush=True)

    print("\nRunning robust evaluation on test set...", flush=True)
    with torch.no_grad():
        test_probs = model(X_test)["probs"]
    eval_results = robust_evaluation(y_test, test_probs, le.classes_)
    print(f"ROC-AUC: {eval_results['roc_auc']}", flush=True)
    print(f"Classification Report:\n{eval_results['classification_report']}", flush=True)
    
    torch.save(model.state_dict(), "qguard_best.pth")
    torch.save(le.classes_, "label_classes.pth")
    torch.save(top_features, "feature_names.pth")
    torch.save(optimal_threshold, "optimal_threshold.pth")
    print("\nArtifacts saved successfully.", flush=True)

if __name__ == "__main__":
    main()
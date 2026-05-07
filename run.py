# run.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

from utils import Config, set_seed, get_device, setup_logging, ensure_dir
from data.data_loader import load_data, split_data
from data.preprocess import clean_data, normalize_features, encode_target
from data.feature_engineering import engineer_features, select_top_features
from data.rebalance import rebalance_dataset
from models.qguard_model import QGuardModel
from training.train import train_model
from eval.calibration import TemperatureScaling, expected_calibration_error, plot_reliability_diagram
from eval.metrics import plot_confusion_matrix, plot_roc_pr_curves
from eval.explainability import generate_shap_summary
from export.export_onnx import export_to_onnx
from export.export_trt import export_to_tensorrt

def main():
    config = Config()
    set_seed(config.seed)
    logger = setup_logging()
    device = get_device(config.device_str)
    ensure_dir("plots")
    
    logger.info("=== Q-GUARD ENTERPRISE IDS PIPELINE ===")
    
    # 1. Data Pipeline
    logger.info("Loading and processing data...")
    df = load_data("data.csv", sample_size=100000)
    df = clean_data(df)
    df = engineer_features(df)
    
    target_col = next((c for c in ['Attack Type', 'Label', 'Attack_Type'] if c in df.columns), df.columns[-1])
    df, le = encode_target(df, target_col)
    num_classes = len(le.classes_)
    logger.info(f"Detected {num_classes} classes: {le.classes_}")
    
    top_features = select_top_features(df, target_col, n_features=8)
    df, scaler = normalize_features(df, top_features)
    df = rebalance_dataset(df, target_col, ratio=5)
    
    train_df, val_df, test_df = split_data(df, target_col)
    
    # 2. GPU-Optimized DataLoaders
    train_loader = DataLoader(TensorDataset(torch.tensor(train_df[top_features].values, dtype=torch.float32), torch.tensor(train_df[target_col].values, dtype=torch.long)), batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_df[top_features].values, dtype=torch.float32), torch.tensor(val_df[target_col].values, dtype=torch.long)), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_df[top_features].values, dtype=torch.float32), torch.tensor(test_df[target_col].values, dtype=torch.long)), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)

    # 3. Class Weights
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_df[target_col].values), y=train_df[target_col].values)
    class_weights = torch.tensor(np.clip(weights, 0.5, 5.0), dtype=torch.float32)

    # 4. Training
    logger.info("Initializing model and starting training...")
    model = QGuardModel(input_dim=len(top_features), num_classes=num_classes)
    model, optimal_threshold, best_f1 = train_model(model, train_loader, val_loader, num_classes, device, config, class_weights)
    logger.info(f"Training Complete. Optimal Threshold: {optimal_threshold:.4f} | Best F1: {best_f1:.4f}")

    # 5. Calibration
    logger.info("Calibrating model...")
    temp_scaler = TemperatureScaling().to(device)
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            all_logits.append(model(xb)["logits"].cpu())
            all_labels.append(yb)
    all_logits, all_labels = torch.cat(all_logits), torch.cat(all_labels)
    
    temp_scaler.fit(all_logits.to(device), all_labels.to(device))
    calibrated_probs = temp_scaler.get_calibrated_probs(all_logits.to(device))
    ece = expected_calibration_error(all_labels.to(device), calibrated_probs)
    logger.info(f"Calibration ECE: {ece:.4f} | Temperature: {temp_scaler.temperature.item():.4f}")
    
    # 6. Visualizations
    logger.info("Generating evaluation plots...")
    plot_reliability_diagram(all_labels, calibrated_probs.cpu(), "plots/reliability_diagram.png")
    
    all_test_probs, all_test_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_test_probs.append(model(xb.to(device))["probs"].cpu())
            all_test_labels.append(yb)
    all_test_probs, all_test_labels = torch.cat(all_test_probs), torch.cat(all_test_labels)
    
    if num_classes == 2:
        test_preds = (all_test_probs[:, 1] >= optimal_threshold).long()
    else:
        test_preds = all_test_probs.argmax(dim=-1)
        
    plot_confusion_matrix(all_test_labels, test_preds, le.classes_, "plots/confusion_matrix.png")
    plot_roc_pr_curves(all_test_labels, all_test_probs, num_classes, le.classes_, "plots/roc_curve.png", "plots/pr_curve.png")
    
    # 7. Explainability
    logger.info("Generating SHAP explanations...")
    X_bg = torch.tensor(train_df[top_features].values[:50], dtype=torch.float32)
    X_exp = torch.tensor(test_df[top_features].values[:20], dtype=torch.float32)
    generate_shap_summary(model, X_bg, X_exp, device, top_features, "plots/shap_summary.png")

    # 8. Export (ONNX + TensorRT)
    logger.info("Exporting model artifacts...")
    export_to_onnx(model, len(top_features))
    export_to_tensorrt("qguard.onnx")

    # 9. Save Metadata
    torch.save(model.state_dict(), "qguard_best.pth")
    torch.save(le.classes_, "label_classes.pth")
    torch.save(top_features, "feature_names.pth")
    torch.save(optimal_threshold, "optimal_threshold.pth")
    logger.info("=== Q-GUARD PIPELINE FINISHED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
# eval/explainability.py
import torch
import numpy as np
import shap

def get_shap_values(model: torch.nn.Module, X_background: torch.Tensor, X_explain: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    def predict_fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.tensor(x_np, dtype=torch.float32).to(device))["probs"].cpu().numpy()
            
    explainer = shap.KernelExplainer(predict_fn, X_background.cpu().numpy())
    shap_values = explainer.shap_values(X_explain.cpu().numpy(), nsamples=100)
    
    # Return mean absolute SHAP per class
    importance = {}
    for c in range(shap_values.shape[0]):
        importance[f"class_{c}"] = np.abs(shap_values[c]).mean(axis=0).tolist()
    return importance

def explain_prediction(model: torch.nn.Module, X: torch.Tensor, features: list, device: torch.device) -> list:
    model.eval()
    with torch.no_grad():
        out = model(X.to(device))
    pred_class = out["logits"].argmax(dim=-1).item()
    uncertainty = out["uncertainty"].item()
    
    explainer = shap.KernelExplainer(
        lambda x: model(torch.tensor(x, dtype=torch.float32).to(device))["probs"].cpu().detach().numpy(),
        torch.randn((10, X.shape[1])).numpy()
    )
    sv = explainer.shap_values(X.cpu().numpy(), nsamples=50)
    att_weights = np.abs(sv[pred_class][0])
    total = att_weights.sum()
    norm_weights = (att_weights / total).tolist() if total > 0 else att_weights.tolist()
    
    return {
        "predicted_class": pred_class,
        "uncertainty": uncertainty,
        "confidence": out["probs"][0][pred_class].item(),
        "feature_contribution": dict(zip(features, norm_weights))
    }
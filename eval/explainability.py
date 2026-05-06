# eval/explainability.py
import torch
import numpy as np
import shap

def explain_predictions(model: torch.nn.Module, X_background: torch.Tensor, X_explain: torch.Tensor, device: torch.device) -> list:
    model.eval()
    X_bg_dev = X_background.to(device)
    
    def model_predict(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
            return model(x_tensor)["probs"].cpu().numpy()
            
    explainer = shap.KernelExplainer(model_predict, X_bg_dev.cpu().numpy())
    shap_values = explainer.shap_values(X_explain.cpu().numpy(), nsamples=100)
    return shap_values
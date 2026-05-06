# eval/explainability.py
import torch
import numpy as np
import shap

def get_shap_values(model: torch.nn.Module, X_background: torch.Tensor, X_explain: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    
    def predict_fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
            return model(x_tensor)["probs"].cpu().numpy()
            
    explainer = shap.KernelExplainer(predict_fn, X_background.cpu().numpy())
    shap_values = explainer.shap_values(X_explain.cpu().numpy(), nsamples=100)
    return shap_values
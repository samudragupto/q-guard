import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logger = logging.getLogger("Q-GUARD")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

# Reduce excessive SHAP logging
logging.getLogger("shap").setLevel(logging.WARNING)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# SHAP VALUE EXTRACTION
# =========================================================

def get_shap_values(
    model: torch.nn.Module,
    X_background: torch.Tensor,
    X_explain: torch.Tensor,
    device: torch.device
) -> dict:

    logger.info("Generating SHAP values...")

    model.eval()

    def predict_fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_tensor = torch.tensor(
                x_np,
                dtype=torch.float32,
                device=device
            )

            outputs = model(x_tensor)

            return outputs["probs"].cpu().numpy()

    background_np = X_background.detach().cpu().numpy()
    explain_np = X_explain.detach().cpu().numpy()

    explainer = shap.KernelExplainer(
        predict_fn,
        background_np
    )

    shap_values = explainer.shap_values(
        explain_np,
        nsamples=100
    )

    importance = {}

    # Handle binary/multiclass safely
    if isinstance(shap_values, list):

        for c in range(len(shap_values)):
            importance[f"class_{c}"] = (
                np.abs(shap_values[c]).mean(axis=0).tolist()
            )

    else:
        importance["class_0"] = (
            np.abs(shap_values).mean(axis=0).tolist()
        )

    logger.info("SHAP value generation complete.")

    return importance

# =========================================================
# SINGLE PREDICTION EXPLANATION
# =========================================================

def explain_prediction(
    model: torch.nn.Module,
    X: torch.Tensor,
    features: list,
    device: torch.device
) -> dict:

    logger.info("Generating prediction explanation...")

    model.eval()

    with torch.no_grad():

        outputs = model(X.to(device))

        pred_class = outputs["logits"].argmax(dim=-1).item()

        uncertainty = outputs["uncertainty"].item()

        confidence = outputs["probs"][0][pred_class].item()

    background = torch.randn(
        (10, X.shape[1]),
        device=device
    ).cpu().numpy()

    def predict_fn(x_np: np.ndarray) -> np.ndarray:

        with torch.no_grad():

            x_tensor = torch.tensor(
                x_np,
                dtype=torch.float32,
                device=device
            )

            return model(x_tensor)["probs"].cpu().numpy()

    explainer = shap.KernelExplainer(
        predict_fn,
        background
    )

    shap_values = explainer.shap_values(
        X.detach().cpu().numpy(),
        nsamples=50
    )

    # Binary / multiclass safe handling
    if isinstance(shap_values, list):
        att_weights = np.abs(shap_values[pred_class][0])
    else:
        att_weights = np.abs(shap_values[0])

    total = att_weights.sum()

    if total > 0:
        norm_weights = (att_weights / total).tolist()
    else:
        norm_weights = att_weights.tolist()

    logger.info("Prediction explanation generated successfully.")

    return {
        "predicted_class": pred_class,
        "uncertainty": uncertainty,
        "confidence": confidence,
        "feature_contribution": dict(
            zip(features, norm_weights)
        )
    }

# =========================================================
# SHAP SUMMARY PLOT
# =========================================================

def generate_shap_summary(
    model: torch.nn.Module,
    X_background: torch.Tensor,
    X_explain: torch.Tensor,
    device: torch.device,
    feature_names: list,
    path: str
):

    logger.info("Generating SHAP summary plot...")

    model.eval()

    def predict_fn(x_np: np.ndarray) -> np.ndarray:

        with torch.no_grad():

            x_tensor = torch.tensor(
                x_np,
                dtype=torch.float32,
                device=device
            )

            return model(x_tensor)["probs"].cpu().numpy()

    background_np = X_background.detach().cpu().numpy()
    explain_np = X_explain.detach().cpu().numpy()

    explainer = shap.KernelExplainer(
        predict_fn,
        background_np
    )

    shap_values = explainer.shap_values(
        explain_np,
        nsamples=50
    )

    plt.figure(figsize=(12, 6))

    shap.summary_plot(
        shap_values,
        explain_np,
        feature_names=feature_names,
        show=False
    )

    plt.savefig(
        path,
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    logger.info(
        f"SHAP summary plot saved successfully -> {path}"
    )
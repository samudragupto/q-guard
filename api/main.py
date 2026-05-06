# api/main.py
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from models.qguard_model import QGuardModel
from eval.explainability import explain_predictions

app = FastAPI(title="Q-GUARD API")

class FlowInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    label: str
    confidence: float
    attack_probability: float

class ExplainOutput(BaseModel):
    feature_importances: list[float]

model = None
background_data = None
device = None
label_map = {0: "BENIGN", 1: "ALERT"}

@app.on_event("startup")
def load_model():
    global model, background_data, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QGuardModel(input_dim=8).to(device)
    try:
        model.load_state_dict(torch.load("qguard_weights.pth", weights_only=True, map_location=device))
    except FileNotFoundError:
        pass
    model.eval()
    background_data = torch.randn((10, 8))

@app.post("/predict", response_model=PredictionOutput)
def predict(flow: FlowInput):
    x = torch.tensor([flow.features], dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(x)
    conf, pred = output["probs"].max(dim=-1)
    return PredictionOutput(
        label=label_map.get(pred.item(), "UNKNOWN"),
        confidence=conf.item(),
        attack_probability=output["probs"][0][1].item()
    )

@app.post("/explain", response_model=ExplainOutput)
def explain(flow: FlowInput):
    x = torch.tensor([flow.features], dtype=torch.float32)
    shap_values = explain_predictions(model, background_data, x, device)
    attack_importance = np.abs(shap_values[1][0])
    total = attack_importance.sum()
    normalized = (attack_importance / total).tolist() if total > 0 else attack_importance.tolist()
    return ExplainOutput(feature_importances=normalized)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}
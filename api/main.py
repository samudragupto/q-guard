# api/main.py
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from models.qguard_model import QGuardModel
from eval.explainability import explain_prediction

app = FastAPI(title="Q-GUARD SOC API")

class FlowInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    label: str
    confidence: float
    uncertainty: float
    threshold_used: float

class ExplainOutput(BaseModel):
    predicted_class: str
    confidence: float
    uncertainty: float
    feature_contribution: dict[str, float]

model = None
device = None
label_classes = None
feature_names = None
optimal_threshold = 0.5

@app.on_event("startup")
def load_model():
    global model, device, label_classes, feature_names, optimal_threshold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_classes = torch.load("label_classes.pth", weights_only=True)
    feature_names = torch.load("feature_names.pth", weights_only=True)
    try:
        optimal_threshold = torch.load("optimal_threshold.pth", weights_only=True)
    except FileNotFoundError:
        pass
    model = QGuardModel(input_dim=len(feature_names), num_classes=len(label_classes)).to(device)
    try:
        model.load_state_dict(torch.load("qguard_best.pth", weights_only=True, map_location=device))
    except FileNotFoundError:
        pass
    model.eval()

@app.post("/predict", response_model=PredictionOutput)
def predict(flow: FlowInput):
    x = torch.tensor([flow.features], dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(x)
    attack_prob = out["probs"][0][1].item()
    pred = 1 if attack_prob >= optimal_threshold else 0
    return PredictionOutput(
        label=label_classes[pred],
        confidence=out["probs"][0][pred].item(),
        uncertainty=out["uncertainty"][0].item(),
        threshold_used=optimal_threshold
    )

@app.post("/explain", response_model=ExplainOutput)
def explain(flow: FlowInput):
    x = torch.tensor([flow.features], dtype=torch.float32)
    res = explain_prediction(model, x, feature_names, device)
    return ExplainOutput(
        predicted_class=label_classes[res["predicted_class"]],
        confidence=res["confidence"],
        uncertainty=res["uncertainty"],
        feature_contribution=res["feature_contribution"]
    )

@app.get("/health")
def health():
    return {"status": "healthy", "device": str(device)}
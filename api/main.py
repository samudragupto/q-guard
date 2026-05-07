# api/main.py
import torch
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.qguard_model import QGuardModel
import logging

logger = logging.getLogger("Q-GUARD")
app = FastAPI(title="Q-GUARD IDS API", version="2.0")

class PacketInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    label: str
    confidence: float
    uncertainty: float
    threshold_used: float

class ExplainOutput(BaseModel):
    label: str
    confidence: float
    feature_contributions: dict[str, float]

model = None
device = None
label_classes = None
feature_names = None
optimal_threshold = 0.5

@app.on_event("startup")
def startup():
    global model, device, label_classes, feature_names, optimal_threshold
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        label_classes = torch.load("label_classes.pth", weights_only=True)
        feature_names = torch.load("feature_names.pth", weights_only=True)
        optimal_threshold = torch.load("optimal_threshold.pth", weights_only=True)
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return
        
    model = QGuardModel(input_dim=len(feature_names), num_classes=len(label_classes)).to(device)
    try:
        model.load_state_dict(torch.load("qguard_best.pth", weights_only=True, map_location=device))
        model.eval()
        logger.info("Model loaded successfully on {device}.")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")

@app.post("/predict", response_model=PredictionOutput)
def predict_packet(packet: PacketInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(packet.features) != len(feature_names):
        raise HTTPException(status_code=400, detail=f"Expected {len(feature_names)} features")
        
    x = torch.tensor([packet.features], dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(x)
        
    probs = out["probs"][0]
    if len(label_classes) == 2:
        pred_idx = 1 if probs[1].item() >= optimal_threshold else 0
    else:
        pred_idx = torch.argmax(probs).item()
        
    return PredictionOutput(
        label=label_classes[pred_idx],
        confidence=probs[pred_idx].item(),
        uncertainty=out["uncertainty"][0].item(),
        threshold_used=optimal_threshold
    )

@app.post("/explain", response_model=ExplainOutput)
def explain_packet(packet: PacketInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    x = torch.tensor([packet.features], dtype=torch.float32).to(device)
    
    def predict_fn(x_np):
        with torch.no_grad():
            return model(torch.tensor(x_np, dtype=torch.float32).to(device))["probs"].cpu().numpy()
            
    background = torch.randn((10, len(feature_names))).numpy()
    explainer = shap.KernelExplainer(predict_fn, background)
    sv = explainer.shap_values(x.cpu().numpy(), nsamples=20)
    
    with torch.no_grad():
        out = model(x)
        
    pred_idx = torch.argmax(out["probs"][0]).item()
    att_weights = np.abs(sv[pred_idx][0])
    total = att_weights.sum()
    norm_weights = (att_weights / total).tolist() if total > 0 else att_weights.tolist()
    
    return ExplainOutput(
        label=label_classes[pred_idx],
        confidence=out["probs"][0][pred_idx].item(),
        feature_contributions=dict(zip(feature_names, norm_weights))
    )

@app.get("/health")
def health():
    return {"status": "healthy", "device": str(device), "model_loaded": model is not None}
# api/main.py
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from models.qguard_model import QGuardModel
from eval.explainability import get_shap_values

app = FastAPI(title="Q-GUARD API")

class FlowInput(BaseModel):
    features: list[float]

class BatchFlowInput(BaseModel):
    batch: list[FlowInput]

class PredictionOutput(BaseModel):
    label: str
    confidence: float
    uncertainty: float
    probabilities: dict[str, float]

class ExplainOutput(BaseModel):
    label: str
    feature_importances: dict[str, float]

model = None
background_data = None
device = None
label_classes = None
feature_names = None
optimal_threshold = 0.5

@app.on_event("startup")
def load_model():
    global model, background_data, device, label_classes, feature_names, optimal_threshold
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
    background_data = torch.randn((20, len(feature_names)))

@app.post("/predict", response_model=list[PredictionOutput])
def predict(flow: BatchFlowInput):
    x = torch.tensor([f.features for f in flow.batch], dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(x)
    
    results = []
    for i in range(len(flow.batch)):
        if len(label_classes) == 2:
            pred = 1 if output["probs"][i][1].item() > optimal_threshold else 0
            conf = output["probs"][i][pred].item()
        else:
            conf, pred = output["probs"][i].max(dim=-1)
            pred = pred.item()
            conf = conf.item()
            
        probs = {label_classes[j]: output["probs"][i][j].item() for j in range(len(label_classes))}
        results.append(PredictionOutput(
            label=label_classes[pred],
            confidence=conf,
            uncertainty=output["uncertainty"][i].item(),
            probabilities=probs
        ))
    return results

@app.post("/explain", response_model=list[ExplainOutput])
def explain(flow: BatchFlowInput):
    x = torch.tensor([f.features for f in flow.batch], dtype=torch.float32)
    shap_values = get_shap_values(model, background_data, x, device)
    
    results = []
    for i in range(len(flow.batch)):
        pred = model(x[i].unsqueeze(0).to(device))["logits"].argmax(dim=-1).item()
        importances = np.abs(shap_values[pred][i])
        total = importances.sum()
        norm_imp = (importances / total).tolist() if total > 0 else importances.tolist()
        results.append(ExplainOutput(
            label=label_classes[pred],
            feature_importances=dict(zip(feature_names, norm_imp))
        ))
    return results

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "device": str(device)}
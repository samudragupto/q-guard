# api/main.py
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from models.qguard_model import QGuardModel

app = FastAPI(title="Q-GUARD API")

class FlowInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    label: str
    confidence: float
    uncertainty: float
    circuit_depth: float

model = None
label_map = {0: "BENIGN", 1: "ALERT", 2: "REVIEW"}

@app.on_event("startup")
def load_model():
    global model
    model = QGuardModel(input_dim=8)
    try:
        model.load_state_dict(torch.load("qguard_weights.pth", map_location="cpu"))
        print("[DEBUG] Model weights loaded successfully.")
    except FileNotFoundError:
        print("[DEBUG] Warning: qguard_weights.pth not found. Using untrained model.")
    model.eval()

@app.post("/predict", response_model=PredictionOutput)
def predict(flow: FlowInput):
    x = torch.tensor([flow.features], dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
    
    decision = output["decisions"][0].item()
    confidence = output["confidence"][0].item()
    uncertainty = output["uncertainty"][0].item()
    depth = output["depth"][0].item()
    
    return PredictionOutput(
        label=label_map.get(int(decision), "REVIEW"),
        confidence=confidence,
        uncertainty=uncertainty,
        circuit_depth=depth
    )

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}
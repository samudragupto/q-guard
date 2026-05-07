# utils.py
import torch
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 256
    device: str = "cuda"
    seed: int = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(device_str: str = "cuda") -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available. Falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_str)

def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, non_blocking=True)
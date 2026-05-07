# utils.py
import torch
import random
import numpy as np
import logging
import os
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 7
    adv_training: bool = True
    use_focal: bool = True
    eps_adv: float = 0.03
    num_workers: int = 4
    pin_memory: bool = True
    device_str: str = "cuda"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

def get_device(device_str: str = "cuda") -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        logging.warning("[Q-GUARD] CUDA not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
    return logging.getLogger("Q-GUARD")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
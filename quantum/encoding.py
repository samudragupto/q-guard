# quantum/encoding.py
import pennylane as qml
import torch

def angle_encoding(x: torch.Tensor, wires: list) -> None:
    for i, wire in enumerate(wires):
        qml.RY(x[i], wires=wire)
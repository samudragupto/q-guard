# quantum/attention.py
import pennylane as qml
import torch
import torch.nn as nn
import math
from quantum.encoding import angle_encoding

class QuantumAttention(nn.Module):
    def __init__(self, n_qubits: int = 8, max_depth: int = 4, temp: float = 0.5):
        super().__init__()
        self.n_qubits = n_qubits
        self.max_depth = max_depth
        self.temp = temp
        
        self.mask_logits = nn.Parameter(torch.randn(n_qubits, n_qubits) * 0.1)
        self.ry_weights = nn.Parameter(torch.randn(max_depth, n_qubits) * 0.1)
        
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=None)
        self.wires = list(range(n_qubits))
        
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="parameter-shift")

    def _get_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.n_qubits, self.n_qubits)
        for i in range(self.n_qubits):
            for j in range(i+1, self.n_qubits):
                prob = torch.sigmoid(self.mask_logits[i, j])
                gate_val = self._gumbel_sample(prob)
                mask[i, j] = gate_val
                mask[j, i] = gate_val
        return mask

    def _gumbel_sample(self, prob: torch.Tensor) -> torch.Tensor:
        if self.training:
            uniform = torch.rand_like(prob)
            uniform = torch.clamp(uniform, 1e-6, 1 - 1e-6)
            g = -torch.log(-torch.log(uniform))
            return torch.sigmoid((torch.log(prob + 1e-8) - torch.log(1 - prob + 1e-8) + g) / self.temp)
        return (prob > 0.5).float()

    def _circuit(self, x: torch.Tensor, mask: torch.Tensor, depth: float) -> list:
        angle_encoding(x, self.wires)
        
        int_depth = int(torch.round(depth).item())
        int_depth = max(1, min(int_depth, self.max_depth))
        
        for d in range(int_depth):
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    if mask[i, j] > 0.5:
                        qml.CZ(wires=[self.wires[i], self.wires[j]])
            for i in range(self.n_qubits):
                qml.RY(self.ry_weights[d, i], wires=self.wires[i])
                
        return [qml.expval(qml.PauliZ(w)) for w in self.wires]

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        mask = self._get_mask()
        batch_size = x.shape[0]
        results = []
        
        for i in range(batch_size):
            res = self.qnode(x[i], mask, depth[i].detach())
            results.append(torch.stack(res))
            
        return torch.stack(results)
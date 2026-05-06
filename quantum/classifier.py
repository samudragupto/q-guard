# quantum/classifier.py
import pennylane as qml
import torch
import torch.nn as nn
from quantum.encoding import angle_encoding

class QuantumClassifier(nn.Module):
    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=None)
        self.wires = list(range(n_qubits))
        
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="parameter-shift")

    def _circuit(self, x: torch.Tensor) -> tuple:
        angle_encoding(x, self.wires)
        
        for d in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(self.weights[d, i], wires=self.wires[i])
            for i in range(self.n_qubits):
                qml.CNOT(wires=[self.wires[i], self.wires[(i+1) % self.n_qubits]])
                
        half = self.n_qubits // 2
        op_benign = qml.PauliZ(self.wires[0])
        for i in range(1, half):
            op_benign = op_benign @ qml.PauliZ(self.wires[i])
            
        op_attack = qml.PauliZ(self.wires[half])
        for i in range(half+1, self.n_qubits):
            op_attack = op_attack @ qml.PauliZ(self.wires[i])
            
        return qml.expval(op_benign), qml.expval(op_attack), qml.var(op_attack - op_benign)

    def forward(self, x: torch.Tensor) -> tuple:
        batch_size = x.shape[0]
        logits, uncertainties = [], []
        
        for i in range(batch_size):
            l_b, l_a, var_diff = self.qnode(x[i])
            logits.append(torch.stack([l_b, l_a]))
            uncertainties.append(var_diff)
            
        logits = torch.stack(logits)
        uncertainties = torch.stack(uncertainties)
        return logits, uncertainties
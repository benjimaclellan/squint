
# Noise

Real quantum devices are noisy -- this guide shows you how to add noise (Kraus) channels to a `squint` circuit

```python
from squint.ops.dv import DiscreteVariableState, HGate, CXGate
from squint.ops.noise import DepolarizingChannel

def create_noisy_sensor(n_qubits, p=0.1):
    """Create a quantum sensor with depolarizing noise."""
    circuit = Circuit(backend="mixed")  # Use mixed state backend for noise
    
    # Initialize qubits
    for i in range(n_qubits):
        circuit.add(DiscreteVariableState(wires=(i,), n=(0,)))
    
    # GHZ preparation with noise
    circuit.add(HGate(wires=(0,)))
    circuit.add(DepolarizingChannel(wires=(0,), p=noise_strength))
    
    for i in range(1, n_qubits):
        circuit.add(CXGate(wires=(0, i)))
        # Add noise after each two-qubit gate
        circuit.add(DepolarizingChannel(wires=(0,), gamma=noise_strength))
        circuit.add(DepolarizingChannel(wires=(i,), gamma=noise_strength))
    
    # Phase sensing
    for i in range(n_qubits):
        circuit.add(RZGate(wires=(i,), phi=0.0), f"phase_{i}")
    
    # Measurement
    for i in range(n_qubits):
        circuit.add(HGate(wires=(i,)))
    
    return circuit

```
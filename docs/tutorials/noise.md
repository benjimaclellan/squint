
# Noise

Real quantum devices are noisy -- this guide shows you how to add noise (Kraus) channels to a `squint` circuit

```python
from squint.circuit import Circuit
from squint.ops.base import Wire
from squint.ops.dv import DiscreteVariableState, HGate, CXGate, RZGate
from squint.ops.noise import DepolarizingChannel

def create_noisy_sensor(n_qubits, noise_strength=0.1):
    """Create a quantum sensor with depolarizing noise."""
    wires = [Wire(dim=2, idx=i) for i in range(n_qubits)]

    circuit = Circuit(backend="mixed")  # Use mixed state backend for noise

    # Initialize qubits
    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

    # GHZ preparation with noise
    circuit.add(HGate(wires=(wires[0],)))
    circuit.add(DepolarizingChannel(wires=(wires[0],), p=noise_strength))

    for i in range(1, n_qubits):
        circuit.add(CXGate(wires=(wires[0], wires[i])))
        # Add noise after each two-qubit gate
        circuit.add(DepolarizingChannel(wires=(wires[0],), p=noise_strength))
        circuit.add(DepolarizingChannel(wires=(wires[i],), p=noise_strength))

    # Phase sensing
    for i, w in enumerate(wires):
        circuit.add(RZGate(wires=(w,), phi=0.0), f"phase_{i}")

    # Measurement
    for w in wires:
        circuit.add(HGate(wires=(w,)))

    return circuit

```
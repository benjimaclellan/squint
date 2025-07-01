# Entangled sensors

Now let's explore how entanglement can improve sensing precision beyond classical limits. The Greenberger-Horne-Zeilinger (GHZ) state is a maximally entangled state that can achieve the Heisenberg limit for phase estimation:

$$|\mathrm{GHZ}\rangle_n = \frac{1}{\sqrt{2}}(|00...0\rangle + |11...1\rangle)$$

```python
from squint.circuit import Circuit
from squint.ops.base import SharedGate
from squint.ops.dv import CXGate, HGate, DiscreteVariableState, RZGate

def create_ghz_circuit(n_qubits, phi=0.0):
    """Create a GHZ state preparation circuit for n qubits."""
    circuit = Circuit(backend="pure")
    
    # Initialize all qubits in |0⟩
    for i in range(n_qubits):
        circuit.add(DiscreteVariableState(wires=(i,), n=(0,)))
    
    # Create GHZ state: H on first qubit, then CNOTs
    circuit.add(HGate(wires=(0,)))
    for i in range(1, n_qubits):
        circuit.add(CNOTGate(wires=(0, i)))
    
    # Phase evolution on all qubits
    circuit.add(
        SharedGate(op=RZGate(wires=(0,), phi=0.0 * jnp.pi), wires=tuple(range(1, n_qubits))),
        "phase",
    )
    
    # Final measurement basis rotation
    for i in range(n_qubits):
        circuit.add(HGate(wires=(i,)))
    
    return circuit

# Create a 4-qubit GHZ sensor
n_qubits = 4
circuit = create_ghz_circuit(n_qubits)
```

- **Standard Quantum Limit (SQL)**: $\mathcal{I}_\phi \sim n$ (linear scaling)
- **Heisenberg Limit (HL)**: $\mathcal{I}_\phi \sim n^2$ (quadratic scaling)

The GHZ state can achieve the Heisenberg limit, providing quadratic improvement in precision.

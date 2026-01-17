# Entangled sensors

Now let's explore how entanglement can improve sensing precision beyond classical limits. The Greenberger-Horne-Zeilinger (GHZ) state is a maximally entangled state that can achieve the Heisenberg limit for phase estimation:

$$|\mathrm{GHZ}\rangle_n = \frac{1}{\sqrt{2}}(|00...0\rangle + |11...1\rangle)$$

```python
import jax.numpy as jnp
from squint.circuit import Circuit
from squint.ops.base import SharedGate, Wire
from squint.ops.dv import CXGate, HGate, DiscreteVariableState, RZGate

def create_ghz_circuit(n_qubits, phi=0.0):
    """Create a GHZ state preparation circuit for n qubits."""
    wires = [Wire(dim=2, idx=i) for i in range(n_qubits)]

    circuit = Circuit(backend="pure")

    # Initialize all qubits in |0⟩
    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

    # Create GHZ state: H on first qubit, then CXs (CNOTs)
    circuit.add(HGate(wires=(wires[0],)))
    for i in range(1, n_qubits):
        circuit.add(CXGate(wires=(wires[0], wires[i])))

    # Phase evolution on all qubits
    circuit.add(
        SharedGate(op=RZGate(wires=(wires[0],), phi=0.0 * jnp.pi), wires=tuple(wires[1:])),
        "phase",
    )

    # Final measurement basis rotation
    for w in wires:
        circuit.add(HGate(wires=(w,)))

    return circuit

# Create a 4-qubit GHZ sensor
n_qubits = 4
circuit = create_ghz_circuit(n_qubits)
```

- **Standard Quantum Limit (SQL)**: $\mathcal{I}_\phi \sim n$ (linear scaling)
- **Heisenberg Limit (HL)**: $\mathcal{I}_\phi \sim n^2$ (quadratic scaling)

The GHZ state can achieve the Heisenberg limit, providing quadratic improvement in precision.

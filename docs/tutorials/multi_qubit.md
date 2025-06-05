
## Tutorial 2: Multi-Qubit Entangled Sensors

Now let's explore how entanglement can improve sensing precision beyond classical limits.

### The GHZ State Protocol

The Greenberger-Horne-Zeilinger (GHZ) state is a maximally entangled state that can achieve the Heisenberg limit for phase estimation:

$$|GHZ_n\rangle = \frac{1}{\sqrt{2}}(|00...0\rangle + |11...1\rangle)$$

```python
from squint.ops.dv import CNOTGate

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
    for i in range(n_qubits):
        circuit.add(RZGate(wires=(i,), phi=phi), f"phase_{i}")
    
    # Final measurement basis rotation
    for i in range(n_qubits):
        circuit.add(HGate(wires=(i,)))
    
    return circuit

# Create a 4-qubit GHZ sensor
n_qubits = 4
circuit = create_ghz_circuit(n_qubits)
```

### Classical vs Quantum Scaling

Let's compare how Fisher Information scales with the number of qubits:

```python
def compare_scaling():
    """Compare classical and quantum scaling of Fisher Information."""
    max_qubits = 6
    classical_fi = []
    quantum_fi = []
    
    for n in range(1, max_qubits + 1):
        # Create GHZ circuit
        circuit = create_ghz_circuit(n)
        dim = 2
        params, static = partition_op(circuit, [f"phase_{i}" for i in range(n)])
        sim = circuit.compile(static, dim, params).jit()
        
        # Set phase to π/4 for maximum Fisher Information
        phi = jnp.pi / 4
        params_dict = {f"phase_{i}": {"phi": phi} for i in range(n)}
        
        # Calculate Fisher Information
        qfi = sim.amplitudes.qfim(params_dict)
        cfi = sim.probabilities.cfim(params_dict)
        
        quantum_fi.append(float(jnp.sum(qfi)))  # Total QFI
        classical_fi.append(float(jnp.sum(cfi)))  # Total CFI
    
    # Plot scaling comparison
    n_values = jnp.arange(1, max_qubits + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, classical_fi, 'o-', label='Classical Fisher Info', linewidth=2)
    plt.plot(n_values, quantum_fi, 's-', label='Quantum Fisher Info', linewidth=2)
    plt.plot(n_values, n_values, '--', label='Standard Quantum Limit (∝n)', alpha=0.7)
    plt.plot(n_values, n_values**2, '--', label='Heisenberg Limit (∝n²)', alpha=0.7)
    
    plt.xlabel('Number of Qubits')
    plt.ylabel('Fisher Information')
    plt.legend()
    plt.title('Scaling of Fisher Information with System Size')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()

compare_scaling()
```

### Understanding the Advantage

- **Standard Quantum Limit (SQL)**: $\mathcal{I}_\phi \sim n$ (linear scaling)
- **Heisenberg Limit (HL)**: $\mathcal{I}_\phi \sim n^2$ (quadratic scaling)
- **Precision improvement**: $\Delta\phi \sim 1/\sqrt{\mathcal{I}_\phi}$

The GHZ state can achieve the Heisenberg limit, providing quadratic improvement in precision!

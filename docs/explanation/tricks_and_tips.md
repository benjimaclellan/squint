
# Technical Reference

### Circuits

The `Circuit` class is the main interface for building quantum sensing protocols.

```python
from squint.circuit import Circuit

# Initialize circuit
circuit = Circuit(backend="pure")  # or "mixed" for noisy simulations

# Add quantum operations
circuit.add(operation, label=None)

# Compile for simulation
sim = circuit.compile(static_params, dimension, variable_params)
```

The key methods are,

- `add()`: Add quantum operations to the circuit
- `compile()`: Compile circuit for efficient simulation

### Operations

Discrete variable
```python
from squint.ops.dv import *

# Pauli gates
XGate(wires=(0,))
YGate(wires=(0,))  
ZGate(wires=(0,))

# Rotation gates
RXGate(wires=(0,), phi=angle)
RYGate(wires=(0,), phi=angle)
RZGate(wires=(0,), phi=angle)

# Hadamard gate
HGate(wires=(0,))
```

#### Two-Qubit Gates
```python
# CNOT gate
CXGate(wires=(control, target))

# Controlled-Z gate
CZGate(wires=(control, target))

# Controlled-Phase gate
CPhaseGate(wires=(control, target), phi=angle)
```

#### Channels
```python
# Dephasing channel
DephasingChannel(wires=(0,), p=noise_strength)

# Depolarizing channel
DepolarizingChannel(wires=(0,), p=error_probability)
```

### Simulation interface

After compiling a circuit, you get a simulation object with these methods:

```python
# Forward simulation
amplitudes = sim.amplitudes.forward(params)      # Quantum amplitudes
probabilities = sim.probabilities.forward(params) # Measurement probabilities

# Gradient computation
damp = sim.amplitudes.grad(params)               # ∂|ψ⟩/∂θ  
dprob = sim.probabilities.grad(params)           # ∂p/∂θ

# Fisher Information
qfim = sim.amplitudes.qfim(params)               # Quantum Fisher Information Matrix
cfim = sim.probabilities.cfim(params)            # Classical Fisher Information Matrix

```

### Utilities

```python
from squint.utils import *

# Parameter partitioning
params, static = partition_op(circuit, parameter_labels)

# State visualization  
print_nonzero_entries(quantum_state)

# Circuit optimization
optimized_params = optimize_circuit(circuit, objective_fn, **kwargs)
```

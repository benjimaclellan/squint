
# Technical Reference

### Circuits

The `Circuit` class is the main interface for building quantum sensing protocols.

```python
from squint.circuit import Circuit
from squint.simulator.tn import Simulator

# Initialize circuit (backend auto-selected based on operations)
circuit = Circuit()

# Add quantum operations
circuit.add(operation, label=None)

# Compile for simulation
sim = Simulator.compile(static_params, variable_params)
```

The key methods are,

- `add()`: Add quantum operations to the circuit
- `Simulator.compile()`: Compile circuit for efficient simulation

### Operations

#### Discrete variable
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

# Hadamard Gate
HGate(wires=(0,))

```

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

#### Fock/photon-number
```python
from squint.ops.fock import *

FockState(wires=(0,), n=(0,))

# Beam splitter
BeamSplitter(wires=(0, 1), r=jnp.pi/4)

# Phase shift
Phase(wires=(0,), phi=0.0)
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

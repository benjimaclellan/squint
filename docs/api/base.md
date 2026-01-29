# Base Module

The `squint.ops.base` module contains the core abstractions for building quantum circuits in Squint.

## Key Concepts

### Wires and Degrees of Freedom

A **Wire** represents a quantum subsystem with a specific Hilbert space dimension. Each wire can optionally specify a degree of freedom (DoF) type to distinguish between different physical encodings:

- **DV** - Discrete variable systems (qubits, qudits)
- **CV** - Continuous variable systems (optical modes in Fock space)
- **TimeBin**, **FreqBin**, **Spatial** - Photonic encoding schemes

### Operation Hierarchy

All quantum operations inherit from `AbstractOp`:

- **States**: `AbstractPureState`, `AbstractMixedState` - Initial quantum states
- **Gates**: `AbstractGate` - Unitary transformations
- **Channels**: `AbstractKrausChannel`, `AbstractErasureChannel` - Non-unitary operations
- **SharedGate** - Parameter sharing across multiple wires (e.g., for phase estimation)
- **Block** - Grouping multiple operations

### Typical Usage

```python
from squint.ops.base import Wire, DV, SharedGate

# Create qubit wires
q0 = Wire(dim=2, dof=DV, idx=0)
q1 = Wire(dim=2, dof=DV, idx=1)

# Use in operations
from squint.ops.dv import DiscreteVariableState, RZGate
state = DiscreteVariableState(wires=(q0,), n=(0,))
phase = RZGate(wires=(q0,), phi=0.0)
```

---

<!-- prettier-ignore -->
::: squint.ops.base
    options:
        heading_level: 3

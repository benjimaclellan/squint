# Quantum Operations

This page documents all quantum operations available in Squint, organized by category.

---

## Fock Space Operations

Operations for continuous variable (CV) quantum systems using the Fock (photon number) basis. These are commonly used for simulating linear optical networks and photonic quantum computing.

**Key classes:**

- `FockState` - Fock basis states and superpositions
- `BeamSplitter` - Two-mode beam splitter
- `Phase` - Single-mode phase shift
- `LinearOpticalUnitaryGate` - General passive linear optical transformation

<!-- prettier-ignore -->
::: squint.ops.fock
    options:
        heading_level: 3

---

## Discrete Variable Operations

Operations for finite-dimensional quantum systems including qubits (dim=2) and qudits (dim>2). Includes standard gates like Pauli rotations, Hadamard, and controlled gates.

**Key classes:**

- `DiscreteVariableState` - Computational basis states and superpositions
- `XGate`, `ZGate`, `HGate` - Standard single-qubit gates (generalized for qudits)
- `RXGate`, `RYGate`, `RZGate` - Parameterized rotation gates
- `CXGate`, `CZGate` - Controlled gates

<!-- prettier-ignore -->
::: squint.ops.dv
    options:
        heading_level: 3

---

## Noise Channels

Quantum noise channels for modeling decoherence and errors. These require the "mixed" backend as they produce density matrices.

**Key classes:**

- `BitFlipChannel` - Random X errors with probability p
- `PhaseFlipChannel` - Random Z errors (dephasing) with probability p
- `DepolarizingChannel` - Symmetric Pauli noise
- `ErasureChannel` - Traces out (erases) specified wires

<!-- prettier-ignore -->
::: squint.ops.noise
    options:
        heading_level: 3

---

## Blocks

Blocks allow grouping multiple operations for organizational purposes.

<!-- prettier-ignore -->
::: squint.blocks
    options:
        heading_level: 3
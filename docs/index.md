#

<p align="center">
  <img src="img/squint-logo.png#only-light" alt="squint logo" style="max-height: 200px;">
  <img src="img/squint-logo-dark.png#only-dark" alt="squint logo" style="max-height: 200px;">
</p>

<div align="center">
    <!-- <h2 align="center">
    squint
    </h2> -->
    <div>
    Welcome to <b>squint</b>, a differentiable framework for studying and designing quantum metrology and sensing protocols!
    </div>
</div>

[![CI](https://github.com/benjimaclellan/squint/actions/workflows/pytest.yml/badge.svg)](https://github.com/benjimaclellan/squint/actions/workflows/pytest.yml)
![versions](https://img.shields.io/badge/python-3.11+-blue)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<!-- Welcome to **squint**, a differentiable framework for studying and designing quantum metrology and sensing protocols! -->

## What can it do?

- **Differentiable quantum dynamics** for qubit, qudit, and Fock/photon-number systems
- **Built on JAX ecosystem** for automatic differentiation and GPU hardware acceleration
- **Compute Fisher information** and other fundamental metrics in quantum metrology with ease
- **Benchmark realistic protocols** with noise and loss channels relevant to diverse device architectures

## A quick example

```python
from squint.circuit import Circuit
from squint.ops.base import Wire
from squint.ops.dv import DiscreteVariableState, HGate, RZGate
from squint.utils import print_nonzero_entries, partition_op

# Create a simple one-qubit phase estimation circuit
# |0⟩ --- H --- Rz(φ) --- H --- |⟩
wire = Wire(dim=2, idx=0)  # qubit with dim=2
circuit = Circuit(backend="pure")
circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))       # |0⟩ state
circuit.add(HGate(wires=(wire,)))                               # Hadamard gate
circuit.add(RZGate(wires=(wire,), phi=0.0 * jnp.pi), "phase")   # Phase rotation
circuit.add(HGate(wires=(wire,)))                               # Second Hadamard

# Compile the circuit for simulation
params, static = partition_op(circuit, "phase")
sim = circuit.compile(static, params, optimize="greedy").jit()

# Calculate metrics important to quantum metrology & sensing protocols
# the quantum state and its gradient
psi = sim.amplitudes.forward(params)      # |ψ(θ)⟩
dpsi = sim.amplitudes.grad(params)        # ∂|ψ(θ)⟩/∂θ

# Probabilities and their gradients
p = sim.probabilities.forward(params)     # p(s|θ)
dp = sim.probabilities.grad(params)       # ∂p(s|θ)/∂θ

qfi = sim.amplitudes.qfim(params)       # Quantum Fisher Information
cfi = sim.probabilities.cfim(params)    # Classical Fisher Information
```


## Acknowledgments

The authors of `squint` acknowledge the kind support of
[Ki3 Photonics Technologies](https://ki3photonics.com),
[Perimeter Institute Quantum Intelligence Lab](https://perimeterinstitute.ca/perimeter-institute-quantum-intelligence-lab-piquil),
[Institute for Quantum Computing](https://uwaterloo.ca/institute-for-quantum-computing/).

## Citing

If you found this package, please consider citing our work!

```
@article{maclellan2024endtoend,
      title={End-to-end variational quantum sensing}, 
      author={Benjamin MacLellan and Piotr Roztocki and Stefanie Czischek and Roger G. Melko},
      year={2024},
      eprint={2403.02394},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
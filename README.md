
<h1 align="center">
    squint
</h1>

    > can't see that star? squint a little harder

<!-- [![doc](https://img.shields.io/badge/documentation-lightblue)]() -->
<!-- [![PyPI Version](https://img.shields.io/pypi/v/oqd-core)](https://pypi.org/project/oqd-core) -->
[![CI](https://github.com/benjimaclellan/squint/actions/workflows/pytest.yml/badge.svg)](https://github.com/benjimaclellan/squint/actions/workflows/pytest.yml)
![versions](https://img.shields.io/badge/python->=3.11-blue)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)



## Installation

```bash
pip install squint@git+https://github.com/benjimaclellan/squint
```
or
```bash
uv pip install squint@git+https://github.com/benjimaclellan/squint
```

Simply clone the repo locally,

```bash
git clone https://github.com/benjimaclellan/squint
cd squint
```

Use `uv` for the package management. Installation instructions are [here](https://docs.astral.sh/uv/getting-started/installation/).

Create a virtual environment with all the correct dependencies and activate it,

```bash
uv sync
source .venv/bin/activate
```

## Example

> Last updated: 2025-03-06

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.dv import DiscreteVariableState, HGate, RZGate
from squint.utils import print_nonzero_entries, partition_op

# let's implement a simple one-qubit circuit for phase estimation;
#          _____     _________     _____      _____
# |0 > --- | H | --- | Rz(Φ) | --- | H | ---- | / |====
#          -----     ---------     -----      -----

circuit = Circuit(backend="pure")
circuit.add(DiscreteVariableState(wires=(0,), n=(0,)))
circuit.add(HGate(wires=(0,)))
circuit.add(RZGate(wires=(0,), phi=0.0 * jnp.pi), "phase")
circuit.add(HGate(wires=(0,)))

dim = 2  # qubit circuit
params, static = partition_op(circuit, "phase")
sim = circuit.compile(static, dim, params, optimize="greedy").jit()

# calculate the quantum probability amplitudes and their derivatives with respect to Φ
ket = sim.amplitudes.forward(params)
dket = sim.amplitudes.grad(params)

# calculate the classical probabilities and their derivatives with respect to Φ
prob = sim.probabilities.forward(params)
dprob = sim.probabilities.grad(params)

# calculate the quantum and classical Fisher Information with respect to Φ
qfi = sim.amplitudes.qfim(params)
cfi = sim.probabilities.cfim(params)
```

### Acknowledgements

This work is supported by the Perimeter Institute Quantum Intelligence Lab, Institute for Quantum Computing, and Ki3 Photonics Technologies.

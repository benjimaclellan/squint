# squint

> can't see that star? squint a little harder 


## Installation
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

> Last updated: 2025-02-09

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import paramax
from rich.pretty import pprint

from squint.ops import BeamSplitter, Circuit, FockState, Phase
from squint.utils import print_nonzero_entries

jax.config.update("jax_enable_x64", True)

dim = 6
circuit = Circuit()

circuit.add(FockState(wires=(0,), n=(1,)))
circuit.add(FockState(wires=(1,), n=(1,)))
circuit.add(Phase(wires=(0,), phi=98 * jnp.pi / 100))
circuit.add(BeamSplitter(wires=(0, 1), r=jnp.pi / 0.4))
circuit.add(Phase(wires=(0,), phi=98 * jnp.pi / 100), "phase")
circuit.add(BeamSplitter(wires=(0, 1), r=20 * jnp.pi / 40))

params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)

sim = circuit.compile(params, static, cut=cutoff, optimize="greedy").jit()
pr = sim.probability(params)
print_nonzero_entries(pr)
```
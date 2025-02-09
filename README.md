# QUARK: Networked Quantum Sensing

> Simulate and optimize quantum sensor arrays.


## Installation
Simply clone the repo locally,
```bash
git clone https://github.com/benjimaclellan/qtelescope
```

Use a package manager tool, such as `uv` (installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/)).


Then, navigate to the repo,
```bash
cd qtelescope
```

Create a virtual environment and activate it,
```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install the package locally,
```bash
uv pip install -e .
```
This will install all dependencies and the package source in editable mode.


## Example

> Last updated: 2025-02-09

```python
from qtelescope.ops import BeamSplitter, Circuit, FockState, Phase, S2, create, destroy
from qtelescope.utils import partition_op, print_nonzero_entries

circuit = Circuit()
circuit.add(FockState(wires=(0,), n=(2,)))
circuit.add(FockState(wires=(1,), n=(2,)))
circuit.add(BeamSplitter(wires=(0, 1), r=jnp.pi/4))
circuit.add(Phase(wires=(0,), phi=jnp.pi/4), 'phase')
circuit.add(BeamSplitter(wires=(0, 1), r=jnp.pi/4))
pprint(circuit)
```
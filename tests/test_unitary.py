# %%
# %%

import equinox as eqx
import jax.numpy as jnp

from squint.circuit import Circuit
from squint.ops.base import Block, Wire
from squint.ops.fock import (
    BeamSplitter,
    FockState,
    Phase,
)

# %%

dim = 4  # Fock space cutoff
wires = tuple(Wire(dim=dim, idx=i) for i in range(4))
circuit = Circuit(backend="pure")

source = Block()
for w in wires:
    source.add(FockState(wires=(w,), n=(1,)))
source.add(Phase(wires=(wires[0],), phi=0.001), "phase")
circuit.add(source, "source")

detection = Block()
# for i, j in itertools.combinations(range(4), 2):
#     detection.add(BeamSplitter(wires=(wires[i], wires[j]), r=jnp.pi/4), f"a{i}{j}")
#     detection.add(Phase(wires=(wires[i],), phi=0.1), f"b{i}")
#     detection.add(Phase(wires=(wires[j],), phi=0.2), f"c{j}")
# circuit.add(detection, "detection")

detection.add(BeamSplitter(wires=(wires[0], wires[1]), r=jnp.pi / 4))
detection.add(BeamSplitter(wires=(wires[2], wires[3]), r=jnp.pi / 4))
detection.add(BeamSplitter(wires=(wires[0], wires[1]), r=jnp.pi / 4))
detection.add(BeamSplitter(wires=(wires[2], wires[3]), r=jnp.pi / 4))
detection.add(BeamSplitter(wires=(wires[2], wires[3]), r=jnp.pi / 4))
detection.add(BeamSplitter(wires=(wires[2], wires[3]), r=jnp.pi / 4))

circuit.add(detection, "detection")

eqx.tree_pprint(circuit, short_arrays=False)

# Print circuit subscripts (detection block doesn't have subscripts property)
print(circuit.subscripts)


# %%


# %%

import numpy as np


class Test:
    def __init__(self):
        self.a = np.array([[1, 2, 3], [4, 5, 6]])

    def __getitem__(self, key):
        return self.a[key]


t = Test()
print(t)

print(t.a)
print(t.a[0, 1:2:1])
print(t[0, 1:2:1])


# %%

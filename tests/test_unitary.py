#%%
# %%
import equinox as eqx
import jax.numpy as jnp
import ultraplot as uplt
import optax
import jax.random as jr
import itertools
from rich.pretty import pprint

from squint.circuit import Circuit, compile, subscripts_mixed, subscripts_pure
from squint.visualize import draw
from squint.ops.base import Block, dft, eye
from squint.ops.fock import (
    BeamSplitter,
    FixedEnergyFockState,
    FockState,
    TwoModeWeakThermalState,
    Phase,
    LinearOpticalUnitaryGate,
)
from squint.ops.noise import ErasureChannel
from squint.utils import partition_op, print_nonzero_entries, partition_by_leaves, partition_by_branches

#%%

wires = (0, 1, 2, 3)
circuit = Circuit(backend="pure")

source = Block()
for i in wires:
    source.add(FockState(wires=(i,), n=(1,)))
source.add(
    Phase(wires=(0,), phi=0.001),
    'phase'
)
circuit.add(source, "source")

detection = Block()
# for i, j in itertools.combinations(wires, 2):
#     detection.add(BeamSplitter(wires=(i, j), r=jnp.pi/4), f"a{i}{j}")
#     detection.add(Phase(wires=(i,), phi=0.1), f"b{i}")
#     detection.add(Phase(wires=(j,), phi=0.2), f"c{j}")
# circuit.add(detection, "detection")

detection.add(BeamSplitter(wires=(0, 1), r=jnp.pi/4))
detection.add(BeamSplitter(wires=(2, 3), r=jnp.pi/4))
detection.add(BeamSplitter(wires=(0, 1), r=jnp.pi/4))
detection.add(BeamSplitter(wires=(2, 3), r=jnp.pi/4))
detection.add(BeamSplitter(wires=(2, 3), r=jnp.pi/4))
detection.add(BeamSplitter(wires=(2, 3), r=jnp.pi/4))

circuit.add(detection, "detection")

eqx.tree_pprint(circuit, short_arrays=False)

# detection = circuit.ops['detection']

subscripts_pure(detection)


#%%



#%%

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


#%%
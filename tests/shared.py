#%%
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Int, ArrayLike

#%%
class Phase(eqx.Module):
    phi: Array
    def __init__(self, phi: float):
        self.phi = jnp.array(phi)
        
        
class TestShared(eqx.Module):
    shared: eqx.nn.Shared

    def __init__(self):
        n = 4
        phases = [Phase(phi=0.1 * i) for i in range(n)]
        
        # These two weights will now be tied together.
        # where = lambda shared: [shared[0].phi for _ in shared]
        where = lambda shared: [phase.phi for phase in shared[1:]]
        
        # get = lambda shared: [phase.phi for phase in shared]
        get = lambda shared: [shared[0].phi for phase in shared[1:]]
        self.shared = eqx.nn.Shared(phases, where, get)

    def __call__(self):
        # Expand back out so we can evaluate these layers.
        phases = self.shared()
        # assert a is b  # same parameter!
        # Now go ahead and evaluate your language model.
        print(phases)
        return 
    
    def unwrap(self):
        # Expand back out so we can evaluate these layers.
        phases = self.shared()
        # assert a is b  # same parameter!
        # Now go ahead and evaluate your language model.
        print(phases)
        return 
    
    
model = TestShared()
model.shared()[0]

#%%
params, static = eqx.partition(model, eqx.is_inexact_array)
print(params)
params = eqx.tree_at(lambda params: params.shared.pytree[0].phi, model, jnp.array(0.2))
print([params.shared()[i].phi for i in range(4)])
params()

# %%
import functools
import timeit

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import paramax
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.fock import BeamSplitter, FockState, Phase
from squint.ops.base import SharedGate
from squint.utils import partition_op

# %%  Express the optical circuit.
# ------------------------------------------------------------------
circuit = Circuit()
circuit.add(FockState(wires=(0,), n=(1,)))
circuit.add(FockState(wires=(1,), n=(1,)))
circuit.add(FockState(wires=(2,), n=(1,)))
# circuit.add(Phase(wires=(0,), phi=0.0), "phase")
phase = Phase(wires=(0,), phi=0.0)
circuit.add(SharedGate(main=phase, wires=(1, 2)), "phase")

pprint(circuit)
circuit.verify()

#%%
dim = 3
# [op for op in circuit.ops]
[op(dim=dim) for op_wrapped in circuit.ops.values() for op in op_wrapped.unwrap()]

# [op(dim=dim) for op_wrapped in circuit.ops.values() for op in op_wrapped.unwrap()]
# [op_unwrapped for op in circuit.ops for op_unwrapped in op]

# %%
params, static = eqx.partition(circuit, eqx.is_inexact_array)
print(params)

# %%

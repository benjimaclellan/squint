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
from squint.ops.dv import HGate, XGate, Conditional, ZGate, DiscreteState
from squint.ops.base import SharedGate
from squint.utils import partition_op, print_nonzero_entries

# %%  Express the optical circuit.
# ------------------------------------------------------------------
dim = 6

circuit = Circuit()
# circuit.add(FockState(wires=(0,), n=(1,)))
# circuit.add(FockState(wires=(1,), n=(1,)))
# circuit.add(FockState(wires=(2,), n=(1,)))
# phase = Phase(wires=(0,), phi=0.3)
# circuit.add(SharedGate(main=phase, wires=(1, 2)), "phase")

m = 5
for i in range(m):
    circuit.add(DiscreteState(wires=(i,)))
circuit.add(HGate(wires=(0,)))
for i in range(0, m-1):
    circuit.add(Conditional(gate=XGate, wires=(i, i+1)))

circuit.add(SharedGate(op=Phase(wires=(0,), phi=0.7), wires=tuple(range(1, m))), "phase")
    
for i in range(m):
    circuit.add(HGate(wires=(i,)))

pprint(circuit)
circuit.verify()

params, static = eqx.partition(circuit, eqx.is_inexact_array)
# pprint(params)
# pprint(static)

sim = circuit.compile(params, static, dim=dim)
pr = sim.probability(params)
print_nonzero_entries(pr)

#%%
params = eqx.tree_at(lambda params: params.ops['phase'].op.phi, params, jnp.array(jnp.pi/2))
pr = sim.probability(params)
# print_nonzero_entries(pr)

grads = sim.grad(params)
dpr = grads.ops['phase'].op.phi
print((dpr**2 / (pr + 1e-14)).sum())

# %%

#%%
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from loguru import logger
from rich.pretty import pprint
import paramax
from typing import Callable

from squint.circuit import Circuit
from squint.ops.base import SharedGate
from squint.ops.dv import Conditional, DiscreteState, HGate, Phase, XGate
from squint.ops.distributed import GlobalParameter
from squint.utils import print_nonzero_entries

#%%
circuit = Circuit()

m = 3

for i in range(m):
    circuit.add(DiscreteState(wires=(i,)))
    
circuit.add(HGate(wires=(0,)))

for i in range(m-1):
    circuit.add(Conditional(gate=XGate, wires=(i, i+1)))
    

dop = GlobalParameter(
    ops=[Phase(wires=(i,), phi=0.0) for i in range(m)],
    weights=jnp.ones(shape=(m,)) / m
)

shared = SharedGate(op=Phase(wires=(0,), phi=0.1), wires=tuple(range(1, m)))

circuit.add(shared, "phases")
# circuit.add(dop, "phases")

for i in range(m):
    circuit.add(HGate(wires=(i,)))

pprint(circuit)
#%%
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)

sim = circuit.compile(params, static, dim=2, optimize="greedy")



#%%
probs = sim.prob(params)
grads = sim.grad(params)
pprint(grads)

# give a lambda function which extracts the relevant params
# get = lambda params: jnp.array([op.phi for op in params.ops['phases'].ops])
get = lambda params: jnp.array([params.ops['phases'].op.phi])
get(params)

#%%
def classical_fisher_information(get: Callable, probs, grads):
    return jnp.einsum("i..., j..., ... -> ij", get(grads), get(grads), 1 / probs)

def quantum_fisher_information(get: Callable, amplitudes, )

cfim = classical_fisher_information(get, probs, grads)
pprint(cfim)

# %%

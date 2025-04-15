#%%import functools
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import polars as pl
import tqdm
from rich.pretty import pprint
from beartype import beartype
import einops 
import seaborn as sns

from squint.circuit import Circuit
from squint.ops.dv import DiscreteState, Phase, ZGate, HGate, Conditional, AbstractGate, RX, RY
from squint.utils import print_nonzero_entries

jax.config.update("jax_enable_x64", True)

#%%
n_phases = 1
n_qubit = 2
n_ancilla = 2

wires_ancilla = tuple(range(n_qubit, n_qubit + n_ancilla))
wires_qubit = tuple(range(0, n_qubit))

circuit = Circuit()
# for i in range(n_qubit):
circuit.add(
    DiscreteState(wires=wires_qubit, n=[(1.0, (0, 1)), (1.0, (1, 0))])
)
circuit.add(Phase(wires=(0,), phi=0.1), "phase")  # photon

for wire in wires_qubit:
    circuit.add(HGate(wires=(wire,)))
    
pprint(circuit)

params, static = eqx.partition(circuit, eqx.is_inexact_array)
pprint(params)

get = lambda pytree: jnp.array([pytree.ops["phase"].phi])
sim = circuit.compile(params, static, dim=2, optimize="greedy")
print_nonzero_entries(sim.amplitudes.forward(params))

print(sim.prob.cfim(get, params))
print(sim.amplitudes.qfim(get, params))

#%%
for i in range(n_qubit + n_ancilla):
    circuit.add(
        DiscreteState(wires=(i,))
    )
circuit.add(Phase(wires=(0,), phi=0.1), "phase")  # photon


# resource state prep
for k in range(4):
    for i in range(2, n_ancilla+2):
        circuit.add(RX(wires=(i,), phi=0.1))
        circuit.add(RY(wires=(i,), phi=0.1))
    for i in range(2, ):
params, static = eqx.partition(circuit, eqx.is_inexact_array)
pprint(params)

get = lambda pytree: jnp.array([pytree.ops["phase"].phi])
sim = circuit.compile(params, static, dim=2, optimize="greedy")
print_nonzero_entries(sim.amplitudes.forward(params))

print(sim.prob.cfim(get, params))
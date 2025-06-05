# %%
import equinox as eqx
import jax.numpy as jnp
import pytest
import jax 

from squint.circuit import Circuit, compile
from squint.ops.base import SharedGate, dft, eye
from squint.ops.fock import FockState, Phase, FixedEnergyFockState, LinearOpticalUnitaryGate
from squint.utils import partition_op, print_nonzero_entries
from squint.diagram import draw

import timeit
import functools

#%%
dim = 4
wires = (0, 1, 2)

circuit = Circuit(backend="pure")
state = FockState(wires=wires, n=((1, 0, 0)))
circuit.add(state)
n = sum(state.n[0][1])

unitary_modes = dft(len(wires))


op = LinearOpticalUnitaryGate(
    wires=wires,
    unitary_modes=unitary_modes
)

circuit.add(Phase(wires=(0, ), phi=0.0), 'phase')
circuit.add(op)

params, static = partition_op(circuit, 'phase')
sim = compile(
    static, dim, params, **{"optimize": "greedy", "argnum": 0}
).jit()

draw(circuit);

#%%
probs = sim.probabilities.forward(params)
grads = sim.probabilities.grad(params)
cfim = sim.probabilities.cfim(params)

nonzero_indices = jnp.array(jnp.nonzero(probs)).T
nonzero_values = probs[tuple(nonzero_indices.T)]

for idx, p in zip(nonzero_indices, nonzero_values, strict=True):
    print(idx, p)
    if p != 0:
        if jnp.sum(idx) != n:
            raise ValueError("Non-linear excitation index.")
        pass

assert jnp.isclose(jnp.sum(probs), 1.0), "Probabilities do not sum to 1."

# %%

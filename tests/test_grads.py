# %%
import equinox as eqx
import jax.numpy as jnp
import pytest
import functools
import jax
from typing import Literal, Sequence
from jaxtyping import PyTree
import paramax
import jax.tree_util as jtu

import squint
from squint.circuit import Circuit, get_symbol, compile_experimental
from squint.ops.base import SharedGate
from squint.ops.dv import Conditional, DiscreteVariableState, HGate, RZGate, XGate
from squint.ops.noise import BitFlipChannel, DepolarizingChannel, ErasureChannel
from squint.utils import partition_op
from squint.ops.base import AbstractErasureChannel

#%%
n = 2
dim = 2

circuit = Circuit(backend="pure")
for i in range(n):
    circuit.add(DiscreteVariableState(wires=(i,), n=(0,)))

circuit.add(HGate(wires=(0,)))
circuit.add(RZGate(wires=(0,), phi=0.0), "dummy0")
circuit.add(RZGate(wires=(1,), phi=0.0), "dummy1")
for i in range(n - 1):
    circuit.add(Conditional(gate=XGate, wires=(i, i + 1)))

circuit.add(
    SharedGate(op=RZGate(wires=(0,), phi=0.1 * jnp.pi), wires=tuple(range(1, n))),
    "phase",
)
for i in range(n):
    circuit.add(HGate(wires=(i,)))

#%%
params, static = eqx.partition(circuit, eqx.is_inexact_array)
params = (params,)

# params0, params1 = partition_op(params, "phase")
# params1, params2 = partition_op(params1, "dummy1")
# params = (params0, params1, params2)

#%%
sim = compile_experimental(static, dim, *params, **{'optimize': 'greedy', 'argnum': 0})

print(sim.amplitudes.forward(*params))
print(sim.probabilities.forward(*params))

print(jax.tree.flatten(sim.amplitudes.grad(*params)))
print(jax.tree.flatten(sim.probabilities.grad(*params)))

print(sim.probabilities.cfim(*params))
print(sim.amplitudes.qfim(*params))


# #%% 
# import optax 
# start_learning_rate = 1e-2

# optimizer = optax.chain(optax.adam(start_learning_rate), optax.scale(-1.0))

# opt_state = optimizer.init(params)

# @jax.jit
# def step(_params, _opt_state):
#     _val, _grad = value_and_grad(_params)
#     _updates, _opt_state = optimizer.update(_grad, _opt_state)
#     _params = optax.apply_updates(_params, _updates)
#     return _params, _opt_state, _val


# # %%
# cfims = []
# step(params, opt_state)
# for _ in range(300):
#     params, opt_state, val = step(params, opt_state)
#     cfims.append(val)
#     print(val, classical_fisher_information(params))
#     # print(params.ops['phase'].phi, params.ops[2].r, params.ops[4].r, cfim)  # params.ops[1].r,

# pr = sim.probability(params)
# print_nonzero_entries(pr)

# print(classical_fisher_information(params))

# %%
import functools
import timeit

import equinox as eqx
import jax.numpy as jnp
import paramax
from rich.pretty import pprint

from squint.ops import BeamSplitter, Circuit, FockState, Phase
from squint.utils import partition_op

# %%  Express the optical circuit.
# ------------------------------------------------------------------
cutoff = 4
circuit = Circuit()

m = 5
for i in range(m):
    circuit.add(FockState(wires=(i,), n=(1,)))
circuit.add(Phase(wires=(0,), phi=0.0), "phase")
for i in range(m - 1):
    circuit.add(BeamSplitter(wires=(i, i + 1), r=jnp.pi / 4))

pprint(circuit)

# %% split into training parameters and static
# ------------------------------------------------------------------
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)

sim = circuit.compile(params, static, cut=cutoff, optimize="greedy")
sim_jit = sim.jit()

# %%
tensor = sim.forward(params)
pr = sim.probability(params)
grad = sim.grad(params)
print(sim.info)

# %% Differentiate with respect to parameters of interest
name = "phase"
params, static = partition_op(circuit, name)
sim = circuit.compile(params, static, cut=cutoff, optimize="greedy")
sim_jit = sim.jit()

# %%
number = 10

times = timeit.Timer(functools.partial(sim.forward, params)).repeat(
    repeat=3, number=number
)
times_jit = timeit.Timer(functools.partial(sim_jit.forward, params)).repeat(
    repeat=3, number=number
)
print("State time non-JIT:", min(times), max(times))
print("State time JIT:", min(times_jit), max(times_jit))

times = timeit.Timer(functools.partial(sim.grad, params)).repeat(
    repeat=3, number=number
)
times_jit = timeit.Timer(functools.partial(sim_jit.grad, params)).repeat(
    repeat=3, number=number
)
print("Grad time non-JIT:", min(times), max(times))
print("Grad time JIT:", min(times_jit), max(times_jit))

times = timeit.Timer(functools.partial(sim.hess, params)).repeat(
    repeat=3, number=number
)
times_jit = timeit.Timer(functools.partial(sim_jit.hess, params)).repeat(
    repeat=3, number=number
)
print("Hess time non-JIT:", min(times), max(times))
print("Hess time JIT:", min(times_jit), max(times_jit))

# %%

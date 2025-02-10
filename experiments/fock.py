# %%
import functools
import timeit

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import paramax
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.fock import BeamSplitter, FockState, Phase, QFT
from squint.utils import partition_op, print_nonzero_entries

# %%  Express the optical circuit.
# ------------------------------------------------------------------
cutoff = 3
circuit = Circuit()

m = 2
# for i in range(m):
    # circuit.add(FockState(wires=(i,), n=(1,)))
circuit.add(FockState(wires=(0,), n=(0,)))
circuit.add(FockState(wires=(1,), n=(0,)))
circuit.add(FockState(wires=(2,), n=(1,)))
circuit.add(FockState(wires=(3,), n=(0,)))
circuit.add(FockState(wires=(4,), n=(0,)))
circuit.add(FockState(wires=(5,), n=(0,)))
circuit.add(QFT(wires=(0, 1, 2, 3, 4, 5), coeff=jnp.pi / 21))

pprint(circuit)
circuit.verify()

##%% split into training parameters and static
# ------------------------------------------------------------------
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    # is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)

sim = circuit.compile(params, static, dim=cutoff, optimize="greedy")
sim_jit = sim.jit()

##%%
tensor = sim.forward(params)
pr = sim.probability(params)
print_nonzero_entries(pr)

# %%
key = jr.PRNGKey(0)
samples = sim.sample(key, params, shape=(4, 5))
print(samples)

# %% Differentiate with respect to parameters of interest
name = "phase"
params, static = partition_op(circuit, name)
sim = circuit.compile(params, static, dim=cutoff, optimize="greedy")
sim_jit = sim.jit()

# %%
number = 1

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

samples = sim.sample(key, params, shape=(4, 5))

times = timeit.Timer(functools.partial(sim.sample, key, params, shape=(4, 5))).repeat(
    repeat=3, number=number
)
times_jit = timeit.Timer(functools.partial(sim_jit.sample, key, params, shape=(4, 5))).repeat(
    repeat=3, number=number
)
print("Sample time non-JIT:", min(times), max(times))
print("Sample time JIT:", min(times_jit), max(times_jit))

# %%

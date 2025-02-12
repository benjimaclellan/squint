# %%
# import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.fock import BeamSplitter, FockState, Phase, S2
from squint.utils import print_nonzero_entries
import tqdm
import matplotlib.pyplot as plt

# %%  Express the optical circuit.
# ------------------------------------------------------------------
cutoff = 7
circuit = Circuit()


circuit.add(FockState(wires=(0,), n=(1,)))
circuit.add(FockState(wires=(1,), n=(1,)))
circuit.add(S2(wires=(0, 1,), r=0.1, phi=0.1))
circuit.add(BeamSplitter(wires=(0, 1,)))
circuit.add(Phase(wires=(0,), phi=0.25 * jnp.pi), "phase")
circuit.add(S2(wires=(0, 1,), r=0.1, phi=0.0))
circuit.add(BeamSplitter(wires=(0, 1,)))

pprint(circuit)
circuit.verify()

#%%
##%% split into training parameters and static
# ------------------------------------------------------------------
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    # is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)
print(params)

#%%
sim = circuit.compile(params, static, dim=cutoff, optimize="greedy")
sim_jit = sim.jit()

##%%
tensor = sim.forward(params)
pr = sim.probability(params)
print_nonzero_entries(pr)

#%%
fig, ax = plt.subplots()
for idx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    ax.plot(pr[0,0])

# %%

pr = sim.probability(params)
grad = sim.grad(params)
cfi = (grad.ops["phase"].phi ** 2 / (pr + 1e-12)).sum()
print(cfi)

# %%
print(pr.sum())
# %%

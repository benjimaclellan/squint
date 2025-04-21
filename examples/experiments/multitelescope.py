#%%
#%%
import itertools
import functools 
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
from squint.ops.fock import FockState, BeamSplitter, Phase, FixedEnergyFockState, TwoModeWeakCoherentSource
from squint.ops.noise import ErasureChannel
from squint.utils import print_nonzero_entries

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'highest')

#%%

from squint.circuit import Circuit, AbstractPureState
# from squint.ops.dv import DiscreteState, Phase, ZGate, HGate, Conditional, AbstractGate, RX, RY
from squint.ops.fock import FockState, BeamSplitter, Phase, TwoModeWeakCoherentSource
from squint.utils import print_nonzero_entries
import itertools
from rich.pretty import pprint

#%%
n_phases = 1
wires_star = tuple(i for i in range(n_phases+1))
wires_lab = tuple(i for i in range(n_phases+1, 2*(n_phases+1)))

circuit = Circuit(backend='mixed')
circuit.add(
    TwoModeWeakCoherentSource(wires=(0, 1), epsilon=0.3, g=1.0, phi=0.2),
    "star"
)
# circuit.add(
#     FockState(
#         wires=wires_star,
#         n=[(1.0, tuple(1 if i == j else 0 for i in wires_star)) for j in wires_star]
#     )
# )
circuit.add(
    FockState(
        wires=wires_lab,
        n=[(1.0, tuple(1 if i == j else 0 for i in wires_lab)) for j in wires_lab]
    )
)
# for i in range(1, n_phases+1):
#     circuit.add(Phase(wires=(i,), phi=0.1), f"phase{i}") 

for wire_star, wire_lab in zip(wires_star, wires_lab):
    circuit.add(BeamSplitter(wires=(wire_star, wire_lab)))
    
# circuit.add(ErasureChannel(wires=(0,)))
pprint(circuit)
print(circuit.wires)

#%%
circuit.subscripts

#%%
params, static = eqx.partition(circuit, eqx.is_inexact_array)

tensors = circuit.evaluate(dim=2)
[tensor.shape for tensor in tensors]

#%%
path = circuit.path(dim=2, optimize="greedy")
print(path)    
get = lambda pytree: jnp.array([
    pytree.ops[f"star"].epsilon,
    pytree.ops[f"star"].g,
    pytree.ops[f"star"].phi,
    # pytree.ops[f"phase1"].phi,
])

#%%
sim = circuit.compile(params, static, dim=2, optimize="greedy")
probs = sim.prob.forward(params)
print(probs.sum())
pprint(sim.amplitudes.forward(params).shape)
#%%
print_nonzero_entries(probs)

#%%
sim.prob.grad(params)
cfim = sim.prob.cfim(get, params)
pprint(cfim)
    
# #%%
# op = FixedEnergyFockState(wires=(0, 1, 2), n=2)


# print((jnp.abs(op(dim=4))**2).sum())
# op(dim=4)
# %%
A = jnp.ones([4, 4])
v = jnp.ones([4])

print(A)
print(v)


jnp.matmul(A, v)

jnp.einsum("ij,j->i", A, v)

# %%

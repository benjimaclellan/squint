#%%
import itertools
import functools 

import einops
import equinox as eqx
import jax
import optax
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from rich.pretty import pprint
import polars as pl
from beartype import beartype
from opt_einsum.parser import get_symbol

from squint.ops.base import AbstractChannel, basis_operators, AbstractGate, AbstractState
from squint.ops.dv import DiscreteState, XGate, ZGate, Phase
from squint.circuit import Circuit
from squint.ops.noise import BitFlipChannel
from squint.utils import print_nonzero_entries

#%%
dim = 2

circuit = Circuit()
n = 1

for i in range(n):
    circuit.add(DiscreteState(wires=(i,)))
    circuit.add(XGate(wires=(i,)))
    # circuit.add(Phase(wires=(i,), phi=0.1))
    circuit.add(BitFlipChannel(wires=(i,), p=0.2))

#%%
path = circuit.path(dim=dim)
print(path)

#%%
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
)

sim = circuit.compile(params, static, dim=2)

#%%
print_nonzero_entries(sim.prob.forward(params))

# %%
# class TwoWireDepolarizing(AbstractChannel):
#     p: float 
    
#     @beartype
#     def __init__(
#         self,
#         wires: tuple[int, int],
#         p: float
#     ):
#         super().__init__(wires=wires)
#         self.p = p  #paramax.non_trainable(p)
#         return

#     def __call__(self, dim: int):
#         assert dim == 2
#         return jnp.array(
#             [
#                 jnp.sqrt(1-self.p) * jnp.einsum('ac, bd-> abcd', basis_operators(dim=2)[3], basis_operators(dim=2)[3]),   # identity
#                 jnp.sqrt(self.p/3) * jnp.einsum('ac, bd-> abcd', basis_operators(dim=2)[0], basis_operators(dim=2)[0]),     # X
#                 jnp.sqrt(self.p/3) * jnp.einsum('ac, bd-> abcd', basis_operators(dim=2)[1], basis_operators(dim=2)[1]),    # X
#                 jnp.sqrt(self.p/3) * jnp.einsum('ac, bd-> abcd', basis_operators(dim=2)[2], basis_operators(dim=2)[2]),    # X
#             ]
#         )


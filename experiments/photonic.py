#%%
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.dv import X, Z, DiscreteState
from squint.utils import print_nonzero_entries
import tqdm
import matplotlib.pyplot as plt

#%%
dim = 3
print(DiscreteState(wires=(0,), n=(1,)))
x = X(wires=(0,))
z = Z(wires=(0,))
print(x)
print(z)
print(x(dim=dim))
print(z(dim=dim))

circuit = Circuit()
circuit.add()

# %%


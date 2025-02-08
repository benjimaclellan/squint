#%%
import jax
import jax.numpy as jnp
import jax.random as jr
from string import ascii_letters, ascii_lowercase, ascii_uppercase
import copy
import equinox as eqx
import matplotlib.pyplot as plt
import paramax
import functools 
from rich.pretty import pprint
from beartype import beartype
import optax 
from typing import Sequence
from jaxtyping import ArrayLike, PyTree

from qtelescope.ops import *

#%%
ops = []

n = 3
for i in range(n):
    ops.append(FockState(wires=(i,), n=(1,)))
for i in range(n-1):
    ops.append(BeamSplitter(wires=(i, i+1)))
    # ops.append(AbstractGate(wires=(i,)))

circuit = Circuit(ops=ops)
pprint(circuit)
print(circuit.wires)


#%%
# todo: jit
cut = 4
path, info = jnp.einsum_path(circuit.subscripts, *[op(cut=cut) for op in circuit.ops], optimize=True)
print(info)

# %%
def _tensor_func(circ, subscripts: str, optimize: tuple):
    return jnp.einsum(subscripts, *[op(cut=cut) for op in circ.ops], optimize=optimize)
    
tensor_func = jax.jit(functools.partial(_tensor_func, subscripts=subscripts, optimize=path))

# %%
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)
print(params)

#%%
def forward(params, static):
    circuit = paramax.unwrap(eqx.combine(params, static))
    return tensor_func(circuit)

forward_jit = jax.jit(functools.partial(forward, static=static))

#%%
tensor = forward_jit(params)
print(tensor)

# %%
grads = jax.jacrev(functools.partial(forward, static=static))(params)
hess = jax.jacfwd(jax.jacrev(functools.partial(forward, static=static)))(params)
print(grads.ops[3].phi)

print(hess.ops[3].phi.ops[3].phi)

# grads

#%%
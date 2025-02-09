# %%
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import paramax
from rich.pretty import pprint

from qtelescope.ops import BeamSplitter, Circuit, FockState, Phase
from qtelescope.utils import partition_op

# %% express quantum optical circuit
cutoff = 4
circuit = Circuit(cutoff=cutoff)

n = 3
for i in range(n):
    circuit.add(FockState(wires=(i,), n=(1,)))

for i in range(n - 1):
    circuit.add(BeamSplitter(wires=(i, i + 1)))

circuit.add(Phase(wires=(0,), phi=0.2), "mark")
circuit.add(Phase(wires=(0,), phi=0.4), "mark2")

pprint(circuit.ops["mark"])
pprint(circuit)

# %%
# todo: jit
cut = 4
subscripts = circuit.subscripts
path, info = jnp.einsum_path(
    subscripts, *[op(cut=cut) for op in circuit.ops.values()], optimize=True
)
print(info)


# %%
def _tensor_func(circ, subscripts: str, optimize: tuple):
    return jnp.einsum(
        subscripts, *[op(cut=cut) for op in circ.ops.values()], optimize=optimize
    )


tensor_func = jax.jit(
    functools.partial(_tensor_func, subscripts=subscripts, optimize=path)
)

# %%
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)
print(params)


# %%
def forward(params, static):
    circuit = paramax.unwrap(eqx.combine(params, static))
    return tensor_func(circuit)


forward_jit = jax.jit(functools.partial(forward, static=static))

# %%
tensor = forward_jit(params)
print(tensor)

# %%
name = "mark2"
params, static = partition_op(circuit, name)

# %%
print(params)

# %%
grads = jax.jacrev(functools.partial(forward, static=static))(params)
hess = jax.jacfwd(jax.jacrev(functools.partial(forward, static=static)))(params)
print(grads)

print(hess.ops[name].phi.ops[name].phi)


# %%

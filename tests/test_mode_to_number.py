# %%
import equinox as eqx
import jax.numpy as jnp
import pytest

from squint.circuit import Circuit, compile
from squint.ops.base import SharedGate
from squint.ops.fock import FockState, FixedEnergyFockState, LinearOpticalUnitaryGate

#%%
dim = 4
n = 2
wires = (0, 1,)

circuit = Circuit(backend="pure")
circuit.add(FockState(wires=wires, n=(2, 1)))

U = jnp.sqrt(0.5) * jnp.array(
    [
        [1.0, -1.0], 
        [1.0, 1.0],
        # [1.0, -1.0, -1.0], 
        # [-1.0, 1.0, -1.0], 
        # [-1.0, -1.0, 1.0], 
    ]
)
U = jnp.sqrt(0.5) * jnp.array(
    [
        [1.0, 1.0j], 
        [1.0j, 1.0],
    ]
)


op = LinearOpticalUnitaryGate(
    wires=wires,
    unitary_modes=U
)
# op(dim)
print(U @ U.T.conj())

circuit.add(op)

params, static = eqx.partition(circuit, eqx.is_inexact_array)
sim = compile(
    static, dim, params, **{"optimize": "greedy", "argnum": 0}
)

probs = sim.probabilities.forward(params)

nonzero_indices = jnp.array(jnp.nonzero(probs)).T
nonzero_values = probs[tuple(nonzero_indices.T)]


for idx, p in zip(nonzero_indices, nonzero_values, strict=True):
    print(idx, p)
    # if p != 0:
    #     if jnp.sum(idx) != n:
    #         raise ValueError("Non-linear excitation index.")
    #     pass
print(jnp.sum(probs))
# assert jnp.isclose(jnp.sum(probs), 1.0), "Probabilities do not sum to 1."

# %%

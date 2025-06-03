# %%
import equinox as eqx
import jax.numpy as jnp
import pytest

from squint.circuit import Circuit, compile
from squint.ops.base import SharedGate
from squint.ops.fock import FockState, FixedEnergyFockState, LinearOpticalUnitaryGate

#%%

@pytest.mark.parametrize("n", [2, 3,])
def test_locc_op(n: int):
    circuit = Circuit(backend="pure")
    circuit.add(FockState(wires=tuple(range(n)), n=[1 for _ in range(n)]))
    circuit.add(
        LinearOpticalUnitaryGate(
            wires=tuple(range(n)),
            # rs=jnp.array([0.3, 0.0, 1.0]),
        )
    )

    dim = n + 1

    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    sim = compile(
        static, dim, params, **{"optimize": "greedy", "argnum": 0}
    )

    probs = sim.probabilities.forward(params)
    
    nonzero_indices = jnp.array(jnp.nonzero(probs)).T
    nonzero_values = probs[tuple(nonzero_indices.T)]
    
    
    for idx, p in zip(nonzero_indices, nonzero_values, strict=True):
        print(idx, p)
        if p != 0:
            if jnp.sum(idx) != n:
                raise ValueError("Non-linear excitation index.")
            pass
        
    assert jnp.isclose(jnp.sum(probs), 1.0), "Probabilities do not sum to 1."

# %%

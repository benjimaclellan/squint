# %%

import jax.numpy as jnp
import pytest

from squint.circuit import Circuit
from squint.ops.base import Wire, dft, eye
from squint.ops.fock import (
    FockState,
    LinearOpticalUnitaryGate,
    Phase,
)
from squint.utils import partition_op, print_nonzero_entries


# %%
@pytest.mark.parametrize("m", [2, 3, 4])
def test_qft_splitter_one_photon(m: int):
    dim = 3
    wires = tuple(Wire(dim=dim, idx=i) for i in range(m))

    ns = [tuple(jnp.zeros(m, dtype=jnp.int64).at[i].set(1).tolist()) for i in range(m)]
    for i in range(m):
        circuit = Circuit(backend="pure")
        state = FockState(
            wires=wires, n=(jnp.zeros(m, dtype=jnp.int64).at[i].set(1).tolist())
        )
        circuit.add(state)
        n = 1

        unitary_modes = dft(len(wires))

        op = LinearOpticalUnitaryGate(wires=wires, unitary_modes=unitary_modes)

        circuit.add(Phase(wires=(wires[0],), phi=0.0), "phase")
        circuit.add(op)

        params, static = partition_op(circuit, "phase")
        sim = circuit.compile(
            static, params, **{"optimize": "greedy", "argnum": 0}
        ).jit()

        probs = sim.probabilities.forward(params)

        nonzero_indices = jnp.array(jnp.nonzero(probs)).T
        nonzero_values = probs[tuple(nonzero_indices.T)]

        exact = jnp.zeros((dim,) * m).at[*ns].set(1 / m)
        assert jnp.allclose(exact, probs), (
            "Simulated probabilities do not match expected distributed (even probability)."
        )

        for idx, p in zip(nonzero_indices, nonzero_values, strict=True):
            print(idx, p)
            if p != 0:
                if jnp.sum(idx) != n:
                    raise ValueError("Non-linear excitation index.")
                pass

        assert jnp.isclose(jnp.sum(probs), 1.0), "Probabilities do not sum to 1."


@pytest.mark.parametrize("m", [2, 3, 4])
def test_identity(m: int):
    # TODO: failing test - work on LinearOpticalUnitaryGate is needed
    dim = 3
    m = 3
    wires = tuple(Wire(dim=dim, idx=i) for i in range(m))
    n = 1

    for i in range(m):
        circuit = Circuit(backend="pure")

        basis = jnp.zeros(m, dtype=jnp.int64).at[i].set(1).tolist()
        circuit.add(FockState(wires=wires, n=basis))
        circuit.add(Phase(wires=(wires[0],), phi=0.0), "phase")

        # params, static = partition_op(circuit, "phase")
        # sim = circuit.compile(static, params, **{"optimize": "greedy", "argnum": 0}).jit()

        # probs = sim.probabilities.forward(params)

        unitary_modes = eye(len(wires))

        op = LinearOpticalUnitaryGate(wires=wires, unitary_modes=unitary_modes)

        circuit.add(op)

        params, static = partition_op(circuit, "phase")
        sim = circuit.compile(
            static, params, **{"optimize": "greedy", "argnum": 0}
        ).jit()

        probs = sim.probabilities.forward(params)

        nonzero_indices = jnp.array(jnp.nonzero(probs)).T
        nonzero_values = probs[tuple(nonzero_indices.T)]

        exact = jnp.zeros((dim,) * m).at[*basis].set(1)
        assert jnp.allclose(exact, probs), (
            "Simulated probabilities do not match expected distributed (even probability)."
        )
        print_nonzero_entries(exact)
        print_nonzero_entries(probs)
        for idx, p in zip(nonzero_indices, nonzero_values, strict=True):
            print(idx, p)
            if p != 0:
                if jnp.sum(idx) != n:
                    raise ValueError("Non-linear excitation index.")
                pass

        assert jnp.isclose(jnp.sum(probs), 1.0), "Probabilities do not sum to 1."

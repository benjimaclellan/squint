# %%
import jax.numpy as jnp
import pytest

import jax

from squint import Circuit
from squint.interface.base import Block, SharedGate, Wire
from squint.interface.dv import (
    Conditional,
    CZGate,
    DiscreteVariableState,
    HGate,
    RXGate,
    RYGate,
    RZGate,
    XGate,
    x,
)
from squint.backends.tensornetwork.simulator import Simulator
from squint.math.information_matrices import quantum_fisher_information_matrix, classical_fisher_information_matrix
from squint.utils import partition_op


# %%
@pytest.mark.parametrize("n", [2, 3, 4])
def test_block_hl(n: int):
    wires = [Wire(dim=2, idx=i) for i in range(n)]

    circuit = Circuit()
    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

    block = Block()
    block.add(HGate(wires=(wires[0],)))
    for i in range(n - 1):
        block.add(Conditional(ufunc=x, wires=(wires[i], wires[i + 1])))
    circuit.add(block, "preparation")

    circuit.add(
        SharedGate(
            op=RZGate(wires=(wires[0],), phi=0.1 * jnp.pi), wires=tuple(wires[1:])
        ),
        "phase",
    )
    for w in wires:
        circuit.add(HGate(wires=(w,)))

    params, static = partition_op(circuit, "phase")

    sim = Simulator(static=static, params=params)

    def forward_probs(p):
        return jnp.abs(sim.forward(p))**2

    grad_probs = jax.jacfwd(forward_probs)

    qfi = quantum_fisher_information_matrix(sim.forward, sim.grad, params)
    cfi = classical_fisher_information_matrix(forward_probs, grad_probs, params)

    assert jnp.allclose(qfi, cfi)
    assert jnp.isclose(qfi.squeeze(), n**2)
    assert jnp.isclose(cfi.squeeze(), n**2)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_brickwork_blocks(n: int):
    from squint.blocks import brickwork

    wires = tuple(Wire(dim=2, idx=i) for i in range(n))
    block = brickwork(
        wires=wires,
        depth=2,
        LocalGates=(RXGate, RYGate, RZGate),
        CouplingGate=CZGate,
        periodic=True,
    )

    circuit = Circuit()
    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

    circuit.add(block, "brickwork")

    circuit.add(
        SharedGate(
            op=RZGate(wires=(wires[0],), phi=0.1 * jnp.pi), wires=tuple(wires[1:])
        ),
        "phase",
    )

    params, static = partition_op(circuit, "phase")

    sim = Simulator(static=static, params=params)

    def forward_probs(p):
        return jnp.abs(sim.forward(p))**2

    grad_probs = jax.jacfwd(forward_probs)

    qfi = quantum_fisher_information_matrix(sim.forward, sim.grad, params).squeeze()
    cfi = classical_fisher_information_matrix(forward_probs, grad_probs, params).squeeze()

    assert jnp.allclose(qfi, cfi)

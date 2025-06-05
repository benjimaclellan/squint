# %%
import jax.numpy as jnp
import pytest

from squint.circuit import Circuit
from squint.ops.base import Block, SharedGate
from squint.ops.dv import (
    Conditional,
    CZGate,
    DiscreteVariableState,
    HGate,
    RXGate,
    RYGate,
    RZGate,
    XGate,
)
from squint.utils import partition_op


# %%
@pytest.mark.parametrize("n", [2, 3, 4])
def test_block_hl(n: int):
    circuit = Circuit(backend="pure")
    for i in range(n):
        circuit.add(DiscreteVariableState(wires=(i,), n=(0,)))

    block = Block()
    block.add(HGate(wires=(0,)))
    for i in range(n - 1):
        block.add(Conditional(gate=XGate, wires=(i, i + 1)))
    circuit.add(block, "preparation")

    circuit.add(
        SharedGate(op=RZGate(wires=(0,), phi=0.1 * jnp.pi), wires=tuple(range(1, n))),
        "phase",
    )
    for i in range(n):
        circuit.add(HGate(wires=(i,)))

    circuit.unwrap()

    params, static = partition_op(circuit, "phase")

    sim = circuit.compile(static, 2, params)
    qfi = sim.amplitudes.qfim(params)
    cfi = sim.probabilities.cfim(params)

    assert jnp.allclose(qfi, cfi)
    assert jnp.isclose(qfi, n**2)
    assert jnp.isclose(cfi, n**2)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_brickwork_blocks(n: int):
    from squint.blocks import brickwork

    wires = tuple(range(n))
    block = brickwork(
        wires=wires,
        depth=2,
        LocalGates=(RXGate, RYGate, RZGate),
        CouplingGate=CZGate,
        periodic=True,
    )

    circuit = Circuit(backend="pure")
    for wire in wires:
        circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))

    circuit.add(block, "brickwork")

    circuit.add(
        SharedGate(op=RZGate(wires=(0,), phi=0.1 * jnp.pi), wires=tuple(range(1, n))),
        "phase",
    )

    params, static = partition_op(circuit, "phase")

    sim = circuit.compile(static, 2, params)
    qfi = sim.amplitudes.qfim(params).squeeze()
    cfi = sim.probabilities.cfim(params).squeeze()

    assert jnp.allclose(qfi, cfi)

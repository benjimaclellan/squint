# %%
import jax.numpy as jnp
import pytest

from squint.circuit import Circuit
from squint.ops.base import Block, SharedGate, Wire
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
from squint.simulator.tn import Simulator
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
        block.add(Conditional(gate=XGate, wires=(wires[i], wires[i + 1])))
    circuit.add(block, "preparation")

    circuit.add(
        SharedGate(
            op=RZGate(wires=(wires[0],), phi=0.1 * jnp.pi), wires=tuple(wires[1:])
        ),
        "phase",
    )
    for w in wires:
        circuit.add(HGate(wires=(w,)))

    circuit.unwrap()

    params, static = partition_op(circuit, "phase")

    sim = Simulator.compile(static, params)
    qfi = sim.amplitudes.qfim(params)
    cfi = sim.probabilities.cfim(params)

    assert jnp.allclose(qfi, cfi)
    assert jnp.isclose(qfi, n**2)
    assert jnp.isclose(cfi, n**2)


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

    sim = Simulator.compile(static, params)
    qfi = sim.amplitudes.qfim(params).squeeze()
    cfi = sim.probabilities.cfim(params).squeeze()

    assert jnp.allclose(qfi, cfi)

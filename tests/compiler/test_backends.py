"""Tests for pure vs mixed backend consistency."""

import equinox as eqx
import jax
import jax.numpy as jnp

from squint import Circuit
from squint.interface.base import Wire
from squint.interface.fock import (
    BeamSplitter,
    FockState,
    Phase,
)
from squint.backends.tensornetwork.simulator import Simulator
from squint.utils import partition_op


def _make_gjc_circuit(dim=3):
    wire0 = Wire(dim=dim, idx=0)
    wire1 = Wire(dim=dim, idx=1)
    wire2 = Wire(dim=dim, idx=2)
    wire3 = Wire(dim=dim, idx=3)

    circuit = Circuit()
    circuit.add(
        FockState(
            wires=(wire0, wire2),
            n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
        )
    )
    circuit.add(Phase(wires=(wire0,), phi=0.01), "phase")
    circuit.add(
        FockState(
            wires=(wire1, wire3),
            n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
        )
    )
    circuit.add(BeamSplitter(wires=(wire0, wire1)))
    circuit.add(BeamSplitter(wires=(wire2, wire3)))
    return circuit


def test_pure_vs_mixed_backend():
    """Probabilities are normalised across all phases for both backends."""
    circuit = _make_gjc_circuit()
    params, static = partition_op(circuit, "phase")
    sim = Simulator(static=static, params=params)
    sim.jit()

    phis = jnp.linspace(-jnp.pi, jnp.pi, 100)

    def update(phi, params):
        return eqx.tree_at(lambda pytree: pytree.ops["phase"].phi, params, phi)

    all_probs = jax.lax.map(
        lambda phi: jnp.abs(sim.forward(update(phi, params))) ** 2,
        phis,
    )

    # Every probability distribution should sum to 1
    norms = jnp.sum(all_probs.reshape(100, -1), axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-5)

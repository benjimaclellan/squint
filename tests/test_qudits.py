"""Tests for qudit (higher-dimensional) quantum systems."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from squint.circuit import Circuit
from squint.ops.base import Wire
from squint.ops.dv import DiscreteVariableState, HGate, RZGate
from squint.simulator.tn import Simulator


@pytest.mark.parametrize("dim", [2, 4, 6])
def test_qudit_circuit_runs(dim: int):
    """Test that qudit circuits compile and run for various dimensions."""
    wire = Wire(dim=dim, idx=0)

    circuit = Circuit()
    circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
    circuit.add(HGate(wires=(wire,)))
    circuit.add(RZGate(wires=(wire,), phi=0.1 * jnp.pi), "phase")
    circuit.add(HGate(wires=(wire,)))

    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    sim = Simulator.compile(static, params, optimize="greedy", argnum=0)

    # Test that forward pass produces valid amplitudes
    amplitudes = sim.amplitudes.forward(params)
    assert amplitudes.shape == (dim,), (
        f"Expected shape ({dim},), got {amplitudes.shape}"
    )

    # Test normalization - probabilities should sum to 1
    probs = sim.probabilities.forward(params)
    assert jnp.isclose(jnp.sum(probs), 1.0), (
        f"Probabilities should sum to 1, got {jnp.sum(probs)}"
    )

    # Test that QFIM and CFIM are computed without error
    qfim = sim.amplitudes.qfim(params)
    cfim = sim.probabilities.cfim(params)

    assert qfim.shape == (1, 1), f"QFIM shape should be (1, 1), got {qfim.shape}"
    assert cfim.shape == (1, 1), f"CFIM shape should be (1, 1), got {cfim.shape}"

    # Fisher information should be non-negative
    assert qfim.squeeze() >= 0, "QFIM should be non-negative"
    assert cfim.squeeze() >= 0, "CFIM should be non-negative"


def test_qudit_fisher_information_over_phase_range():
    """Test qudit circuit Fisher information varies correctly with phase."""
    dim = 6
    wire = Wire(dim=dim, idx=0)

    circuit = Circuit()
    circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
    circuit.add(HGate(wires=(wire,)))
    circuit.add(RZGate(wires=(wire,), phi=0.1 * jnp.pi), "phase")
    circuit.add(HGate(wires=(wire,)))

    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    sim = Simulator.compile(static, params, optimize="greedy", argnum=0)

    phis = jnp.linspace(-jnp.pi, jnp.pi, 50)
    params_batch = eqx.tree_at(lambda pytree: pytree.ops["phase"].phi, params, phis)

    probs = eqx.filter_vmap(sim.probabilities.forward)(params_batch)
    cfims = eqx.filter_vmap(sim.probabilities.cfim)(params_batch)
    qfims = eqx.filter_vmap(sim.amplitudes.qfim)(params_batch)

    # Check output shapes
    assert probs.shape == (50, dim), (
        f"Probs shape should be (50, {dim}), got {probs.shape}"
    )
    assert cfims.shape == (50, 1, 1), (
        f"CFIM shape should be (50, 1, 1), got {cfims.shape}"
    )
    assert qfims.shape == (50, 1, 1), (
        f"QFIM shape should be (50, 1, 1), got {qfims.shape}"
    )

    # All probabilities should sum to 1
    assert jnp.allclose(jnp.sum(probs, axis=1), 1.0), (
        "All probability distributions should sum to 1"
    )

    # Fisher information should be non-negative everywhere
    assert jnp.all(cfims >= 0), "CFIM should be non-negative for all phases"
    assert jnp.all(qfims >= 0), "QFIM should be non-negative for all phases"

    # QFIM should upper bound CFIM (quantum Cramer-Rao bound)
    assert jnp.all(qfims >= cfims - 1e-6), (
        "QFIM should be >= CFIM (quantum Cramer-Rao bound)"
    )

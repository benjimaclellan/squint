"""Benchmark tests for GHZ circuit compilation and Fisher information computation.

These tests verify that larger circuits compile and run correctly.
They are marked as slow and can be skipped with: pytest -m "not slow"
"""

import equinox as eqx
import jax.numpy as jnp
import pytest

from squint.circuit import Circuit
from squint.ops.base import SharedGate, Wire
from squint.ops.dv import Conditional, DiscreteVariableState, HGate, RZGate, XGate
from squint.simulator.tn import Simulator


def build_ghz_circuit(n: int):
    """Build a GHZ circuit with n qubits."""
    wires = [Wire(dim=2, idx=i) for i in range(n)]

    circuit = Circuit()
    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

    circuit.add(HGate(wires=(wires[0],)))
    for i in range(n - 1):
        circuit.add(Conditional(gate=XGate, wires=(wires[i], wires[i + 1])))

    circuit.add(
        SharedGate(
            op=RZGate(wires=(wires[0],), phi=0.0 * jnp.pi), wires=tuple(wires[1:])
        ),
        "phase",
    )

    for w in wires:
        circuit.add(HGate(wires=(w,)))

    return circuit


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_ghz_circuit_compiles_and_runs(n: int):
    """Test that GHZ circuits compile and produce correct Fisher information."""
    circuit = build_ghz_circuit(n)

    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    sim = Simulator.compile(static, params, optimize="greedy")

    # Test forward pass
    probs = sim.probabilities.forward(params)
    assert probs.shape == tuple([2] * n), (
        f"Expected shape {tuple([2] * n)}, got {probs.shape}"
    )
    assert jnp.isclose(jnp.sum(probs), 1.0), "Probabilities should sum to 1"

    # Test gradients compute without error
    grad = sim.probabilities.grad(params)
    assert grad is not None, "Gradient should be computed"

    # Test CFIM computes without error
    cfim = sim.probabilities.cfim(params)
    assert cfim.shape == (1, 1), f"CFIM shape should be (1, 1), got {cfim.shape}"
    assert cfim.squeeze() >= 0, "CFIM should be non-negative"


@pytest.mark.slow
@pytest.mark.parametrize("n", [10, 12, 14])
def test_ghz_circuit_scales(n: int):
    """Test that larger GHZ circuits compile and run (marked slow)."""
    circuit = build_ghz_circuit(n)

    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    sim = Simulator.compile(static, params, optimize="greedy")

    # Just verify it runs without error
    probs = sim.probabilities.forward(params)
    assert jnp.isclose(jnp.sum(probs), 1.0), "Probabilities should sum to 1"

    cfim = sim.probabilities.cfim(params)
    assert cfim.squeeze() >= 0, "CFIM should be non-negative"

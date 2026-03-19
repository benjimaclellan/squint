"""Tests for the tensor network Simulator."""

import jax.numpy as jnp
import pytest

from squint import Circuit
from squint.interface.base import Wire
from squint.interface.dv import DiscreteVariableState, HGate, RZGate, CXGate
from squint.interface.fock import FockState, Phase, BeamSplitter
from squint.backends.tensornetwork.simulator import Simulator
from squint.utils import partition_op


def test_forward_single_qubit():
    """Forward pass returns a normalised state vector."""
    wire = Wire(dim=2, idx=0)
    circuit = Circuit()
    circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
    circuit.add(HGate(wires=(wire,)))
    circuit.add(RZGate(wires=(wire,), phi=0.3), "phase")

    params, static = partition_op(circuit, "phase")
    sim = Simulator(static=static, params=params)
    state = sim.forward(params)

    assert state.shape == (2,)
    assert jnp.isclose(jnp.sum(jnp.abs(state) ** 2), 1.0)


def test_forward_two_qubit_bell():
    """Bell circuit produces a 2×2 amplitude tensor."""
    w0, w1 = Wire(dim=2, idx=0), Wire(dim=2, idx=1)
    circuit = Circuit()
    circuit.add(DiscreteVariableState(wires=(w0,), n=(0,)))
    circuit.add(DiscreteVariableState(wires=(w1,), n=(0,)))
    circuit.add(HGate(wires=(w0,)))
    circuit.add(CXGate(wires=(w0, w1)))
    circuit.add(RZGate(wires=(w0,), phi=0.0), "phase")

    params, static = partition_op(circuit, "phase")
    sim = Simulator(static=static, params=params)
    state = sim.forward(params)

    assert state.shape == (2, 2)
    assert jnp.isclose(jnp.sum(jnp.abs(state) ** 2), 1.0)


def test_grad_returns_pytree():
    """Grad returns a pytree with the same structure as params."""
    wire = Wire(dim=2, idx=0)
    circuit = Circuit()
    circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
    circuit.add(HGate(wires=(wire,)))
    circuit.add(RZGate(wires=(wire,), phi=0.3), "phase")

    params, static = partition_op(circuit, "phase")
    sim = Simulator(static=static, params=params)
    grad = sim.grad(params)

    # Gradient should have the same tree structure as params
    phi_grad = grad.ops["phase"].phi
    assert phi_grad is not None
    assert jnp.all(jnp.isfinite(phi_grad))


def test_jit_gives_same_result():
    """JIT-compiled forward matches non-JIT forward."""
    wire = Wire(dim=2, idx=0)
    circuit = Circuit()
    circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
    circuit.add(HGate(wires=(wire,)))
    circuit.add(RZGate(wires=(wire,), phi=0.5), "phase")

    params, static = partition_op(circuit, "phase")
    sim = Simulator(static=static, params=params)

    state_eager = sim.forward(params)
    sim.jit()
    state_jit = sim.forward(params)

    assert jnp.allclose(state_eager, state_jit)


def test_fock_forward():
    """Fock + BeamSplitter circuit produces a normalised amplitude tensor."""
    dim = 3
    w0, w1 = Wire(dim=dim, idx=0), Wire(dim=dim, idx=1)
    circuit = Circuit()
    circuit.add(FockState(wires=(w0, w1), n=(1, 0)))
    circuit.add(Phase(wires=(w0,), phi=0.1), "phase")
    circuit.add(BeamSplitter(wires=(w0, w1)))

    params, static = partition_op(circuit, "phase")
    sim = Simulator(static=static, params=params)
    state = sim.forward(params)

    assert state.shape == (dim, dim)
    assert jnp.isclose(jnp.sum(jnp.abs(state) ** 2), 1.0)

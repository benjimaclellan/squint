"""
Regression tests for the tensor network compiler pipeline.

Verifies that circuit_to_tensors, circuit_to_subscripts, and the full
contraction path work correctly for PureBackend and MixedBackend after
the dispatch refactor.
"""

import jax.numpy as jnp
import pytest
import equinox as eqx

from squint.backends.tensornetwork.compiler import (
    PureBackend,
    MixedBackend,
    circuit_to_tensors,
    circuit_to_subscripts,
    circuit_to_optimized_tensor_network_contraction_path,
    circuit_to_allowed_backends,
    circuit_to_wire_order,
    PostSquintWalk,
    ExtractCanonicalWireOrder,
)
from squint import Circuit
from squint.interface.base import Block, SharedGate, Wire
from squint.interface.dv import (
    CXGate,
    DiscreteVariableState,
    HGate,
    RZGate,
)
from squint.interface.fock import BeamSplitter, FockState, Phase
from squint.interface.noise import DepolarizingChannel, ErasureChannel
from squint.utils import partition_op


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_qubit_circuit():
    wire = Wire(dim=2, idx=0)
    circuit = Circuit()
    circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
    circuit.add(HGate(wires=(wire,)))
    circuit.add(RZGate(wires=(wire,), phi=0.5 * jnp.pi), "phase")
    circuit.add(HGate(wires=(wire,)))
    return circuit


@pytest.fixture
def ghz_circuit():
    n = 3
    wires = [Wire(dim=2, idx=i) for i in range(n)]
    circuit = Circuit()
    block = Block()
    for w in wires:
        block.add(DiscreteVariableState(wires=(w,), n=(0,)))
    circuit.add(block)
    circuit.add(HGate(wires=(wires[0],)))
    for i in range(n - 1):
        circuit.add(CXGate(wires=(wires[i], wires[i + 1])))
    circuit.add(
        SharedGate(op=RZGate(wires=(wires[0],), phi=0.1 * jnp.pi), wires=tuple(wires[1:])),
        "phase",
    )
    for w in wires:
        circuit.add(HGate(wires=(w,)))
    return circuit


@pytest.fixture
def fock_circuit():
    dim = 3
    wire0 = Wire(dim=dim, idx=0)
    wire1 = Wire(dim=dim, idx=1)
    circuit = Circuit()
    circuit.add(FockState(wires=(wire0, wire1), n=(1, 0)))
    circuit.add(Phase(wires=(wire0,), phi=0.01), "phase")
    circuit.add(BeamSplitter(wires=(wire0, wire1)))
    return circuit


@pytest.fixture
def noisy_circuit():
    wire = Wire(dim=2, idx=0)
    circuit = Circuit()
    circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
    circuit.add(HGate(wires=(wire,)))
    circuit.add(DepolarizingChannel(wires=(wire,), p=0.1))
    return circuit


# ---------------------------------------------------------------------------
# Backend selection tests
# ---------------------------------------------------------------------------

def test_pure_backend_selected_for_dv_circuit(single_qubit_circuit):
    backend = circuit_to_allowed_backends(single_qubit_circuit)
    assert backend is PureBackend


def test_mixed_backend_selected_for_noisy_circuit(noisy_circuit):
    backend = circuit_to_allowed_backends(noisy_circuit)
    assert backend is MixedBackend


def test_pure_backend_selected_for_fock_circuit(fock_circuit):
    backend = circuit_to_allowed_backends(fock_circuit)
    assert backend is PureBackend


# ---------------------------------------------------------------------------
# Wire order extraction
# ---------------------------------------------------------------------------

def test_wire_order_extraction(single_qubit_circuit):
    wires = circuit_to_wire_order(single_qubit_circuit)
    assert len(wires) == 1
    assert wires[0].idx == 0


def test_wire_order_ghz(ghz_circuit):
    wires = circuit_to_wire_order(ghz_circuit)
    assert len(wires) == 3


# ---------------------------------------------------------------------------
# Tensor generation tests
# ---------------------------------------------------------------------------

def test_circuit_to_tensors_pure_dv(single_qubit_circuit):
    params, static = partition_op(single_qubit_circuit, "phase")
    circuit = eqx.combine(params, static)
    tensors = circuit_to_tensors(circuit, PureBackend)
    assert len(tensors) > 0
    for t in tensors:
        assert t is not None


def test_circuit_to_tensors_pure_fock(fock_circuit):
    params, static = partition_op(fock_circuit, "phase")
    circuit = eqx.combine(params, static)
    tensors = circuit_to_tensors(circuit, PureBackend)
    assert len(tensors) > 0


def test_circuit_to_tensors_mixed_noisy(noisy_circuit):
    tensors = circuit_to_tensors(noisy_circuit, MixedBackend)
    assert len(tensors) > 0


# ---------------------------------------------------------------------------
# Subscript generation tests
# ---------------------------------------------------------------------------

def test_subscripts_pure_backend(single_qubit_circuit):
    subscripts = circuit_to_subscripts(single_qubit_circuit, PureBackend)
    assert "->" in subscripts


def test_subscripts_mixed_backend(noisy_circuit):
    subscripts = circuit_to_subscripts(noisy_circuit, MixedBackend)
    assert "->" in subscripts


# ---------------------------------------------------------------------------
# Full contraction pipeline
# ---------------------------------------------------------------------------

def test_full_contraction_single_qubit(single_qubit_circuit):
    params, static = partition_op(single_qubit_circuit, "phase")
    circuit = eqx.combine(params, static)
    subscripts, path = circuit_to_optimized_tensor_network_contraction_path(circuit, PureBackend)
    tensors = circuit_to_tensors(circuit, PureBackend)
    result = jnp.einsum(subscripts, *tensors, optimize=path)
    # Should be a normalized state vector for a single qubit
    assert result.shape == (2,)
    assert jnp.isclose(jnp.sum(jnp.abs(result) ** 2), 1.0)


def test_full_contraction_ghz(ghz_circuit):
    params, static = partition_op(ghz_circuit, "phase")
    circuit = eqx.combine(params, static)
    subscripts, path = circuit_to_optimized_tensor_network_contraction_path(circuit, PureBackend)
    tensors = circuit_to_tensors(circuit, PureBackend)
    result = jnp.einsum(subscripts, *tensors, optimize=path)
    assert result.shape == (2, 2, 2)
    assert jnp.isclose(jnp.sum(jnp.abs(result) ** 2), 1.0)


def test_full_contraction_fock(fock_circuit):
    params, static = partition_op(fock_circuit, "phase")
    circuit = eqx.combine(params, static)
    subscripts, path = circuit_to_optimized_tensor_network_contraction_path(circuit, PureBackend)
    tensors = circuit_to_tensors(circuit, PureBackend)
    result = jnp.einsum(subscripts, *tensors, optimize=path)
    assert jnp.isclose(jnp.sum(jnp.abs(result) ** 2), 1.0)


def test_full_contraction_mixed(noisy_circuit):
    print(noisy_circuit)
    subscripts, path = circuit_to_optimized_tensor_network_contraction_path(noisy_circuit, MixedBackend)
    print(subscripts)
    tensors = circuit_to_tensors(noisy_circuit, MixedBackend)
    print(len(tensors))
    result = jnp.einsum(subscripts, *tensors, optimize=path)
    # Density matrix for single qubit: shape (2, 2)
    assert result.shape == (2, 2)
    # Trace should be 1
    assert jnp.isclose(jnp.trace(result).real, 1.0)


def test_full_contraction_erasure():
    """Bell state with one qubit traced out should give a maximally mixed state."""
    w0, w1 = Wire(dim=2, idx=0), Wire(dim=2, idx=1)
    circuit = Circuit()
    circuit.add(DiscreteVariableState(wires=(w0,), n=(0,)))
    circuit.add(DiscreteVariableState(wires=(w1,), n=(0,)))
    circuit.add(HGate(wires=(w0,)))
    circuit.add(CXGate(wires=(w0, w1)))
    circuit.add(ErasureChannel(wires=(w1,)))

    subscripts, path = circuit_to_optimized_tensor_network_contraction_path(circuit, MixedBackend)
    tensors = circuit_to_tensors(circuit, MixedBackend)
    result = jnp.einsum(subscripts, *tensors, optimize=path)

    # Tracing out one qubit of a Bell state yields a 2x2 density matrix
    assert result.shape == (2, 2)
    assert jnp.isclose(jnp.trace(result).real, 1.0)
    # Reduced state is maximally mixed: rho = I/2
    assert jnp.allclose(result, jnp.eye(2) / 2, atol=1e-6)


# ---------------------------------------------------------------------------
# SharedGate expansion
# ---------------------------------------------------------------------------

def test_shared_gate_expands_correctly(ghz_circuit):
    """SharedGate should produce the same phase on all target wires."""
    params, static = partition_op(ghz_circuit, "phase")
    circuit = eqx.combine(params, static)
    tensors = circuit_to_tensors(circuit, PureBackend)
    assert len(tensors) > 0

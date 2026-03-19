"""
Tests for the backend dispatch mechanism.

Verifies that all DV ops correctly dispatch via plum's @dispatch to
lower(backend: TensorNetworkBackend), and that ops_for_backend discovery works.
"""

import jax.numpy as jnp
import pytest

from squint.interface.base import AbstractProcess, Wire
from squint.interface.dv import (
    CXGate,
    DiscreteVariableState,
    HGate,
    MaximallyMixedState,
    RXGate,
    RYGate,
    RZGate,
    XGate,
    ZGate,
)
from squint.backends.base import TensorNetworkBackend
from squint.backends.tensornetwork.compiler import PureBackend, MixedBackend


def ops_for_backend(backend_type: type) -> list[type]:
    """Return all registered AbstractProcess subclasses that implement lower() for the given backend type."""
    return [
        cls for cls in AbstractProcess._registry
        if hasattr(cls, 'lower') and any(
            backend_type in sig.signature.types
            for sig in cls.lower.methods
        )
    ]


# --- Discovery tests ---

def test_ops_registered_for_tensor_network_backend():
    ops = ops_for_backend(TensorNetworkBackend)
    assert DiscreteVariableState in ops
    assert HGate in ops
    assert RZGate in ops
    assert XGate in ops
    assert ZGate in ops


def test_pure_backend_is_tensor_network_backend():
    assert issubclass(PureBackend, TensorNetworkBackend)
    assert issubclass(MixedBackend, TensorNetworkBackend)


# --- Dispatch correctness tests for DV ops ---

def test_discrete_variable_state_dispatch():
    wire = Wire(dim=2, idx=0)
    state = DiscreteVariableState(wires=(wire,), n=(0,))
    backend = PureBackend()
    tensor = state(backend)
    assert tensor.shape == (2,)
    assert jnp.allclose(tensor, jnp.array([1.0, 0.0]))


def test_discrete_variable_state_excited():
    wire = Wire(dim=2, idx=0)
    state = DiscreteVariableState(wires=(wire,), n=(1,))
    backend = PureBackend()
    tensor = state(backend)
    assert jnp.allclose(tensor, jnp.array([0.0, 1.0]))


def test_hgate_dispatch():
    wire = Wire(dim=2, idx=0)
    gate = HGate(wires=(wire,))
    backend = PureBackend()
    tensor = gate(backend)
    assert tensor.shape == (2, 2)
    expected = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)
    assert jnp.allclose(tensor, expected)


def test_xgate_dispatch():
    wire = Wire(dim=2, idx=0)
    gate = XGate(wires=(wire,))
    backend = PureBackend()
    tensor = gate(backend)
    assert tensor.shape == (2, 2)
    expected = jnp.array([[0, 1], [1, 0]], dtype=jnp.float32)
    assert jnp.allclose(tensor, expected)


def test_zgate_dispatch():
    wire = Wire(dim=2, idx=0)
    gate = ZGate(wires=(wire,))
    backend = PureBackend()
    tensor = gate(backend)
    assert tensor.shape == (2, 2)
    expected = jnp.diag(jnp.array([1.0, -1.0], dtype=jnp.complex128))
    assert jnp.allclose(tensor, expected)


def test_rzgate_dispatch():
    wire = Wire(dim=2, idx=0)
    phi = jnp.pi / 2
    gate = RZGate(wires=(wire,), phi=phi)
    backend = PureBackend()
    tensor = gate(backend)
    assert tensor.shape == (2, 2)
    expected = jnp.diag(jnp.array([1.0, jnp.exp(1j * phi)]))
    assert jnp.allclose(tensor, expected)


def test_cxgate_dispatch():
    wire0 = Wire(dim=2, idx=0)
    wire1 = Wire(dim=2, idx=1)
    gate = CXGate(wires=(wire0, wire1))
    backend = PureBackend()
    tensor = gate(backend)
    assert tensor.shape == (2, 2, 2, 2)


def test_mixed_backend_dispatches_via_tensor_network():
    """MixedBackend is a TensorNetworkBackend, so ops dispatch to it correctly."""
    wire = Wire(dim=2, idx=0)
    state = MaximallyMixedState(wires=(wire,))
    backend = MixedBackend()
    tensor = state(backend)
    assert tensor.shape == (2, 2)
    expected = jnp.eye(2) / 2
    assert jnp.allclose(tensor, expected)

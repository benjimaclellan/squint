# Tests for all discrete variable (DV) operations

#%%
import equinox as eqx
import jax.numpy as jnp
import pytest

from squint.circuit import Circuit
from squint.ops.base import Wire
from squint.ops.dv import (
    CXGate,
    CZGate,
    Conditional,
    DiscreteVariableState,
    EmbeddedRGate,
    HGate,
    MaximallyMixedState,
    RXGate,
    RXXGate,
    RYGate,
    RZGate,
    RZZGate,
    TwoLocalHermitianBasisGate,
    XGate,
    ZGate,
)
#%%

# =============================================================================
# DiscreteVariableState Tests
# =============================================================================
class TestDiscreteVariableState:
    def test_basic_state_creation(self):
        """Test creating a basic |0> state."""
        wire = Wire(dim=2, idx=0)
        state = DiscreteVariableState(wires=(wire,), n=(0,))
        tensor = state()

        expected = jnp.array([1.0 + 0j, 0.0 + 0j])
        assert jnp.allclose(tensor, expected)

    def test_excited_state(self):
        """Test creating a |1> state."""
        wire = Wire(dim=2, idx=0)
        state = DiscreteVariableState(wires=(wire,), n=(1,))
        tensor = state()

        expected = jnp.array([0.0 + 0j, 1.0 + 0j])
        assert jnp.allclose(tensor, expected)

    def test_multi_wire_state(self):
        """Test creating a multi-wire state |01>."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        state = DiscreteVariableState(wires=(wire0, wire1), n=(0, 1))
        tensor = state()

        expected = jnp.zeros((2, 2), dtype=jnp.complex128)
        expected = expected.at[0, 1].set(1.0)
        assert jnp.allclose(tensor, expected)

    def test_superposition_state(self):
        """Test creating a superposition state (|0> + |1>)/sqrt(2)."""
        wire = Wire(dim=2, idx=0)
        state = DiscreteVariableState(
            wires=(wire,), n=[(1.0, (0,)), (1.0, (1,))]
        )
        tensor = state()

        # Should be normalized
        expected = jnp.array([1.0, 1.0]) / jnp.sqrt(2)
        assert jnp.allclose(tensor, expected)

    def test_qudit_state(self):
        """Test creating a state for a qutrit (dim=3)."""
        wire = Wire(dim=3, idx=0)
        state = DiscreteVariableState(wires=(wire,), n=(2,))
        tensor = state()

        expected = jnp.array([0.0 + 0j, 0.0 + 0j, 1.0 + 0j])
        assert jnp.allclose(tensor, expected)

    def test_default_state(self):
        """Test that default state is |0...0>."""
        wire = Wire(dim=2, idx=0)
        state = DiscreteVariableState(wires=(wire,))
        tensor = state()

        expected = jnp.array([1.0 + 0j, 0.0 + 0j])
        assert jnp.allclose(tensor, expected)

    def test_state_in_circuit(self):
        """Test using DiscreteVariableState in a circuit."""
        wire = Wire(dim=2, idx=0)
        circuit = Circuit(backend="pure")
        circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        expected = jnp.array([1.0 + 0j, 0.0 + 0j])
        assert jnp.allclose(amplitudes, expected)


# =============================================================================
# MaximallyMixedState Tests
# =============================================================================
class TestMaximallyMixedState:
    # Note: MaximallyMixedState has a bug where dims generator is exhausted.
    # These tests are marked as xfail until the bug is fixed.

    def test_single_qubit_maximally_mixed(self):
        """Test maximally mixed state for a single qubit."""
        wire = Wire(dim=2, idx=0)
        state = MaximallyMixedState(wires=(wire,))
        tensor = state()

        # Should be I/2 reshaped to (2, 2)
        expected = jnp.array([[0.5, 0.0], [0.0, 0.5]], dtype=jnp.complex128)
        assert jnp.allclose(tensor, expected)

    def test_two_qubit_maximally_mixed(self):
        """Test maximally mixed state for two qubits."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        state = MaximallyMixedState(wires=(wire0, wire1))
        tensor = state()

        # Should be I/4 reshaped to (2, 2, 2, 2)
        assert tensor.shape == (2, 2, 2, 2)
        # Check trace is 1
        trace = jnp.einsum("ijij->", tensor)
        assert jnp.isclose(trace, 1.0)

    def test_qutrit_maximally_mixed(self):
        """Test maximally mixed state for a qutrit."""
        wire = Wire(dim=3, idx=0)
        state = MaximallyMixedState(wires=(wire,))
        tensor = state()

        # Should be I/3
        expected_diag = 1.0 / 3.0
        assert jnp.isclose(tensor[0, 0], expected_diag)
        assert jnp.isclose(tensor[1, 1], expected_diag)
        assert jnp.isclose(tensor[2, 2], expected_diag)
        assert jnp.isclose(tensor[0, 1], 0.0)

    def test_maximally_mixed_in_circuit(self):
        """Test using MaximallyMixedState in a mixed circuit."""
        wire = Wire(dim=2, idx=0)
        circuit = Circuit(backend="mixed")
        circuit.add(MaximallyMixedState(wires=(wire,)))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        density = sim.amplitudes.forward(params)

        expected = jnp.array([[0.5, 0.0], [0.0, 0.5]], dtype=jnp.complex128)
        assert jnp.allclose(density, expected)


# =============================================================================
# XGate Tests
# =============================================================================
class TestXGate:
    def test_qubit_x_gate(self):
        """Test X gate for qubits is the Pauli-X matrix."""
        wire = Wire(dim=2, idx=0)
        gate = XGate(wires=(wire,))
        matrix = gate()

        expected = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        assert jnp.allclose(matrix, expected)

    def test_x_gate_unitarity(self):
        """Test that X gate is unitary."""
        wire = Wire(dim=2, idx=0)
        gate = XGate(wires=(wire,))
        matrix = gate()

        identity = jnp.eye(2)
        assert jnp.allclose(matrix @ matrix.conj().T, identity)

    def test_x_gate_flips_state(self):
        """Test that X gate flips |0> to |1>."""
        wire = Wire(dim=2, idx=0)
        circuit = Circuit(backend="pure")
        circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
        circuit.add(XGate(wires=(wire,)))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        expected = jnp.array([0.0 + 0j, 1.0 + 0j])
        assert jnp.allclose(amplitudes, expected)

    def test_qutrit_x_gate(self):
        """Test generalized X (shift) gate for qutrits."""
        wire = Wire(dim=3, idx=0)
        gate = XGate(wires=(wire,))
        matrix = gate()

        # X|0> = |1>, X|1> = |2>, X|2> = |0>
        expected = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float64)
        assert jnp.allclose(matrix, expected)


# =============================================================================
# ZGate Tests
# =============================================================================
class TestZGate:
    def test_qubit_z_gate(self):
        """Test Z gate for qubits is the Pauli-Z matrix."""
        wire = Wire(dim=2, idx=0)
        gate = ZGate(wires=(wire,))
        matrix = gate()

        expected = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        assert jnp.allclose(matrix, expected)

    def test_z_gate_unitarity(self):
        """Test that Z gate is unitary."""
        wire = Wire(dim=2, idx=0)
        gate = ZGate(wires=(wire,))
        matrix = gate()

        identity = jnp.eye(2)
        assert jnp.allclose(matrix @ matrix.conj().T, identity)

    def test_z_gate_phase_flip(self):
        """Test that Z gate applies phase to |1> state."""
        wire = Wire(dim=2, idx=0)
        circuit = Circuit(backend="pure")
        circuit.add(DiscreteVariableState(wires=(wire,), n=(1,)))
        circuit.add(ZGate(wires=(wire,)))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        expected = jnp.array([0.0 + 0j, -1.0 + 0j])
        assert jnp.allclose(amplitudes, expected)

    def test_qutrit_z_gate(self):
        """Test generalized Z (phase) gate for qutrits."""
        wire = Wire(dim=3, idx=0)
        gate = ZGate(wires=(wire,))
        matrix = gate()

        # Should be diagonal with phases exp(2*pi*i*k/3)
        omega = jnp.exp(2j * jnp.pi / 3)
        expected = jnp.diag(jnp.array([1.0, omega, omega**2]))
        assert jnp.allclose(matrix, expected)


# =============================================================================
# HGate Tests
# =============================================================================
class TestHGate:
    def test_qubit_h_gate(self):
        """Test H gate for qubits is the Hadamard matrix."""
        wire = Wire(dim=2, idx=0)
        gate = HGate(wires=(wire,))
        matrix = gate()

        expected = jnp.array([[1.0, 1.0], [1.0, -1.0]]) / jnp.sqrt(2)
        assert jnp.allclose(matrix, expected)

    def test_h_gate_unitarity(self):
        """Test that H gate is unitary."""
        wire = Wire(dim=2, idx=0)
        gate = HGate(wires=(wire,))
        matrix = gate()

        identity = jnp.eye(2)
        assert jnp.allclose(matrix @ matrix.conj().T, identity)

    def test_h_gate_creates_superposition(self):
        """Test that H gate creates |+> from |0>."""
        wire = Wire(dim=2, idx=0)
        circuit = Circuit(backend="pure")
        circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
        circuit.add(HGate(wires=(wire,)))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        expected = jnp.array([1.0, 1.0]) / jnp.sqrt(2)
        assert jnp.allclose(amplitudes, expected)

    def test_h_gate_self_inverse(self):
        """Test that H^2 = I."""
        wire = Wire(dim=2, idx=0)
        gate = HGate(wires=(wire,))
        matrix = gate()

        identity = jnp.eye(2)
        assert jnp.allclose(matrix @ matrix, identity)

    def test_qutrit_h_gate(self):
        """Test generalized H (DFT) gate for qutrits."""
        wire = Wire(dim=3, idx=0)
        gate = HGate(wires=(wire,))
        matrix = gate()

        # Should be the 3x3 DFT matrix
        omega = jnp.exp(2j * jnp.pi / 3)
        expected = (
            jnp.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, omega, omega**2],
                    [1.0, omega**2, omega**4],
                ]
            )
            / jnp.sqrt(3)
        )
        assert jnp.allclose(matrix, expected)


# =============================================================================
# RZGate Tests
# =============================================================================
class TestRZGate:
    def test_rz_zero_angle(self):
        """Test RZ(0) is identity."""
        wire = Wire(dim=2, idx=0)
        gate = RZGate(wires=(wire,), phi=0.0)
        matrix = gate()

        expected = jnp.eye(2)
        assert jnp.allclose(matrix, expected)

    def test_rz_pi_angle(self):
        """Test RZ(pi) applies correct phases."""
        wire = Wire(dim=2, idx=0)
        gate = RZGate(wires=(wire,), phi=jnp.pi)
        matrix = gate()

        expected = jnp.diag(jnp.array([1.0, -1.0]))
        assert jnp.allclose(matrix, expected)

    def test_rz_unitarity(self):
        """Test that RZ gate is unitary for arbitrary angle."""
        wire = Wire(dim=2, idx=0)
        gate = RZGate(wires=(wire,), phi=0.7)
        matrix = gate()

        identity = jnp.eye(2)
        assert jnp.allclose(matrix @ matrix.conj().T, identity)

    def test_rz_in_circuit(self):
        """Test RZ gate in a circuit."""
        wire = Wire(dim=2, idx=0)
        circuit = Circuit(backend="pure")
        circuit.add(DiscreteVariableState(wires=(wire,), n=(1,)))
        circuit.add(RZGate(wires=(wire,), phi=jnp.pi / 2))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        expected = jnp.array([0.0, jnp.exp(1j * jnp.pi / 2)])
        assert jnp.allclose(amplitudes, expected)

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_rz_qudit(self, dim):
        """Test RZ gate for qudits of various dimensions."""
        wire = Wire(dim=dim, idx=0)
        gate = RZGate(wires=(wire,), phi=0.5)
        matrix = gate()

        # Should be diagonal
        assert jnp.allclose(matrix, jnp.diag(jnp.diag(matrix)))
        # Should be unitary
        assert jnp.allclose(matrix @ matrix.conj().T, jnp.eye(dim))


# =============================================================================
# RXGate Tests
# =============================================================================
class TestRXGate:
    def test_rx_zero_angle(self):
        """Test RX(0) is identity."""
        wire = Wire(dim=2, idx=0)
        gate = RXGate(wires=(wire,), phi=0.0)
        matrix = gate()

        expected = jnp.eye(2)
        assert jnp.allclose(matrix, expected)

    def test_rx_pi_angle(self):
        """Test RX(pi) = -i*X."""
        wire = Wire(dim=2, idx=0)
        gate = RXGate(wires=(wire,), phi=jnp.pi)
        matrix = gate()

        expected = -1j * jnp.array([[0.0, 1.0], [1.0, 0.0]])
        assert jnp.allclose(matrix, expected)

    def test_rx_unitarity(self):
        """Test that RX gate is unitary."""
        wire = Wire(dim=2, idx=0)
        gate = RXGate(wires=(wire,), phi=1.2)
        matrix = gate()

        identity = jnp.eye(2)
        assert jnp.allclose(matrix @ matrix.conj().T, identity)

    def test_rx_flips_state(self):
        """Test RX(pi) flips |0> to |1> (up to global phase)."""
        wire = Wire(dim=2, idx=0)
        circuit = Circuit(backend="pure")
        circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
        circuit.add(RXGate(wires=(wire,), phi=jnp.pi))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        # |0> -> -i|1>
        expected = jnp.array([0.0, -1j])
        assert jnp.allclose(amplitudes, expected)


# =============================================================================
# RYGate Tests
# =============================================================================
class TestRYGate:
    def test_ry_zero_angle(self):
        """Test RY(0) is identity."""
        wire = Wire(dim=2, idx=0)
        gate = RYGate(wires=(wire,), phi=0.0)
        matrix = gate()

        expected = jnp.eye(2)
        assert jnp.allclose(matrix, expected)

    def test_ry_pi_angle(self):
        """Test RY(pi) = -i*Y."""
        wire = Wire(dim=2, idx=0)
        gate = RYGate(wires=(wire,), phi=jnp.pi)
        matrix = gate()

        expected = -1j * jnp.array([[0.0, -1j], [1j, 0.0]])
        assert jnp.allclose(matrix, expected)

    def test_ry_unitarity(self):
        """Test that RY gate is unitary."""
        wire = Wire(dim=2, idx=0)
        gate = RYGate(wires=(wire,), phi=0.8)
        matrix = gate()

        identity = jnp.eye(2)
        assert jnp.allclose(matrix @ matrix.conj().T, identity)

    def test_ry_creates_real_superposition(self):
        """Test RY(pi/2) creates equal real superposition from |0>."""
        wire = Wire(dim=2, idx=0)
        circuit = Circuit(backend="pure")
        circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
        circuit.add(RYGate(wires=(wire,), phi=jnp.pi / 2))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        # Should have equal magnitudes
        probs = jnp.abs(amplitudes) ** 2
        assert jnp.allclose(probs, jnp.array([0.5, 0.5]))


# =============================================================================
# Conditional Gate Tests
# =============================================================================
class TestConditional:
    def test_conditional_x_creates_cnot(self):
        """Test Conditional with XGate creates CNOT."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        gate = Conditional(gate=XGate, wires=(wire0, wire1))
        matrix = gate()

        # CNOT matrix in tensor form
        assert matrix.shape == (2, 2, 2, 2)
        # Test action on |00> -> |00>
        assert jnp.isclose(matrix[0, 0, 0, 0], 1.0)
        # Test action on |10> -> |11>
        assert jnp.isclose(matrix[1, 1, 1, 0], 1.0)

    def test_conditional_z_creates_cz(self):
        """Test Conditional with ZGate creates CZ."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        gate = Conditional(gate=ZGate, wires=(wire0, wire1))
        matrix = gate()

        assert matrix.shape == (2, 2, 2, 2)

    def test_cnot_entangles(self):
        """Test that CNOT creates entanglement."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)

        circuit = Circuit(backend="pure")
        circuit.add(DiscreteVariableState(wires=(wire0,), n=(0,)))
        circuit.add(DiscreteVariableState(wires=(wire1,), n=(0,)))
        circuit.add(HGate(wires=(wire0,)))
        circuit.add(Conditional(gate=XGate, wires=(wire0, wire1)))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        # Should create Bell state (|00> + |11>)/sqrt(2)
        expected = jnp.zeros((2, 2), dtype=jnp.complex128)
        expected = expected.at[0, 0].set(1 / jnp.sqrt(2))
        expected = expected.at[1, 1].set(1 / jnp.sqrt(2))
        assert jnp.allclose(amplitudes, expected)


# =============================================================================
# CXGate Tests
# =============================================================================
class TestCXGate:
    def test_cx_gate_creation(self):
        """Test CXGate can be created."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        gate = CXGate(wires=(wire0, wire1))
        matrix = gate()

        assert matrix.shape == (2, 2, 2, 2)

    def test_cx_equals_conditional_x(self):
        """Test CXGate equals Conditional with XGate."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)

        cx = CXGate(wires=(wire0, wire1))
        cond_x = Conditional(gate=XGate, wires=(wire0, wire1))

        assert jnp.allclose(cx(), cond_x())


# =============================================================================
# CZGate Tests
# =============================================================================
class TestCZGate:
    def test_cz_gate_creation(self):
        """Test CZGate can be created."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        gate = CZGate(wires=(wire0, wire1))
        matrix = gate()

        assert matrix.shape == (2, 2, 2, 2)

    def test_cz_equals_conditional_z(self):
        """Test CZGate equals Conditional with ZGate."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)

        cz = CZGate(wires=(wire0, wire1))
        cond_z = Conditional(gate=ZGate, wires=(wire0, wire1))

        assert jnp.allclose(cz(), cond_z())

    def test_cz_symmetric(self):
        """Test that CZ is symmetric (control/target interchangeable)."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)

        # CZ should give same result regardless of which qubit is control
        # This is because CZ only applies phase to |11>
        circuit1 = Circuit(backend="pure")
        circuit1.add(DiscreteVariableState(wires=(wire0, wire1), n=(1, 1)))
        circuit1.add(CZGate(wires=(wire0, wire1)))

        circuit2 = Circuit(backend="pure")
        circuit2.add(DiscreteVariableState(wires=(wire0, wire1), n=(1, 1)))
        circuit2.add(CZGate(wires=(wire1, wire0)))

        params1, static1 = eqx.partition(circuit1, eqx.is_inexact_array)
        params2, static2 = eqx.partition(circuit2, eqx.is_inexact_array)

        sim1 = circuit1.compile(static1, params1)
        sim2 = circuit2.compile(static2, params2)

        amp1 = sim1.amplitudes.forward(params1)
        amp2 = sim2.amplitudes.forward(params2)

        assert jnp.allclose(amp1, amp2)


# =============================================================================
# EmbeddedRGate Tests
# =============================================================================
class TestEmbeddedRGate:
    def test_embedded_r_identity(self):
        """Test EmbeddedRGate with theta=0 is identity on subspace."""
        wire = Wire(dim=3, idx=0)
        gate = EmbeddedRGate(wires=(wire,), levels=(0, 1), theta=0.0, phi=0.0)
        matrix = gate()

        expected = jnp.eye(3, dtype=jnp.complex128)
        assert jnp.allclose(matrix, expected)

    def test_embedded_r_unitarity(self):
        """Test that EmbeddedRGate is unitary."""
        wire = Wire(dim=3, idx=0)
        gate = EmbeddedRGate(wires=(wire,), levels=(0, 1), theta=0.5, phi=0.3)
        matrix = gate()

        identity = jnp.eye(3)
        assert jnp.allclose(matrix @ matrix.conj().T, identity)

    def test_embedded_r_different_levels(self):
        """Test EmbeddedRGate acting on different levels."""
        wire = Wire(dim=4, idx=0)
        gate = EmbeddedRGate(wires=(wire,), levels=(1, 2), theta=jnp.pi, phi=0.0)
        matrix = gate()

        # Should only affect levels 1 and 2
        assert jnp.isclose(matrix[0, 0], 1.0)
        assert jnp.isclose(matrix[3, 3], 1.0)
        # Should be unitary
        assert jnp.allclose(matrix @ matrix.conj().T, jnp.eye(4))


# =============================================================================
# RXXGate Tests
# =============================================================================
class TestRXXGate:
    def test_rxx_zero_angle(self):
        """Test RXX(0) is identity."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        gate = RXXGate(wires=(wire0, wire1), angle=0.0)
        matrix = gate()

        expected = jnp.eye(4).reshape(2, 2, 2, 2)
        assert jnp.allclose(matrix, expected)

    def test_rxx_unitarity(self):
        """Test that RXX gate is unitary."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        gate = RXXGate(wires=(wire0, wire1), angle=0.7)
        matrix = gate().reshape(4, 4)

        identity = jnp.eye(4)
        assert jnp.allclose(matrix @ matrix.conj().T, identity)

    def test_rxx_creates_entanglement(self):
        """Test that RXX can create entanglement."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)

        circuit = Circuit(backend="pure")
        circuit.add(DiscreteVariableState(wires=(wire0,), n=(0,)))
        circuit.add(DiscreteVariableState(wires=(wire1,), n=(0,)))
        circuit.add(RXXGate(wires=(wire0, wire1), angle=jnp.pi / 4))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        # Should create some entanglement (non-product state)
        # Check that amplitudes[0,0] and amplitudes[1,1] are non-zero
        assert jnp.abs(amplitudes[0, 0]) > 0.5
        assert jnp.abs(amplitudes[1, 1]) > 0.1


# =============================================================================
# RZZGate Tests
# =============================================================================
class TestRZZGate:
    def test_rzz_zero_angle(self):
        """Test RZZ(0) is identity."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        gate = RZZGate(wires=(wire0, wire1), angle=0.0)
        matrix = gate()

        expected = jnp.eye(4).reshape(2, 2, 2, 2)
        assert jnp.allclose(matrix, expected)

    def test_rzz_unitarity(self):
        """Test that RZZ gate is unitary."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        gate = RZZGate(wires=(wire0, wire1), angle=0.5)
        matrix = gate().reshape(4, 4)

        identity = jnp.eye(4)
        assert jnp.allclose(matrix @ matrix.conj().T, identity)

    def test_rzz_diagonal(self):
        """Test that RZZ is diagonal in computational basis."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        gate = RZZGate(wires=(wire0, wire1), angle=0.3)
        matrix = gate().reshape(4, 4)

        # RZZ should be diagonal
        off_diag = matrix - jnp.diag(jnp.diag(matrix))
        assert jnp.allclose(off_diag, 0.0)


# =============================================================================
# TwoLocalHermitianBasisGate Tests
# =============================================================================
class TestTwoLocalHermitianBasisGate:
    def test_two_local_unitarity(self):
        """Test that TwoLocalHermitianBasisGate is unitary."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        gate = TwoLocalHermitianBasisGate(
            wires=(wire0, wire1), angles=0.5, _basis_op_indices=(1, 1)
        )
        matrix = gate().reshape(4, 4)

        identity = jnp.eye(4)
        assert jnp.allclose(matrix @ matrix.conj().T, identity)

    def test_two_local_zero_angle(self):
        """Test that TwoLocalHermitianBasisGate with angle=0 is identity."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        gate = TwoLocalHermitianBasisGate(
            wires=(wire0, wire1), angles=0.0, _basis_op_indices=(2, 2)
        )
        matrix = gate()

        expected = jnp.eye(4).reshape(2, 2, 2, 2)
        assert jnp.allclose(matrix, expected)


# =============================================================================
# Integration Tests
# =============================================================================
class TestDVIntegration:
    def test_bell_state_probabilities(self):
        """Test that Bell state has correct measurement probabilities."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)

        circuit = Circuit(backend="pure")
        circuit.add(DiscreteVariableState(wires=(wire0,), n=(0,)))
        circuit.add(DiscreteVariableState(wires=(wire1,), n=(0,)))
        circuit.add(HGate(wires=(wire0,)))
        circuit.add(CXGate(wires=(wire0, wire1)))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        probs = sim.probabilities.forward(params)

        # Bell state: (|00> + |11>)/sqrt(2)
        # Probabilities: P(00) = P(11) = 0.5, P(01) = P(10) = 0
        assert jnp.isclose(probs[0, 0], 0.5)
        assert jnp.isclose(probs[1, 1], 0.5)
        assert jnp.isclose(probs[0, 1], 0.0)
        assert jnp.isclose(probs[1, 0], 0.0)

    def test_qft_circuit(self):
        """Test a simple QFT-like circuit."""
        wire = Wire(dim=2, idx=0)

        circuit = Circuit(backend="pure")
        circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
        circuit.add(HGate(wires=(wire,)))
        circuit.add(RZGate(wires=(wire,), phi=jnp.pi / 4))
        circuit.add(HGate(wires=(wire,)))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        # Should be normalized
        norm = jnp.sum(jnp.abs(amplitudes) ** 2)
        assert jnp.isclose(norm, 1.0)

    def test_mixed_backend_with_dv_state(self):
        """Test DV states work with mixed backend."""
        wire = Wire(dim=2, idx=0)

        circuit = Circuit(backend="mixed")
        circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
        circuit.add(HGate(wires=(wire,)))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        density = sim.amplitudes.forward(params)

        # |+><+| = [[0.5, 0.5], [0.5, 0.5]]
        expected = jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.complex128)
        assert jnp.allclose(density, expected)

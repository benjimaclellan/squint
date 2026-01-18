# Tests for all Fock space (continuous variable) operations
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from squint.circuit import Circuit
from squint.ops.base import Wire
from squint.ops.fock import (
    BeamSplitter,
    FixedEnergyFockState,
    FockState,
    LinearOpticalUnitaryGate,
    Phase,
    TwoModeSqueezingGate,
    TwoModeWeakThermalState,
)


# =============================================================================
# FockState Tests
# =============================================================================
class TestFockState:
    def test_vacuum_state(self):
        """Test creating the vacuum state |0>."""
        wire = Wire(dim=4, idx=0)
        state = FockState(wires=(wire,), n=(0,))
        tensor = state()

        expected = jnp.zeros(4, dtype=jnp.complex128)
        expected = expected.at[0].set(1.0)
        assert jnp.allclose(tensor, expected)

    def test_single_photon_state(self):
        """Test creating the single photon state |1>."""
        wire = Wire(dim=4, idx=0)
        state = FockState(wires=(wire,), n=(1,))
        tensor = state()

        expected = jnp.zeros(4, dtype=jnp.complex128)
        expected = expected.at[1].set(1.0)
        assert jnp.allclose(tensor, expected)

    def test_multi_photon_state(self):
        """Test creating a multi-photon state |3>."""
        wire = Wire(dim=5, idx=0)
        state = FockState(wires=(wire,), n=(3,))
        tensor = state()

        expected = jnp.zeros(5, dtype=jnp.complex128)
        expected = expected.at[3].set(1.0)
        assert jnp.allclose(tensor, expected)

    def test_two_mode_fock_state(self):
        """Test creating a two-mode Fock state |1,2>."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)
        state = FockState(wires=(wire0, wire1), n=(1, 2))
        tensor = state()

        expected = jnp.zeros((4, 4), dtype=jnp.complex128)
        expected = expected.at[1, 2].set(1.0)
        assert jnp.allclose(tensor, expected)

    def test_noon_state(self):
        """Test creating a NOON state (|N,0> + |0,N>)/sqrt(2)."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)
        state = FockState(
            wires=(wire0, wire1), n=[(1.0, (2, 0)), (1.0, (0, 2))]
        )
        tensor = state()

        # Should be normalized
        norm = jnp.sqrt(jnp.sum(jnp.abs(tensor) ** 2))
        assert jnp.isclose(norm, 1.0)

        # Check correct amplitudes
        assert jnp.isclose(jnp.abs(tensor[2, 0]), 1 / jnp.sqrt(2))
        assert jnp.isclose(jnp.abs(tensor[0, 2]), 1 / jnp.sqrt(2))

    def test_default_vacuum(self):
        """Test that default state is vacuum |0,...,0>."""
        wire0 = Wire(dim=3, idx=0)
        wire1 = Wire(dim=3, idx=1)
        state = FockState(wires=(wire0, wire1))
        tensor = state()

        expected = jnp.zeros((3, 3), dtype=jnp.complex128)
        expected = expected.at[0, 0].set(1.0)
        assert jnp.allclose(tensor, expected)

    def test_fock_state_normalization(self):
        """Test that superposition states are normalized."""
        wire = Wire(dim=4, idx=0)
        state = FockState(
            wires=(wire,), n=[(2.0, (0,)), (3.0, (1,)), (4.0, (2,))]
        )
        tensor = state()

        norm = jnp.sum(jnp.abs(tensor) ** 2)
        assert jnp.isclose(norm, 1.0)

    def test_fock_state_in_circuit(self):
        """Test using FockState in a circuit."""
        wire = Wire(dim=4, idx=0)
        circuit = Circuit(backend="pure")
        circuit.add(FockState(wires=(wire,), n=(1,)))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        expected = jnp.zeros(4, dtype=jnp.complex128)
        expected = expected.at[1].set(1.0)
        assert jnp.allclose(amplitudes, expected)


# =============================================================================
# FixedEnergyFockState Tests
# =============================================================================
class TestFixedEnergyFockState:
    def test_single_photon_two_modes(self):
        """Test fixed energy state with 1 photon in 2 modes."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)
        state = FixedEnergyFockState(wires=(wire0, wire1), n=1)

        # Should have 2 basis states: (1,0) and (0,1)
        assert len(state.bases) == 2
        assert (1, 0) in state.bases
        assert (0, 1) in state.bases

    def test_two_photons_two_modes(self):
        """Test fixed energy state with 2 photons in 2 modes."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)
        state = FixedEnergyFockState(wires=(wire0, wire1), n=2)

        # Should have 3 basis states: (2,0), (1,1), (0,2)
        assert len(state.bases) == 3
        assert (2, 0) in state.bases
        assert (1, 1) in state.bases
        assert (0, 2) in state.bases

    def test_three_modes_one_photon(self):
        """Test fixed energy state with 1 photon in 3 modes."""
        wires = tuple(Wire(dim=4, idx=i) for i in range(3))
        state = FixedEnergyFockState(wires=wires, n=1)

        # Should have 3 basis states
        assert len(state.bases) == 3

    def test_weights_and_phases_shape(self):
        """Test that weights and phases have correct shapes."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)
        state = FixedEnergyFockState(wires=(wire0, wire1), n=2)

        assert state.weights.shape == (3,)
        assert state.phases.shape == (3,)

    def test_random_initialization(self):
        """Test random initialization of weights and phases."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)
        key = jr.PRNGKey(42)
        state = FixedEnergyFockState(wires=(wire0, wire1), n=2, key=key)

        # Weights and phases should be non-uniform
        assert not jnp.allclose(state.weights, jnp.ones(3))
        assert not jnp.allclose(state.phases, jnp.zeros(3))


# =============================================================================
# TwoModeWeakThermalState Tests
# =============================================================================
class TestTwoModeWeakThermalState:
    def test_basic_creation(self):
        """Test creating a TwoModeWeakThermalState."""
        wire0 = Wire(dim=3, idx=0)
        wire1 = Wire(dim=3, idx=1)
        state = TwoModeWeakThermalState(
            wires=(wire0, wire1), epsilon=0.1, g=0.5, phi=0.0
        )
        tensor = state()

        assert tensor.shape == (3, 3, 3, 3)

    def test_trace_normalization(self):
        """Test that the state is properly normalized (trace = 1)."""
        wire0 = Wire(dim=3, idx=0)
        wire1 = Wire(dim=3, idx=1)
        state = TwoModeWeakThermalState(
            wires=(wire0, wire1), epsilon=0.1, g=0.5, phi=0.0
        )
        tensor = state()

        trace = jnp.einsum("ijij->", tensor)
        assert jnp.isclose(trace, 1.0)

    def test_zero_epsilon(self):
        """Test that epsilon=0 gives vacuum state."""
        wire0 = Wire(dim=3, idx=0)
        wire1 = Wire(dim=3, idx=1)
        state = TwoModeWeakThermalState(
            wires=(wire0, wire1), epsilon=0.0, g=0.5, phi=0.0
        )
        tensor = state()

        # Should be pure vacuum |00><00|
        assert jnp.isclose(tensor[0, 0, 0, 0], 1.0)

    def test_hermiticity(self):
        """Test that the density matrix is Hermitian."""
        wire0 = Wire(dim=3, idx=0)
        wire1 = Wire(dim=3, idx=1)
        state = TwoModeWeakThermalState(
            wires=(wire0, wire1), epsilon=0.1, g=0.5, phi=jnp.pi / 4
        )
        tensor = state()

        # Reshape to matrix form and check Hermiticity
        matrix = tensor.reshape(9, 9)
        assert jnp.allclose(matrix, matrix.conj().T)

    def test_in_mixed_circuit(self):
        """Test using TwoModeWeakThermalState in a mixed circuit."""
        wire0 = Wire(dim=3, idx=0)
        wire1 = Wire(dim=3, idx=1)

        circuit = Circuit(backend="mixed")
        circuit.add(
            TwoModeWeakThermalState(
                wires=(wire0, wire1), epsilon=0.1, g=0.8, phi=0.0
            )
        )

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        density = sim.amplitudes.forward(params)

        # Check trace is 1
        trace = jnp.einsum("ijij->", density)
        assert jnp.isclose(trace, 1.0)


# =============================================================================
# TwoModeSqueezingGate Tests
# =============================================================================
# class TestTwoModeSqueezingGate:
#     def test_basic_creation(self):
#         """Test creating a TwoModeSqueezingGate."""
#         wire0 = Wire(dim=4, idx=0)
#         wire1 = Wire(dim=4, idx=1)
#         gate = TwoModeSqueezingGate(wires=(wire0, wire1), r=0.5, phi=0.0)
#         matrix = gate()

#         assert matrix.shape == (4, 4, 4, 4)

#     def test_zero_squeezing(self):
#         """Test that r=0 gives identity."""
#         wire0 = Wire(dim=4, idx=0)
#         wire1 = Wire(dim=4, idx=1)
#         gate = TwoModeSqueezingGate(wires=(wire0, wire1), r=0.0, phi=0.0)
#         matrix = gate()

#         expected = jnp.eye(16).reshape(4, 4, 4, 4)
#         assert jnp.allclose(matrix, expected)

#     def test_unitarity(self):
#         """Test that the squeezing gate is unitary."""
#         wire0 = Wire(dim=4, idx=0)
#         wire1 = Wire(dim=4, idx=1)
#         gate = TwoModeSqueezingGate(wires=(wire0, wire1), r=0.3, phi=0.5)
#         matrix = gate().reshape(16, 16)

#         identity = jnp.eye(16)
#         assert jnp.allclose(matrix @ matrix.conj().T, identity, atol=1e-6)


# =============================================================================
# BeamSplitter Tests
# =============================================================================
class TestBeamSplitter:
    def test_basic_creation(self):
        """Test creating a BeamSplitter."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)
        bs = BeamSplitter(wires=(wire0, wire1), r=jnp.pi / 4)
        matrix = bs()

        assert matrix.shape == (4, 4, 4, 4)

    def test_fifty_fifty_splitter(self):
        """Test 50:50 beam splitter (r=pi/4)."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)
        bs = BeamSplitter(wires=(wire0, wire1), r=jnp.pi / 4)

        circuit = Circuit(backend="pure")
        circuit.add(FockState(wires=(wire0,), n=(1,)))
        circuit.add(FockState(wires=(wire1,), n=(0,)))
        circuit.add(bs)

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        # |1,0> should split to (|1,0> + i|0,1>)/sqrt(2)
        # Check probabilities are 0.5 each
        probs = jnp.abs(amplitudes) ** 2
        assert jnp.isclose(probs[1, 0], 0.5, atol=1e-5)
        assert jnp.isclose(probs[0, 1], 0.5, atol=1e-5)

    def test_zero_angle_identity(self):
        """Test beam splitter with r=0 is identity."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)
        bs = BeamSplitter(wires=(wire0, wire1), r=0.0)
        matrix = bs()

        expected = jnp.eye(16).reshape(4, 4, 4, 4)
        assert jnp.allclose(matrix, expected, atol=1e-6)

    def test_unitarity(self):
        """Test that beam splitter is unitary."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)
        bs = BeamSplitter(wires=(wire0, wire1), r=0.3)
        matrix = bs().reshape(16, 16)

        identity = jnp.eye(16)
        assert jnp.allclose(matrix @ matrix.conj().T, identity, atol=1e-6)

    def test_photon_number_conservation(self):
        """Test that beam splitter conserves total photon number."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)

        circuit = Circuit(backend="pure")
        circuit.add(FockState(wires=(wire0, wire1), n=(2, 1)))  # 3 photons total
        circuit.add(BeamSplitter(wires=(wire0, wire1), r=0.7))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        probs = sim.probabilities.forward(params)

        # Sum probabilities for all states with total photon number = 3
        total_prob_3_photons = 0.0
        for n1 in range(4):
            for n2 in range(4):
                if n1 + n2 == 3:
                    total_prob_3_photons += probs[n1, n2]

        assert jnp.isclose(total_prob_3_photons, 1.0, atol=1e-5)

    def test_hom_interference(self):
        """Test Hong-Ou-Mandel interference with 50:50 beam splitter."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)

        circuit = Circuit(backend="pure")
        # Two single photons input
        circuit.add(FockState(wires=(wire0, wire1), n=(1, 1)))
        circuit.add(BeamSplitter(wires=(wire0, wire1), r=jnp.pi / 4))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        probs = sim.probabilities.forward(params)

        # HOM effect: both photons should exit together
        # P(1,1) should be 0, P(2,0) = P(0,2) = 0.5
        assert jnp.isclose(probs[1, 1], 0.0, atol=1e-5)
        assert jnp.isclose(probs[2, 0], 0.5, atol=1e-5)
        assert jnp.isclose(probs[0, 2], 0.5, atol=1e-5)


# =============================================================================
# Phase Tests
# =============================================================================
class TestPhase:
    def test_zero_phase(self):
        """Test that Phase(0) is identity."""
        wire = Wire(dim=4, idx=0)
        gate = Phase(wires=(wire,), phi=0.0)
        matrix = gate()

        expected = jnp.eye(4)
        assert jnp.allclose(matrix, expected)

    def test_phase_diagonal(self):
        """Test that Phase gate is diagonal."""
        wire = Wire(dim=4, idx=0)
        gate = Phase(wires=(wire,), phi=0.5)
        matrix = gate()

        # Should be diagonal
        off_diag = matrix - jnp.diag(jnp.diag(matrix))
        assert jnp.allclose(off_diag, 0.0)

    def test_phase_unitarity(self):
        """Test that Phase gate is unitary."""
        wire = Wire(dim=4, idx=0)
        gate = Phase(wires=(wire,), phi=1.2)
        matrix = gate()

        identity = jnp.eye(4)
        assert jnp.allclose(matrix @ matrix.conj().T, identity)

    def test_phase_on_fock_state(self):
        """Test phase gate action on Fock states."""
        wire = Wire(dim=4, idx=0)

        circuit = Circuit(backend="pure")
        circuit.add(FockState(wires=(wire,), n=(2,)))
        circuit.add(Phase(wires=(wire,), phi=jnp.pi / 2))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        # |2> -> exp(i*2*pi/2)|2> = -|2>
        expected = jnp.zeros(4, dtype=jnp.complex128)
        expected = expected.at[2].set(jnp.exp(1j * 2 * jnp.pi / 2))
        assert jnp.allclose(amplitudes, expected)

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_phase_eigenvalues(self, n):
        """Test that |n> is eigenstate of Phase with eigenvalue exp(i*n*phi)."""
        wire = Wire(dim=4, idx=0)
        phi = 0.7

        circuit = Circuit(backend="pure")
        circuit.add(FockState(wires=(wire,), n=(n,)))
        circuit.add(Phase(wires=(wire,), phi=phi))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        expected_phase = jnp.exp(1j * n * phi)
        assert jnp.isclose(amplitudes[n], expected_phase)


# =============================================================================
# LinearOpticalUnitaryGate Tests
# =============================================================================
# class TestLinearOpticalUnitaryGate:
#     def test_identity_unitary(self):
#         """Test that identity mode unitary gives identity gate."""
#         wire0 = Wire(dim=4, idx=0)
#         wire1 = Wire(dim=4, idx=1)

#         U = jnp.eye(2, dtype=jnp.complex128)
#         gate = LinearOpticalUnitaryGate(wires=(wire0, wire1), unitary_modes=U)
#         matrix = gate()

#         expected = jnp.eye(16).reshape(4, 4, 4, 4)
#         assert jnp.allclose(matrix, expected, atol=1e-6)

#     def test_hadamard_mode_unitary(self):
#         """Test Hadamard-like mode transformation."""
#         wire0 = Wire(dim=4, idx=0)
#         wire1 = Wire(dim=4, idx=1)

#         U = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)
#         gate = LinearOpticalUnitaryGate(wires=(wire0, wire1), unitary_modes=U)
#         matrix = gate()

#         assert matrix.shape == (4, 4, 4, 4)

#     def test_unitarity(self):
#         """Test that LinearOpticalUnitaryGate produces unitary transformation."""
#         wire0 = Wire(dim=4, idx=0)
#         wire1 = Wire(dim=4, idx=1)

#         # Random unitary
#         U = jnp.array(
#             [[jnp.cos(0.3), jnp.sin(0.3)], [-jnp.sin(0.3), jnp.cos(0.3)]],
#             dtype=jnp.complex128,
#         )
#         gate = LinearOpticalUnitaryGate(wires=(wire0, wire1), unitary_modes=U)
#         matrix = gate().reshape(16, 16)

#         identity = jnp.eye(16)
#         assert jnp.allclose(matrix @ matrix.conj().T, identity, atol=1e-5)

#     def test_single_photon_transformation(self):
#         """Test single photon transformation matches mode unitary."""
#         wire0 = Wire(dim=3, idx=0)
#         wire1 = Wire(dim=3, idx=1)

#         # Beam splitter-like transformation
#         theta = jnp.pi / 3
#         U = jnp.array(
#             [[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]],
#             dtype=jnp.complex128,
#         )

#         circuit = Circuit(backend="pure")
#         circuit.add(FockState(wires=(wire0, wire1), n=(1, 0)))
#         circuit.add(LinearOpticalUnitaryGate(wires=(wire0, wire1), unitary_modes=U))

#         params, static = eqx.partition(circuit, eqx.is_inexact_array)
#         sim = circuit.compile(static, params)
#         amplitudes = sim.amplitudes.forward(params)

#         # |1,0> -> cos(theta)|1,0> - sin(theta)|0,1>
#         assert jnp.isclose(jnp.abs(amplitudes[1, 0]) ** 2, jnp.cos(theta) ** 2, atol=1e-5)
#         assert jnp.isclose(jnp.abs(amplitudes[0, 1]) ** 2, jnp.sin(theta) ** 2, atol=1e-5)

#     def test_three_mode_unitary(self):
#         """Test three-mode linear optical unitary."""
#         wires = tuple(Wire(dim=3, idx=i) for i in range(3))

#         # DFT matrix
#         omega = jnp.exp(2j * jnp.pi / 3)
#         U = (
#             jnp.array(
#                 [
#                     [1, 1, 1],
#                     [1, omega, omega**2],
#                     [1, omega**2, omega**4],
#                 ],
#                 dtype=jnp.complex128,
#             )
#             / jnp.sqrt(3)
#         )

#         gate = LinearOpticalUnitaryGate(wires=wires, unitary_modes=U)
#         matrix = gate()

#         assert matrix.shape == (3, 3, 3, 3, 3, 3)


# =============================================================================
# Integration Tests
# =============================================================================
class TestFockIntegration:
    def test_mach_zehnder_interferometer(self):
        """Test a Mach-Zehnder interferometer circuit."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)

        circuit = Circuit(backend="pure")
        circuit.add(FockState(wires=(wire0,), n=(1,)))
        circuit.add(FockState(wires=(wire1,), n=(0,)))
        # First beam splitter
        circuit.add(BeamSplitter(wires=(wire0, wire1), r=jnp.pi / 4))
        # Phase shift
        circuit.add(Phase(wires=(wire0,), phi=jnp.pi / 2))
        # Second beam splitter
        circuit.add(BeamSplitter(wires=(wire0, wire1), r=jnp.pi / 4))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        probs = sim.probabilities.forward(params)

        # Total probability should be 1
        total_prob = jnp.sum(probs)
        assert jnp.isclose(total_prob, 1.0)

    def test_mixed_backend_with_fock_state(self):
        """Test Fock states work with mixed backend."""
        wire = Wire(dim=4, idx=0)

        circuit = Circuit(backend="mixed")
        circuit.add(FockState(wires=(wire,), n=(1,)))
        circuit.add(Phase(wires=(wire,), phi=0.5))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        density = sim.amplitudes.forward(params)

        # Should be pure state |1><1| (phase doesn't affect diagonal)
        expected = jnp.zeros((4, 4), dtype=jnp.complex128)
        expected = expected.at[1, 1].set(1.0)
        assert jnp.allclose(density, expected)

    def test_beam_splitter_chain(self):
        """Test multiple beam splitters in series."""
        wires = tuple(Wire(dim=3, idx=i) for i in range(4))

        circuit = Circuit(backend="pure")
        for i, w in enumerate(wires):
            circuit.add(FockState(wires=(w,), n=(1 if i == 0 else 0,)))

        # Chain of beam splitters
        for i in range(3):
            circuit.add(BeamSplitter(wires=(wires[i], wires[i + 1]), r=jnp.pi / 4))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        probs = sim.probabilities.forward(params)

        # Photon should be distributed across modes
        # Total probability should be 1
        total_prob = jnp.sum(probs)
        assert jnp.isclose(total_prob, 1.0)

        # Check photon number conservation (1 photon total)
        one_photon_prob = 0.0
        for idx in range(4):
            indices = [0] * 4
            indices[idx] = 1
            one_photon_prob += probs[tuple(indices)]
        assert jnp.isclose(one_photon_prob, 1.0, atol=1e-5)

    def test_noon_state_interferometry(self):
        """Test NOON state through a phase shift."""
        wire0 = Wire(dim=4, idx=0)
        wire1 = Wire(dim=4, idx=1)

        phi = 0.3

        circuit = Circuit(backend="pure")
        # Create NOON state |2,0> + |0,2>
        circuit.add(
            FockState(
                wires=(wire0, wire1), n=[(1.0, (2, 0)), (1.0, (0, 2))]
            )
        )
        # Apply phase to first mode
        circuit.add(Phase(wires=(wire0,), phi=phi))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(static, params)
        amplitudes = sim.amplitudes.forward(params)

        # |2,0> gets phase exp(2i*phi), |0,2> is unchanged
        # Check relative phase magnitude (sign depends on convention)
        phase_20 = jnp.angle(amplitudes[2, 0])
        phase_02 = jnp.angle(amplitudes[0, 2])
        relative_phase = phase_20 - phase_02

        # Relative phase magnitude should be 2*phi (N=2 enhancement)
        assert jnp.isclose(jnp.abs(relative_phase), 2 * phi, atol=1e-5)

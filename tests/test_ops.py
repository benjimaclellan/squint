# %%
import equinox as eqx
import jax.numpy as jnp
import pytest

from squint.circuit import Circuit
from squint.ops.base import SharedGate, Wire
from squint.ops.dv import Conditional, DiscreteVariableState, HGate, RZGate, XGate
from squint.ops.noise import BitFlipChannel, DepolarizingChannel, ErasureChannel
from squint.utils import partition_op


# %%
@pytest.mark.parametrize("n", [2, 3, 4])
def test_ghz_fisher_information(n: int):
    wires = [Wire(dim=2, idx=i) for i in range(n)]

    circuit = Circuit(backend="pure")
    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

    circuit.add(HGate(wires=(wires[0],)))
    for i in range(n - 1):
        circuit.add(Conditional(gate=XGate, wires=(wires[i], wires[i + 1])))

    circuit.add(
        SharedGate(op=RZGate(wires=(wires[0],), phi=0.1 * jnp.pi), wires=tuple(wires[1:])),
        "phase",
    )
    for w in wires:
        circuit.add(HGate(wires=(w,)))

    params, static = partition_op(circuit, "phase")

    sim = circuit.compile(static, params)
    qfi = sim.amplitudes.qfim(params)
    cfi = sim.probabilities.cfim(params)

    assert jnp.isclose(qfi.squeeze(), n**2), "QFI for the GHZ circuit is not `n**2`"
    assert jnp.isclose(cfi.squeeze(), n**2), "CFI for the GHZ circuit is not `n**2`"


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_mixed_state_density(n: int, p: float):
    wires = [Wire(dim=2, idx=i) for i in range(n)]

    circuit = Circuit(backend="mixed")
    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,)))
        circuit.add(BitFlipChannel(wires=(w,), p=p))

    params, static = eqx.partition(circuit, eqx.is_inexact_array)

    sim = circuit.compile(static, params)
    density = sim.amplitudes.forward(params)

    assert jnp.isclose(density[*(n * [0] + n * [0])], (1 - p) ** n)
    assert jnp.isclose(density[*(n * [1] + n * [1])], p**n)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_pure_state_density(n: int):
    wires = tuple(Wire(dim=2, idx=i) for i in range(n))
    bases = [tuple(1 if i == j else 0 for i in range(n)) for j in range(n)]

    circuit = Circuit(backend="mixed")
    circuit.add(
        DiscreteVariableState(
            wires=wires,
            n=[(1.0, tuple(1 if i == j else 0 for i in range(n))) for j in range(n)],
        )
    )

    params, static = eqx.partition(circuit, eqx.is_inexact_array)

    sim = circuit.compile(static, params)
    density = sim.amplitudes.forward(params)
    for basis_in in bases:
        for basis_out in bases:
            assert jnp.isclose(density[*(basis_in + basis_out)], 1 / n)


def test_depolarizing_vs_erasure():
    """
    Test the equivalence of Kraus operators and Stinespring dilation, that depolarizing noise and erasure of a maximally entangled state are equivalent.
    """
    wire0 = Wire(dim=2, idx=0)
    wire1 = Wire(dim=2, idx=1)

    circuit_erasure = Circuit(backend="mixed")
    circuit_erasure.add(
        DiscreteVariableState(
            wires=(wire0, wire1),
            n=[(1.0, (0, 0)), (1.0, (1, 1))],
        )
    )
    circuit_erasure.add(ErasureChannel(wires=(wire0,)))
    params_erasure, static = eqx.partition(circuit_erasure, eqx.is_inexact_array)
    sim = circuit_erasure.compile(static, params_erasure)
    density_erasure = sim.amplitudes.forward(params_erasure)

    wire_single = Wire(dim=2, idx=0)
    circuit_depolarizing = Circuit(backend="mixed")
    circuit_depolarizing.add(DiscreteVariableState(wires=(wire_single,)))
    circuit_depolarizing.add(DepolarizingChannel(wires=(wire_single,), p=1.0))
    params_depolarizing, static = eqx.partition(
        circuit_depolarizing, eqx.is_inexact_array
    )
    sim = circuit_depolarizing.compile(static, params_depolarizing)
    density_depolarizing = sim.amplitudes.forward(params_depolarizing)

    assert jnp.allclose(density_depolarizing, density_erasure)

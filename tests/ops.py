# %%
import equinox as eqx
import jax.numpy as jnp
import pytest

from squint.circuit import Circuit
from squint.ops.base import SharedGate
from squint.ops.dv import Conditional, DiscreteVariableState, HGate, RZGate, XGate
from squint.ops.noise import BitFlipChannel, DepolarizingChannel, ErasureChannel


# %%
@pytest.mark.parametrize("n", [2, 3, 4])
def test_ghz_fisher_information(n: int):
    circuit = Circuit(backend="pure")
    for i in range(n):
        circuit.add(DiscreteVariableState(wires=(i,), n=(0,)))

    circuit.add(HGate(wires=(0,)))
    for i in range(n - 1):
        circuit.add(Conditional(gate=XGate, wires=(i, i + 1)))

    circuit.add(
        SharedGate(op=RZGate(wires=(0,), phi=0.1 * jnp.pi), wires=tuple(range(1, n))),
        "phase",
    )
    for i in range(n):
        circuit.add(HGate(wires=(i,)))

    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    get = lambda pytree: jnp.array([pytree.ops["phase"].op.phi])

    sim = circuit.compile(params, static, dim=2)
    qfi = sim.amplitudes.qfim(get, params)
    cfi = sim.prob.cfim(get, params)

    assert jnp.isclose(qfi.squeeze(), n**2), "QFI for the GHZ circuit is not `n**2`"
    assert jnp.isclose(cfi.squeeze(), n**2), "CFI for the GHZ circuit is not `n**2`"


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_mixed_state_density(n: int, p: float):
    circuit = Circuit(backend="mixed")
    for i in range(n):
        circuit.add(DiscreteVariableState(wires=(i,)))
        circuit.add(BitFlipChannel(wires=(i,), p=p))

    params, static = eqx.partition(circuit, eqx.is_inexact_array)

    sim = circuit.compile(params, static, dim=2)
    density = sim.amplitudes.forward(params)

    assert jnp.isclose(density[*(n * [0] + n * [0])], (1 - p) ** n)
    assert jnp.isclose(density[*(n * [1] + n * [1])], p**n)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_pure_state_density(n: int):
    wires = tuple(range(n))
    bases = [tuple(1 if i == j else 0 for i in wires) for j in wires]

    circuit = Circuit(backend="mixed")
    circuit.add(
        DiscreteVariableState(
            wires=wires,
            n=[(1.0, tuple(1 if i == j else 0 for i in wires)) for j in wires],
        )
    )

    params, static = eqx.partition(circuit, eqx.is_inexact_array)

    sim = circuit.compile(params, static, dim=2)
    density = sim.amplitudes.forward(params)
    for basis_in in bases:
        for basis_out in bases:
            assert jnp.isclose(density[*(basis_in + basis_out)], 1 / n)


def test_depolarizing_vs_erasure():
    """
    Test the equivalence of Kraus operators and Stinespring dilation, that depolarizing noise and erasure of a maximally entangled state are equivalent.
    """
    circuit_erasure = Circuit(backend="mixed")
    circuit_erasure.add(
        DiscreteVariableState(
            wires=(0, 1),
            n=[(1.0, (0, 0)), (1.0, (1, 1))],
        )
    )
    circuit_erasure.add(ErasureChannel(wires=(0,)))
    params_erasure, static = eqx.partition(circuit_erasure, eqx.is_inexact_array)
    sim = circuit_erasure.compile(circuit_erasure, static, dim=2)
    density_erasure = sim.amplitudes.forward(params_erasure)

    circuit_depolarizing = Circuit(backend="mixed")
    circuit_depolarizing.add(DiscreteVariableState(wires=(0,)))
    circuit_depolarizing.add(DepolarizingChannel(wires=(0,), p=1.0))
    params_depolarizing, static = eqx.partition(
        circuit_depolarizing, eqx.is_inexact_array
    )
    sim = circuit_depolarizing.compile(params_depolarizing, static, dim=2)
    density_depolarizing = sim.amplitudes.forward(params_depolarizing)

    assert jnp.allclose(density_depolarizing, density_erasure)

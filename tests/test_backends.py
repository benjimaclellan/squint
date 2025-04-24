
import equinox as eqx
import jax
import jax.numpy as jnp

from squint.circuit import Circuit
from squint.ops.fock import (
    BeamSplitter,
    FockState,
    Phase,
)


def test_pure_vs_mixed_backend():
    cfims = {}
    probs = {}

    for backend in ("pure", "mixed"):
        dim = 3
        wires_star = ("s0", "s1")
        wires_ancilla = ("a0", "a1")

        circuit = Circuit(backend=backend)
        circuit.add(
            FockState(
                wires=(
                    0,
                    2,
                ),
                n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
            )
        )
        # the stellar photon accumulates a phase shift prior to collection by the left telescope.
        circuit.add(Phase(wires=(0,), phi=0.01), "phase")

        # we add the resources photon, which is in an even superposition of spatial modes 1 and 3
        circuit.add(
            FockState(
                wires=(1, 3),
                n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
            )
        )

        # we add the linear optical circuit at each telescope (by default this is a 50-50 beamsplitter)
        circuit.add(
            BeamSplitter(
                wires=(0, 1),
            )
        )
        circuit.add(
            BeamSplitter(
                wires=(2, 3),
            )
        )

        get = lambda pytree: jnp.array([pytree.ops["phase"].phi])
        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(circuit, static, dim=dim).jit()

        phis = jnp.linspace(-jnp.pi, jnp.pi, 100)

        def update(phi, params):
            return eqx.tree_at(lambda pytree: pytree.ops["phase"].phi, params, phi)

        probs[backend] = jax.lax.map(
            lambda phi: sim.probabilities.forward(update(phi, params)), phis
        )
        cfims[backend] = jax.lax.map(
            lambda phi: sim.probabilities.cfim(get, update(phi, params)), phis
        )

    assert jnp.allclose(probs["pure"], probs["mixed"])
    assert jnp.allclose(cfims["pure"], cfims["mixed"])

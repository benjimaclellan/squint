import equinox as eqx
import jax
import jax.numpy as jnp

from squint.circuit import Circuit
from squint.ops.base import Wire
from squint.ops.fock import (
    BeamSplitter,
    FockState,
    Phase,
)
from squint.utils import partition_op


def test_pure_vs_mixed_backend():
    cfims = {}
    probs = {}

    for backend in ("pure", "mixed"):
        dim = 3
        # Create 4 wires for the Fock space (modes 0, 1, 2, 3)
        wire0 = Wire(dim=dim, idx=0)
        wire1 = Wire(dim=dim, idx=1)
        wire2 = Wire(dim=dim, idx=2)
        wire3 = Wire(dim=dim, idx=3)

        circuit = Circuit(backend=backend)
        circuit.add(
            FockState(
                wires=(wire0, wire2),
                n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
            )
        )
        # the stellar photon accumulates a phase shift prior to collection by the left telescope.
        circuit.add(Phase(wires=(wire0,), phi=0.01), "phase")

        # we add the resources photon, which is in an even superposition of spatial modes 1 and 3
        circuit.add(
            FockState(
                wires=(wire1, wire3),
                n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
            )
        )

        # we add the linear optical circuit at each telescope (by default this is a 50-50 beamsplitter)
        circuit.add(
            BeamSplitter(
                wires=(wire0, wire1),
            )
        )
        circuit.add(
            BeamSplitter(
                wires=(wire2, wire3),
            )
        )

        params, static = partition_op(circuit, "phase")
        sim = circuit.compile(static, params).jit()
        phis = jnp.linspace(-jnp.pi, jnp.pi, 100)

        def update(phi, params):
            return eqx.tree_at(lambda pytree: pytree.ops["phase"].phi, params, phi)

        probs[backend] = jax.lax.map(
            lambda phi: sim.probabilities.forward(update(phi, params)), phis
        )
        cfims[backend] = jax.lax.map(
            lambda phi: sim.probabilities.cfim(update(phi, params)), phis
        )

    assert jnp.allclose(probs["pure"], probs["mixed"])
    assert jnp.allclose(cfims["pure"], cfims["mixed"])

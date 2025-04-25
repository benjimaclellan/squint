# %%
import dataclasses
import itertools

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt

# from tqdm.auto import tqdm
from tqdm.autonotebook import tqdm as notebook_tqdm

from squint.circuit import Circuit
from squint.ops.fock import (
    BeamSplitter,
    FixedEnergyFockState,
    FockState,
    TwoModeWeakThermalState,
)
from squint.ops.noise import ErasureChannel


# %%
def telescope(n_ancilla_modes: int = 1, n_ancilla_photons_per_mode: int = 1):
    dim = n_ancilla_modes * n_ancilla_photons_per_mode + 1 + 1

    wire_star_left = "sl"
    wire_star_right = "sr"
    wires_ancilla_left = tuple(f"al{i}" for i in range(n_ancilla_modes))
    wires_ancilla_right = tuple(f"ar{i}" for i in range(n_ancilla_modes))
    wires_dump_left = tuple(f"dl{i}" for i in range(n_ancilla_modes))
    wires_dump_right = tuple(f"dr{i}" for i in range(n_ancilla_modes))

    circuit = Circuit(backend="mixed")

    # star modes
    circuit.add(
        TwoModeWeakThermalState(
            wires=(wire_star_left, wire_star_right), epsilon=1.0, g=1.0, phi=0.1
        ),
        "star",
    )

    # ancilla modes
    for i, (wire_ancilla_left, wire_ancilla_right) in enumerate(
        zip(wires_ancilla_left, wires_ancilla_right, strict=False)
    ):
        circuit.add(
            FixedEnergyFockState(
                wires=(wire_ancilla_left, wire_ancilla_right),
                n=n_ancilla_photons_per_mode,
            ),
            f"ancilla{i}",
        )

    # loss modes
    for i, wire_dump in enumerate(wires_dump_left + wires_dump_right):
        circuit.add(FockState(wires=(wire_dump,), n=(0,)), f"vac{i}")

    # loss beamsplitters
    for i, (wire_ancilla, wire_dump) in enumerate(
        zip(
            wires_ancilla_left + wires_ancilla_right,
            wires_dump_left + wires_dump_right,
            strict=False,
        )
    ):
        circuit.add(BeamSplitter(wires=(wire_ancilla, wire_dump), r=0.0), f"loss{i}")

    iterator = itertools.count(0)
    for i, wire_ancilla in enumerate(wires_ancilla_left):
        circuit.add(
            BeamSplitter(wires=(wire_ancilla, wire_star_left)), f"ul{next(iterator)}"
        )
    for wire_i, wire_j in itertools.combinations(wires_ancilla_left, 2):
        circuit.add(BeamSplitter(wires=(wire_i, wire_j)), f"ul{next(iterator)}")

    iterator = itertools.count(0)
    for i, wire_ancilla in enumerate(wires_ancilla_right):
        circuit.add(
            BeamSplitter(wires=(wire_ancilla, wire_star_right)), f"ur{next(iterator)}"
        )
    for wire_i, wire_j in itertools.combinations(wires_ancilla_right, 2):
        circuit.add(BeamSplitter(wires=(wire_i, wire_j)), f"ur{next(iterator)}")

    # circuit.add(LinearOpticalUnitaryGate(wires=wires_ancilla_left + (wire_star_left,)), f"ul")
    # circuit.add(LinearOpticalUnitaryGate(wires=wires_ancilla_right + (wire_star_right,)), f"ur")

    for i, wire_dump in enumerate(wires_dump_left + wires_dump_right):
        circuit.add(ErasureChannel(wires=(wire_dump,)), f"ptrace{i}")

    return circuit, dim


# %%
from typing import Any

import numpy as np


@dataclasses.dataclass
class TelescopeAncilla:
    n_ancilla_modes: int
    n_ancilla_photons_per_mode: int
    dim: int
    cfims: Any


get = lambda pytree: jnp.array([pytree.ops["star"].phi])
phis = jnp.linspace(-jnp.pi, jnp.pi, 100)


def update(phi, params):
    return eqx.tree_at(lambda pytree: pytree.ops["star"].phi, params, phi)


data = []
# %%
for n_ancilla_modes, n_ancilla_photons_per_mode in (
    # (1, 1), (1, 2), (1, 3),
    (2, 1)
):
    print(n_ancilla_modes, n_ancilla_photons_per_mode)
    circuit, dim = telescope(n_ancilla_modes, n_ancilla_photons_per_mode)
    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    sim = circuit.compile(circuit, static, dim=dim).jit()

    # cfims = jax.lax.map(lambda phi: sim.probabilities.cfim(get, update(phi, params)), phis)
    cfims = []
    for phi in notebook_tqdm(phis):
        cfims.append(sim.probabilities.cfim(get, update(phi, params)))
    cfims = jnp.array(cfims)

    d = TelescopeAncilla(
        n_ancilla_modes=n_ancilla_modes,
        n_ancilla_photons_per_mode=n_ancilla_photons_per_mode,
        dim=dim,
        cfims=np.array(cfims),
    )

print("Done")

# %%
print(data[0].cfims.shape)

# %%
for i in range(1):
    plt.plot(data[i].cfims.squeeze())
plt.show()

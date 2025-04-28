#%%
import pytest
import dataclasses
import itertools
from typing import Any
import jax 

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import ultraplot as uplt
import optax

from squint.circuit import Circuit, compile_experimental
from squint.diagram import draw
from squint.ops.fock import (
    BeamSplitter,
    FixedEnergyFockState,
    FockState,
    TwoModeWeakThermalState,
    Phase,
)
from squint.ops.noise import ErasureChannel
from squint.utils import partition_op

#%%
def telescope(n_ancilla_modes: int = 1, n_ancilla_photons_per_mode: int = 1):
    dim = n_ancilla_modes * n_ancilla_photons_per_mode + 1 + 1

    wire_star_left = "sl"
    wire_star_right = "sr"
    wires_dump_left = tuple(f"dl{i}" for i in range(n_ancilla_modes))
    wires_dump_right = tuple(f"dr{i}" for i in range(n_ancilla_modes))

    circuit = Circuit(backend="mixed")
    circuit.add(
        FockState(wires=(wire_star_left, wire_star_right), n=((1.0, (1,0)), (1.0, (0, 1)))),
        'star'
    )
    circuit.add(
        Phase(wires=(wire_star_left,), phi=0.1),
        'phase'
    )
    # star modes
    # circuit.add(
    #     TwoModeWeakThermalState(
    #         wires=(wire_star_left, wire_star_right), epsilon=1.0, g=1.0, phi=0.1
    #     ),
    #     "star",
    # )

    # loss modes
    for i, wire_dump in enumerate(wires_dump_left + wires_dump_right):
        circuit.add(FockState(wires=(wire_dump,), n=(0,)), f"vac{i}")

    # loss beamsplitters
    for i, (wire_ancilla, wire_dump) in enumerate(
        zip(
            (wire_star_left, wire_star_right),
            wires_dump_left + wires_dump_right,
            strict=False,
        )
    ):
        circuit.add(BeamSplitter(wires=(wire_ancilla, wire_dump), r=0.2), f"loss{i}")

    for i, wire_dump in enumerate(wires_dump_left + wires_dump_right):
        circuit.add(ErasureChannel(wires=(wire_dump,)), f"ptrace{i}")

    circuit.add(
        BeamSplitter(wires=(wire_star_left, wire_star_right), r=jnp.pi/4), f"u"
    )

    return circuit, dim

# %%
circuit, dim = telescope(n_ancilla_modes=1, n_ancilla_photons_per_mode=2)

_params, static = eqx.partition(circuit, eqx.is_inexact_array)
# params_est, params_opt = partition_op(_params, "phase")    

params_est, params_fix = partition_op(_params, "phase")    


sim = compile_experimental(
    static, dim, params_est, params_opt, params_fix, **{"optimize": "greedy", "argnum": 0}
) #.jit()

# print(circuit)
print(sim.probabilities.cfim(params_est, params_opt, params_fix))

fig = draw(circuit)
fig.savefig('diagram.png')

#%%
# phis = jnp.linspace(-jnp.pi, jnp.pi, 100)
# params_est_vmap = eqx.tree_at(lambda pytree: pytree.ops["phase"].phi, params_est, phis)
    
# probs = jax.vmap(sim.probabilities.forward, in_axes=(0, None))(params_est_vmap, params_opt)
# qfims = jax.vmap(sim.amplitudes.qfim, in_axes=(0, None))(params_est_vmap, params_opt)
# cfims = jax.vmap(sim.probabilities.cfim, in_axes=(0, None))(params_est_vmap, params_opt)

# colors = itertools.cycle(sns.color_palette("deep", n_colors=10))
# fig, axs = uplt.subplots(nrows=3, ncols=1, figsize=[8, 5], sharey=False)
# for i, idx in enumerate(
#     itertools.product(*[list(range(ell)) for ell in probs.shape[1:]])
# ):
#     if probs[:, *idx].mean() < 1e-6:
#         continue
#     axs[0].plot(phis, probs[:, *idx], label=f"{idx}", color=next(colors))
# axs[0].legend()
# axs[0].set(xlabel=r"Phase, $\varphi$", ylabel=r"Probability, $p(\mathbf{x} | \varphi)$")

# axs[1].plot(phis, qfims.squeeze(), color=next(colors))
# axs[1].set(
#     xlabel=r"Phase, $\varphi$",
#     ylabel=r"$\mathcal{I}_\varphi^Q$",
#     ylim=[0, 1.05 * jnp.max(qfims)],
# )

# axs[2].plot(phis, cfims.squeeze(), color=next(colors))
# axs[2].set(
#     xlabel=r"Phase, $\varphi$",
#     ylabel=r"$\mathcal{I}_\varphi^C$",
#     ylim=[0, 1.05 * jnp.max(cfims)],
# )

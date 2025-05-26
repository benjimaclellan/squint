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
    LinearOpticalUnitaryGate,
)
from squint.ops.noise import ErasureChannel
from squint.utils import partition_op, print_nonzero_entries

#%%
def telescope(n_ancilla_modes: int = 1, n_ancilla_photons_per_mode: int = 1):
    dim = n_ancilla_modes * n_ancilla_photons_per_mode + 1 + 1

    wire_star_left = "sl"
    wire_star_right = "sr"
    wires_ancilla_left = tuple(f"al{i}" for i in range(n_ancilla_modes))
    wires_ancilla_right = tuple(f"ar{i}" for i in range(n_ancilla_modes))

    circuit = Circuit(backend="pure")
    circuit.add(
        FockState(wires=(wire_star_left, wire_star_right), n=((1.0, (1, 0)), (1.0, (0, 1)))),
        'star'
    )
    circuit.add(
        Phase(wires=(wire_star_left,), phi=0.1),
        'phase'
    )

    # ancilla modes
    circuit.add(
        FixedEnergyFockState(
            wires=wires_ancilla_left + wires_ancilla_right,
            n=len(wires_ancilla_left) * n_ancilla_photons_per_mode,
        ),
        f"ancilla",
    )

    circuit.add(
        LinearOpticalUnitaryGate(wires=wires_ancilla_left + (wire_star_left,)), f"ul"
    )
    circuit.add(
        LinearOpticalUnitaryGate(wires=wires_ancilla_right + (wire_star_right,)), f"ur"
    )

    print(
        f"Ancilla modes left: {wires_ancilla_left}, Ancilla modes right: {wires_ancilla_right} | Dim = {dim}"
    )
    return circuit, dim


# %%
circuit, dim = telescope(n_ancilla_modes=1, n_ancilla_photons_per_mode=1)

_params, static = eqx.partition(circuit, eqx.is_inexact_array)
# params_est, params_opt = partition_op(_params, "phase")    

params_est, params_opt = partition_op(_params, "phase")    


sim = compile_experimental(
    static, dim, params_est, params_opt, **{"optimize": "greedy", "argnum": 0}
) #.jit()

# print(circuit)
print(sim.probabilities.cfim(params_est, params_opt))

fig = draw(circuit)
fig.savefig('diagram.png')


# print(params_est)
# print(params_fix)
# print(params_opt)

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

#%%

run = True
if run:
    lr = 1e-2
    optimizer = optax.chain(optax.adam(lr), optax.scale(-1.0))
    opt_state = optimizer.init(params_opt)

    def loss(params_est, params_opt):
        return sim.probabilities.cfim(params_est, params_opt).squeeze()

    value_and_grad = jax.value_and_grad(loss, argnums=1)

    # %%
    @jax.jit
    def step(opt_state, params_est, params_opt):
        val, grad = value_and_grad(params_est, params_opt)
        updates, opt_state = optimizer.update(grad, opt_state)
        params_opt = optax.apply_updates(params_opt, updates)
        return params_opt, opt_state, val

    _ = step(opt_state, params_est, params_opt)

    # %%
    cfims = []
    for _ in range(3000):
        params_opt, opt_state, val = step(opt_state, params_est, params_opt)
        cfims.append(val)
        print(val)
    
    eqx.tree_pprint(eqx.combine(params_est, params_opt), short_arrays=False)


probs = sim.probabilities.forward(params_est, params_opt)
print_nonzero_entries(probs)
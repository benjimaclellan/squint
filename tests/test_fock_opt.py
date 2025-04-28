#%%
import pytest
import dataclasses
import itertools
from typing import Any
import jax 

import equinox as eqx
import jax.numpy as jnp
import numpy as np
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
from squint.utils import partition_op

#%%
def telescope(n_ancilla_modes: int = 1, n_ancilla_photons_per_mode: int = 1):
    dim = n_ancilla_modes * n_ancilla_photons_per_mode + 1 + 1

    wire_star_left = "sl"
    wire_star_right = "sr"
    wires_ancilla_left = tuple(f"al{i}" for i in range(n_ancilla_modes))
    wires_ancilla_right = tuple(f"ar{i}" for i in range(n_ancilla_modes))

    circuit = Circuit(backend="pure")
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

    iterator = itertools.count(0)
    for i, wire_ancilla in enumerate(wires_ancilla_left):
        circuit.add(
            BeamSplitter(wires=(wire_ancilla, wire_star_left), r=1.4), f"ul{next(iterator)}"
        )
    for wire_i, wire_j in itertools.combinations(wires_ancilla_left, 2):
        circuit.add(BeamSplitter(wires=(wire_i, wire_j), r=0.4), f"ul{next(iterator)}")

    iterator = itertools.count(0)
    for i, wire_ancilla in enumerate(wires_ancilla_right):
        circuit.add(
            BeamSplitter(wires=(wire_ancilla, wire_star_right)), f"ur{next(iterator)}"
        )
    for wire_i, wire_j in itertools.combinations(wires_ancilla_right, 2):
        circuit.add(BeamSplitter(wires=(wire_i, wire_j)), f"ur{next(iterator)}")

    return circuit, dim

# %%
circuit, dim = telescope(n_ancilla_modes=2, n_ancilla_photons_per_mode=1)

params, static = eqx.partition(circuit, eqx.is_inexact_array)
params_est, params_opt = partition_op(params, "phase")    

sim = compile_experimental(
    static, dim, params, **{"optimize": "greedy", "argnum": 0}
) #.jit()

print(circuit)
print(sim.amplitudes.qfim(params_est, params_opt))

#%%
phis = jnp.linspace(-jnp.pi, jnp.pi, 100)
params_est_vmap = eqx.tree_at(lambda pytree: pytree.ops["phase"].phi, params_est, phis)
    
probs = jax.vmap(sim.probabilities.forward, in_axes=(0, None))(params_est_vmap, params_opt)
qfims = jax.vmap(sim.amplitudes.qfim, in_axes=(0, None))(params_est_vmap, params_opt)
cfims = jax.vmap(sim.probabilities.cfim, in_axes=(0, None))(params_est_vmap, params_opt)

colors = itertools.cycle(sns.color_palette("deep", n_colors=10))
fig, axs = uplt.subplots(nrows=3, ncols=1, figsize=[8, 5], sharey=False)
for i, idx in enumerate(
    itertools.product(*[list(range(ell)) for ell in probs.shape[1:]])
):
    if probs[:, *idx].mean() < 1e-6:
        continue
    axs[0].plot(phis, probs[:, *idx], label=f"{idx}", color=next(colors))
axs[0].legend()
axs[0].set(xlabel=r"Phase, $\varphi$", ylabel=r"Probability, $p(\mathbf{x} | \varphi)$")

axs[1].plot(phis, qfims.squeeze(), color=next(colors))
axs[1].set(
    xlabel=r"Phase, $\varphi$",
    ylabel=r"$\mathcal{I}_\varphi^Q$",
    ylim=[0, 1.05 * jnp.max(qfims)],
)

axs[2].plot(phis, cfims.squeeze(), color=next(colors))
axs[2].set(
    xlabel=r"Phase, $\varphi$",
    ylabel=r"$\mathcal{I}_\varphi^C$",
    ylim=[0, 1.05 * jnp.max(cfims)],
)
    
#%%
lr = 1e-3
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

assert jnp.abs(val - 0.5) < 0.01, (
    f"Optimization did not converge to expected limit for n={n}, final value {val}"
)
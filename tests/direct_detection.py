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
def telescope():
    dim = 2

    wire_star_left = "sl"
    wire_star_right = "sr"
    wire_dump_left = "dl"
    wire_dump_right = "dr"

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
    for i, wire_dump in enumerate((wire_dump_left, wire_dump_right)):
        circuit.add(FockState(wires=(wire_dump,), n=(0,)), f"vac{i}")

    # loss beamsplitters
    for i, (wire_ancilla, wire_dump) in enumerate(
        zip(
            (wire_star_left, wire_star_right),
            (wire_dump_left, wire_dump_right),
            strict=False,
        )
    ):
        circuit.add(BeamSplitter(wires=(wire_ancilla, wire_dump), r=0.0), f"loss{i}")

    for i, wire_dump in enumerate((wire_dump_left, wire_dump_right)):
        circuit.add(ErasureChannel(wires=(wire_dump,)), f"ptrace{i}")

    circuit.add(
        BeamSplitter(wires=(wire_star_left, wire_star_right), r=jnp.pi/4), f"u"
    )

    return circuit, dim

# %%
circuit, dim = telescope()

_params, static = eqx.partition(circuit, eqx.is_inexact_array)
params_est, _params = partition_op(_params, "phase")    
params_fix, params_opt = eqx.partition(
    _params, 
    lambda x: any(x is leaf for leaf in [circuit.ops['loss0'].r, circuit.ops['loss1'].r])
)


sim = compile_experimental(
    static, dim, params_est, params_opt, params_fix, **{"optimize": "greedy", "argnum": 0}
) #.jit()

# print(circuit)
print(sim.probabilities.cfim(params_est, params_opt, params_fix))

fig = draw(circuit)
fig.savefig('diagram.png')

#%%
rs = jnp.linspace(0.0, jnp.pi/2, 100)
# params_fix_vmap = eqx.tree_at(lambda pytree: [pytree.ops["loss0"].r, pytree.ops["loss1"].r], params_fix, [rs, rs])
params_fix_vmap = eqx.tree_at(
    lambda pytree: [pytree.ops["loss0"].r, pytree.ops["loss1"].r], 
    params_fix, 
    [0.25 * jnp.linspace(0.0, jnp.pi/2, 100), 0.5 * jnp.linspace(0.0, jnp.pi/2, 100)]
)
    
probs = jax.vmap(sim.probabilities.forward, in_axes=(None, None, 0))(params_est, params_opt, params_fix_vmap)
cfims = jax.vmap(sim.probabilities.cfim, in_axes=(None, None, 0))(params_est, params_opt, params_fix_vmap)

#%%
colors = itertools.cycle(sns.color_palette("deep", n_colors=10))
fig, axs = uplt.subplots(nrows=2, ncols=1, figsize=[8, 5], sharey=False)
for i, idx in enumerate(
    itertools.product(*[list(range(ell)) for ell in probs.shape[1:]])
):
    if probs[:, *idx].mean() < 1e-6:
        continue
    axs[0].plot(rs, probs[:, *idx], label=f"{idx}", color=next(colors))
axs[0].legend()
axs[0].set(xlabel=r"Phase, $\varphi$", ylabel=r"Probability, $p(\mathbf{x} | \varphi)$")

axs[1].plot(rs, cfims.squeeze(), color=next(colors))
axs[1].set(
    xlabel=r"Phase, $\varphi$",
    ylabel=r"$\mathcal{I}_\varphi^C$",
    ylim=[0, 1.05 * jnp.max(cfims)],
)

# %%
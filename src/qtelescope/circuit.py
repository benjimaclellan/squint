# %%
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import paramax
from rich.pretty import pprint

from qtelescope.ops import BeamSplitter, Circuit, FockState, Phase
from qtelescope.utils import partition_op, print_nonzero_entries

# %%  Express the optical circuit.
# ------------------------------------------------------------------
cutoff = 4
circuit = Circuit()

# circuit.add(
#     FockState(
#         wires=(0, 1),
#         n=[(1 / jnp.sqrt(2).item(), (3, 0)), (1 / jnp.sqrt(2).item(), (1, 3))],
#     )
# )
circuit.add(
    FockState(
        wires=(0, 1, 2),
        n=[
            (1 / jnp.sqrt(3).item(), (3, 0, 0)),
            (1 / jnp.sqrt(3).item(), (0, 3, 0)),
            (1 / jnp.sqrt(3).item(), (0, 0, 3)),
        ],
    )
)
# circuit.add(BeamSplitter(wires=(0, 1), r=jnp.pi/4))
circuit.add(Phase(wires=(0,), phi=0.0), "phase")
# circuit.add(Phase(wires=(0,), phi=0.0), "phase2")
circuit.add(BeamSplitter(wires=(0, 1), r=jnp.pi / 8))
circuit.add(BeamSplitter(wires=(0, 2), r=jnp.pi / 8))

# m = 4
# for i in range(m):
#     circuit.add(FockState(wires=(i,), n=(1,)))
# circuit.add(Phase(wires=(0,), phi=0.0), "phase")
# for i in range(m - 1):
#     circuit.add(BeamSplitter(wires=(i, i + 1), r=jnp.pi / 4))

pprint(circuit)

# %% split into training parameters and static
# ------------------------------------------------------------------
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)
# print(params)

sim = circuit.compile(params, static, cut=cutoff, optimize="greedy")
sim_jit = sim.jit()

# %% split into probe parameters and static, sweep some parameter over
# ------------------------------------------------------------------
name = "phase"
params, static = partition_op(circuit, name)
sim = circuit.compile(params, static, cut=cutoff, optimize="greedy")
sim_jit = sim.jit()


@functools.partial(jax.vmap, in_axes=(0, None))
def sweep_phase(phi, params):
    params = eqx.tree_at(lambda params: params.ops["phase"].phi, params, phi)

    grad = sim.grad(params)
    cfim = (grad.ops[name].phi ** 2 / (pr + 1e-12)).sum()
    return cfim


pr = sim.probability(params)
print_nonzero_entries(pr)

# %%
phis = jnp.linspace(0.0, 2.0, 50)
cfims = sweep_phase(phis, params)

# %%
fig, ax = plt.subplots()
ax.plot(phis, cfims)
fig.show()

# %%
##%% Differentiate with respect to parameters of interest


# cfim = (hess.ops[name].phi.ops[name].phi / (pr + 1e-12)).sum()
# cfim = (grad.ops[name].phi**2 / (pr + 1e-12)).sum()
# print("CFIM:", cfim)

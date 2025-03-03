# %%
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.fock import BeamSplitter, FockState, Phase
from squint.utils import partition_op, print_nonzero_entries

# %%  Express the optical circuit.
# ------------------------------------------------------------------
circuit = Circuit()

n = 3
cutoff = 2 * n + 1

setup = "noon"
# setup = "holland-burnett"

if setup == "noon":
    circuit.add(
        FockState(
            wires=(0, 1),
            n=(
                (1 / jnp.sqrt(2).item(), (2 * n, 0)),
                (1 / jnp.sqrt(2).item(), (0, 2 * n)),
            ),
        )
    )

elif setup == "holland-burnett":
    circuit.add(FockState(wires=(0,), n=(n,)))
    circuit.add(FockState(wires=(1,), n=(n,)))
    circuit.add(
        BeamSplitter(
            wires=(
                0,
                1,
            ),
            r=jnp.pi / 4,
        )
    )

circuit.add(Phase(wires=(0,), phi=jnp.pi / 4), "phase")
circuit.add(
    BeamSplitter(
        wires=(
            0,
            1,
        ),
        r=jnp.pi / 4,
    )
)

pprint(circuit)
circuit.verify()

# %%
##%% split into training parameters and static
# ------------------------------------------------------------------
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    # is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)
print(params)

# %%
sim = circuit.compile(params, static, dim=cutoff, optimize="greedy")
sim_jit = sim.jit()

##%%
tensor = sim.forward(params)
pr = sim.prob(params)
print_nonzero_entries(pr)

# %%
key = jr.PRNGKey(0)
samples = sim.sample(key, params, shape=(4, 5))
print(samples)

# %% Differentiate with respect to parameters of interest
name = "phase"
params, static = partition_op(circuit, name)
sim = circuit.compile(params, static, dim=cutoff, optimize="greedy")
sim_jit = sim.jit()

# %%
grad = sim.grad(params)
pr = sim.prob(params)
cfim = (grad.ops[name].phi ** 2 / (pr + 1e-12)).sum()
print(cfim)


# %%
@functools.partial(jax.vmap, in_axes=(0, None))
def sweep_phase(phi, params):
    params = eqx.tree_at(lambda params: params.ops[name].phi, params, phi)
    grad = sim.grad(params)
    pr = sim.prob(params)
    cfim = (grad.ops[name].phi ** 2 / (pr + 1e-12)).sum()
    return cfim


# %%
phis = jnp.linspace(0.0001, 2 * jnp.pi, 53)
cfims = sweep_phase(phis, params)

# %%
fig, ax = plt.subplots()
ax.plot(phis, cfims)
ax.set(xlabel="Stellar Photon Phase", ylabel="Classical Fisher Information")
fig.show()

# %%

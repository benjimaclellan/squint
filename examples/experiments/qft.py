# %%
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.fock import QFT, BeamSplitter, FockState, Phase
from squint.utils import partition_op, print_nonzero_entries

# %%  Express the optical circuit.
# ------------------------------------------------------------------
cutoff = 4
circuit = Circuit()

circuit.add(FockState(wires=(0,), n=(1,)))
circuit.add(FockState(wires=(1,), n=(1,)))
circuit.add(FockState(wires=(2,), n=(1,)))
circuit.add(
    BeamSplitter(
        wires=(
            0,
            1,
        ),
        r=jnp.pi / 4,
    )
)
circuit.add(Phase(wires=(1,), phi=0.001), "phase")
circuit.add(QFT(wires=(0, 1, 2), coeff=1 / jnp.sqrt(2).item()), "qft")

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
pr = sim.probabilities(params)
print_nonzero_entries(pr)

# %%
name = "qft"


@functools.partial(jax.vmap, in_axes=(0, None))
def sweep_coeff(coeff, params):
    params = eqx.tree_at(lambda params: params.ops[name].coeff, params, coeff)
    pr = sim.probabilities(params)
    return pr


coeffs = jnp.linspace(0.0001, 0.5 * jnp.pi, 50)
prs = sweep_coeff(coeffs, params)

# %%
fig, ax = plt.subplots()
for idx in ((1, 1, 1), (2, 0, 1), (0, 1, 2)):
    ax.plot(coeffs, prs[:, *idx])
fig.show()

# %%
name = "phase"
params, static = partition_op(circuit, name)
sim = circuit.compile(params, static, dim=cutoff, optimize="greedy")
sim_jit = sim.jit()


@functools.partial(jax.vmap, in_axes=(0, None))
def sweep_phase(phi, params):
    params = eqx.tree_at(lambda params: params.ops[name].phi, params, phi)
    grad = sim_jit.grad(params)
    pr = sim_jit.probabilities(params)
    cfim = (grad.ops[name].phi ** 2 / (pr + 1e-12)).sum()
    return cfim


phis = jnp.linspace(1.101, 2 * jnp.pi, 25)
cfims = sweep_phase(phis, params)

fig, ax = plt.subplots()
ax.plot(phis, cfims)
ax.set(xlabel="Stellar Photon Phase", ylabel="Classical Fisher Information")
fig.show()
# %%

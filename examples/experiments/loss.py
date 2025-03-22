# %%
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.fock import BeamSplitter, FockState, Phase
from squint.utils import print_nonzero_entries

# %%  Express the optical circuit.
# ------------------------------------------------------------------
circuit = Circuit()

n = 4
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

circuit.add(FockState(wires=(2,), n=(0,)))
circuit.add(
    BeamSplitter(
        wires=(
            0,
            2,
        ),
        r=0.00001,
    ),
    "loss_bs1",
)
circuit.add(
    BeamSplitter(
        wires=(
            1,
            2,
        ),
        r=0.00001,
    ),
    "loss_bs2",
)

# loss
# circuit.add(FockState(wires=(2,), n=(0,)))
# circuit.add(BeamSplitter(wires=(0, 2,), r=0.00001), "loss_bs1")
# circuit.add(BeamSplitter(wires=(0, 1,), r=0.00001), "loss_bs2")

# phase encoding and beamsplitter detection
circuit.add(Phase(wires=(0,), phi=0.1), "phase")
circuit.add(
    BeamSplitter(
        wires=(
            0,
            1,
        ),
        r=-jnp.pi / 4,
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
pr = sim.probability(params)
print_nonzero_entries(pr)


# %%
@jax.jit
@functools.partial(jax.vmap, in_axes=(0, None))
def sweep_loss(r, params):
    params = eqx.tree_at(lambda params: params.ops["loss_bs1"].r, params, r)
    params = eqx.tree_at(lambda params: params.ops["loss_bs2"].r, params, r)

    def pr_ptrace(params):
        tensor = sim.forward(params)
        pr = jnp.abs(tensor) ** 2
        pr = pr.sum(axis=-1)
        return pr

    grad_fn = jax.jacrev(pr_ptrace)

    grad = grad_fn(params)
    pr = pr_ptrace(params)

    A = grad.ops["phase"].phi ** 2
    B = pr
    cfim = jnp.sum(A * (B / (B**2 + 1e-18)))  # soft-approximation of CFI

    # cfim = (grad.ops["phase"].phi ** 2 / (pr + 1e-12)).sum()
    return cfim


# %%
rs = jnp.linspace(0.0001, 1.5, 250)
cfims = sweep_loss(rs, params)

# %%
fig, ax = plt.subplots()
ax.plot(rs, cfims)
ax.set(xlabel=r"Loss, $\alpha$", ylabel="CFI", ylim=[-0.5, 1.1 * (2 * n) ** 2])
fig.show()

# %%

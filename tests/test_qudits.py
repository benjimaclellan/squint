# %%

import equinox as eqx
import jax.numpy as jnp

from squint.circuit import Circuit
from squint.ops.base import Wire
from squint.ops.dv import DiscreteVariableState, HGate, RZGate

# %%


def test_qudit_simple():
    # %%
    dim = 6
    wire = Wire(dim=dim, idx=0)

    circuit = Circuit()

    circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
    circuit.add(HGate(wires=(wire,)))
    circuit.add(RZGate(wires=(wire,), phi=0.1 * jnp.pi), "phase")
    circuit.add(HGate(wires=(wire,)))

    params, static = eqx.partition(circuit, eqx.is_inexact_array)

    sim = Circuit.compile(static, params, **{"optimize": "greedy", "argnum": 0})
    phis = jnp.linspace(-jnp.pi, jnp.pi, 100)

    print(sim.amplitudes.forward(params))
    print(sim.amplitudes.qfim(params))
    print(sim.probabilities.cfim(params))

    # %%
    params = eqx.tree_at(lambda pytree: pytree.ops["phase"].phi, params, phis)

    probs = eqx.filter_vmap(sim.probabilities.forward)(params)
    cfims = eqx.filter_vmap(sim.probabilities.cfim)(params)
    qfims = eqx.filter_vmap(sim.amplitudes.qfim)(params)
    print(probs.shape)

    # colors = sns.color_palette("deep", n_colors=jnp.prod(jnp.array(probs.shape[1:])))
    # fig, axs = uplt.subplots(nrows=3, ncols=1, figsize=[8, 5], sharey=False)
    # for i, idx in enumerate(
    #     itertools.product(*[list(range(ell)) for ell in probs.shape[1:]])
    # ):
    #     axs[0].plot(phis, probs[:, *idx], label=f"{idx}", color=colors[i])
    # axs[0].legend()
    # axs[0].set(xlabel=r"Phase, $\varphi$", ylabel=r"Probability, $p(\mathbf{x} | \varphi)$")

    # axs[1].plot(phis, qfims.squeeze(), color=colors[i])
    # axs[1].set(
    #     xlabel=r"Phase, $\varphi$",
    #     ylabel=r"$\mathcal{I}_\varphi^Q$",
    #     ylim=[0, 1.05 * jnp.max(qfims)],
    # )

    # axs[2].plot(phis, cfims.squeeze(), color=colors[i])
    # axs[2].set(
    #     xlabel=r"Phase, $\varphi$",
    #     ylabel=r"$\mathcal{I}_\varphi^C$",
    #     ylim=[0, 1.05 * jnp.max(cfims)],
    # )


# %%

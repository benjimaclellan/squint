import marimo

__generated_with = "0.11.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Qubit sensing protocols
        This example shows a one-qubit interference experiment.
        """
    )
    return


@app.cell
def _():
    import itertools

    import equinox as eqx
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import seaborn as sns
    from rich.pretty import pprint

    from squint.circuit import Circuit
    from squint.ops.dv import DiscreteVariableState, HGate, RZGate

    return (
        Circuit,
        DiscreteVariableState,
        HGate,
        RZGate,
        eqx,
        itertools,
        jnp,
        plt,
        pprint,
        sns,
    )


@app.cell
def _(Circuit, DiscreteState, HGate, Phase, eqx, jnp, pprint):
    circuit = Circuit()
    circuit.add(DiscreteState(wires=(0,), n=(0,)))
    circuit.add(HGate(wires=(0,)))
    circuit.add(Phase(wires=(0,), phi=0.1 * jnp.pi), "phase")
    circuit.add(HGate(wires=(0,)))

    params, static = eqx.partition(circuit, eqx.is_inexact_array)

    pprint(circuit)
    pprint(params)
    pprint(static)
    return circuit, params, static


@app.cell
def _(circuit, eqx, itertools, jnp, params, plt, sns, static):
    sim = circuit.compile(params, static, dim=2, optimize="greedy")
    get = lambda pytree: jnp.array([pytree.ops["phase"].phi])
    phis = jnp.linspace(-jnp.pi, jnp.pi, 100)
    params_1 = eqx.tree_at(
        lambda pytree: pytree.ops["phase"].phi, params, jnp.expand_dims(phis, axis=1)
    )
    probs = eqx.filter_vmap(sim.probabilities.forward)(params_1)
    cfims = eqx.filter_vmap(sim.probabilities.cfim, in_axes=(None, 0))(get, params_1)
    qfims = eqx.filter_vmap(sim.amplitudes.qfim, in_axes=(None, 0))(get, params_1)
    colors = sns.color_palette("crest", n_colors=jnp.prod(jnp.array(probs.shape[1:])))
    fig, ax = plt.subplots()
    for i, idx in enumerate(
        itertools.product(*[list(range(ell)) for ell in probs.shape[1:]])
    ):
        ax.plot(phis, probs[:, *idx], label=f"{idx}", color=colors[i])
    ax.legend()
    ax.set(
        xlabel="Phase, $\\varphi$", ylabel="Probability, $p(\\mathbf{x} | \\varphi)$"
    )
    fig, ax = plt.subplots()
    ax.plot(phis, qfims.squeeze(), color=colors[i])
    ax.set(
        xlabel="Phase, $\\varphi$",
        ylabel="$\\mathcal{I}_\\varphi^Q$",
        ylim=[0, 1.05 * jnp.max(qfims)],
    )
    fig, ax = plt.subplots()
    ax.plot(phis, cfims.squeeze(), color=colors[i])
    ax.set(
        xlabel="Phase, $\\varphi$",
        ylabel="$\\mathcal{I}_\\varphi^C$",
        ylim=[0, 1.05 * jnp.max(cfims)],
    )
    return (
        ax,
        cfims,
        colors,
        fig,
        get,
        i,
        idx,
        params_1,
        phis,
        probs,
        qfims,
        sim,
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from rich.pretty import pprint

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx
    import paramax
    import matplotlib.pyplot as plt
    import seaborn as sns
    import copy 

    from qtelescope.ops import (
        Circuit, BeamSplitter, Phase, S2, FockState
    )
    return (
        BeamSplitter,
        Circuit,
        FockState,
        Phase,
        S2,
        copy,
        eqx,
        jax,
        jnp,
        jr,
        mo,
        paramax,
        plt,
        pprint,
        sns,
    )


@app.cell
def _(BeamSplitter, Circuit, FockState, Phase):
    n = 3
    cutoff = 4
    circuit = Circuit(cutoff=cutoff)
    for i in range(n):
        circuit.add(FockState(wires=(i,), n=(1,)))

    for i in range(n-1):
        circuit.add(BeamSplitter(wires=(i, i+1)))

    circuit.add(Phase(wires=(0,), phi=0.2), "mark")
    return circuit, cutoff, i, n


@app.cell
def _(circuit, eqx):
    params, static = eqx.partition(
        circuit,
        eqx.is_inexact_array,
        # is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    print(params, static)
    return params, static


@app.cell
def _():
    return


@app.cell
def _(sliders):
    sliders
    return


@app.cell
def _(jax, params):
    leaves, treedef = jax.tree.flatten(params)
    print(leaves)

    return leaves, treedef


@app.cell
def _(leaves, mo, p):

    tmp = tuple(
        (mo.ui.slider(start=0.0, stop=1.0, step=0.1, value=leaf.item(), on_change=lambda val, name=idx: p(val, name)), f"name{idx}")
        for idx, leaf in enumerate(leaves)
    )

    sliders, names = list(zip(*tmp))
    # print(names)
    return names, sliders, tmp


@app.cell
def _(copy, jnp, leaves, plt):
    fig, ax = plt.subplots()
    tmp_params = copy.copy(leaves)
    def p(val, idx):
        # print(list([slider.value for slider in sliders]))
        print(val, idx)
        tmp_params[idx] = jnp.array(val)
        tmp_params
        ax.plot(tmp_params)
        fig.show()
    return ax, fig, p, tmp_params


@app.cell
def _(i, jnp, tmp_params):
    tmp_params[i] = jnp.array(0.1)
    print(tmp_params)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # `squint` 
        ## Quantum sensing protocols, interactive notebook

        A reactive notebook for interacting with and studying quantum sensing protocols.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import copy

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import marimo as mo
    import matplotlib.pyplot as plt
    import paramax
    import seaborn as sns
    from loguru import logger
    from rich.pretty import pprint

    from squint.circuit import Circuit
    from squint.ops.fock import BeamSplitter, FockState, Phase
    from squint.utils import extract_paths
    return (
        BeamSplitter,
        Circuit,
        FockState,
        Phase,
        copy,
        eqx,
        extract_paths,
        jax,
        jnp,
        jr,
        logger,
        mo,
        paramax,
        plt,
        pprint,
        sns,
    )


@app.cell(hide_code=True)
def _(logger):
    logger.remove()
    logger.add(lambda msg: None, level="WARNING")
    return


@app.cell
def _(BeamSplitter, Circuit, FockState, Phase, jnp):
    dim = 4  # Fock cutoff
    circuit = Circuit()

    # stellar photon in superposition of modes 0 and 3
    circuit.add(
        FockState(
            wires=(
                0,
                3,
            ),
            n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
        )
    )
    circuit.add(Phase(wires=(0,), phi=0.01), "phase")

    # create ancilla state
    circuit.add(
        FockState(
            wires=(
                1,
                4,
            ),
            n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
        )
    )
    circuit.add(
        FockState(
            wires=(
                2,
                5,
            ),
            n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
        )
    )

    for telescope in (0, 1):
        circuit.add(
            BeamSplitter(
                wires=(
                    0 + telescope * 3,
                    1 + telescope * 3,
                ),
                r=jnp.pi / 2.1,
            )
        )
        circuit.add(Phase(wires=(0 + telescope * 3,), phi=0.01))
        circuit.add(
            BeamSplitter(
                wires=(
                    0 + telescope * 3,
                    1 + telescope * 3,
                ),
                r=jnp.pi / 2.1,
            )
        )

        circuit.add(
            BeamSplitter(
                wires=(
                    1 + telescope * 3,
                    2 + telescope * 3,
                ),
                r=jnp.pi / 2.1,
            )
        )
        circuit.add(Phase(wires=(1 + telescope * 3,), phi=0.01))
        circuit.add(
            BeamSplitter(
                wires=(
                    1 + telescope * 3,
                    2 + telescope * 3,
                ),
                r=jnp.pi / 2.1,
            )
        )

        circuit.add(
            BeamSplitter(
                wires=(
                    0 + telescope * 3,
                    1 + telescope * 3,
                ),
                r=jnp.pi / 2.1,
            )
        )
        circuit.add(Phase(wires=(0 + telescope * 3,), phi=0.01))
        circuit.add(
            BeamSplitter(
                wires=(
                    0 + telescope * 3,
                    1 + telescope * 3,
                ),
                r=jnp.pi / 2.1,
            )
        )

    return circuit, dim, telescope


@app.cell
def _(circuit, dim, eqx):
    circuit.verify()

    params, static = eqx.partition(
        circuit,
        eqx.is_inexact_array,
    )

    sim = circuit.compile(params, static, dim=dim, optimize="greedy")
    sim_jit = sim.jit()
    return params, sim, sim_jit, static


@app.cell(hide_code=True)
def _(extract_paths, jax, mo, params):
    leaves, treedef = jax.tree.flatten(params)
    jax.tree.unflatten(treedef, leaves)
    paths = list(extract_paths(params))

    _sliders = tuple(
        (
            mo.ui.slider(
                start=0.0,
                stop=2.0,
                step=0.05,
                value=leaf.item(),
                show_value=True,
                label=f"{op_type}: {path}",
            ),
            f"[{op_type}]{path}",
        )
        for idx, (leaf, (path, op_type, value)) in enumerate(
            zip(leaves, paths, strict=False)
        )
    )

    sliders, names = list(zip(*_sliders, strict=False))
    return leaves, names, paths, sliders, treedef


@app.cell(hide_code=True)
def _(mo, names, sliders):
    mo.vstack(
        [
            mo.md(f"{name} = {slider.value}")
            for slider, name in zip(sliders, names, strict=False)
        ]
    )
    button = mo.ui.run_button(label="Run", keyboard_shortcut="Ctrl-Space")
    return (button,)


@app.cell(hide_code=True)
def _(button, mo, sliders):
    mo.vstack([button, sliders])
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(button, jax, jnp, mo, params, sim_jit, sliders, treedef):
    # all the plotting goes in this cell
    # first bit is boiler plate
    mo.stop(not button.value)

    _leaves = [jnp.array(jnp.pi * slider.value) for slider in sliders]
    _params = jax.tree.unflatten(treedef, _leaves)
    # print(_params)
    # _pr = sim_jit.probability(_params)

    # # can add more calculations, plotting here
    # # cfi = classical_fisher_information(_params)

    # fig, ax = plt.subplots()
    # ket = sim.forward(params)
    # print(ket)
    # if len(ket.shape) == 2:
    #     sns.heatmap(jnp.abs(ket), ax=ax)
    # elif len(ket.shape) == 1:
    #     sns.heatmap(jnp.abs(ket[None, :]), ax=ax)

    get = lambda pytree: jnp.array(
        [pytree.ops["phase"].phi]
    )
    cfim = sim_jit.prob.cfim(get, params)

    # # add it all to the markdown/HTML
    mo.vstack(
        [
            # mo.as_html(fig),
            mo.md(f"CFI: {cfim}"),
            # mo.md(f"Total probability: {_pr.sum()}")
        ]
    )



    return cfim, get


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

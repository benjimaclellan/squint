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
    from squint_dev.ops import ElectroOpticModulator, SinglePhotonComb

    from squint.circuit import Circuit
    from squint.ops.fock import TwoModeSqueezingGate, BeamSplitter, FockState, Phase
    from squint.utils import extract_paths

    return (
        BeamSplitter,
        Circuit,
        ElectroOpticModulator,
        FockState,
        Phase,
        TwoModeSqueezingGate,
        SinglePhotonComb,
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
def _(Circuit, ElectroOpticModulator, SinglePhotonComb, jnp):
    # circuit = Circuit()
    # m = 2
    # dim = 3
    # for i in range(m):
    #     circuit.add(FockState(wires=(i,), n=(1,)))

    # circuit.add(BeamSplitter(wires=(0, 1)))
    # circuit.add(Phase(wires=(0,), phi=0.2), "phase")
    # circuit.add(BeamSplitter(wires=(0, 1)))

    dim = 12

    circuit = Circuit()
    circuit.add(SinglePhotonComb(wires=(0,), modes=1, spacing=3))
    circuit.add(ElectroOpticModulator(wires=(0,), amplitudes=jnp.array(0.1)), "eom")

    # circuit.add(EntangledPhotonComb(wires=(0, 1), modes=3, spacing=3))
    # circuit.add(
    #     SharedGate(
    #         op=ElectroOpticModulator(
    #             wires=(0,),
    #             amplitudes=jnp.ones(
    #                 1,
    #             )
    #             * 0.1,
    #         ),
    #         wires=(1,),
    #     ),
    #     name,
    # )

    return circuit, dim


@app.cell(hide_code=True)
def _(circuit, dim, eqx, extract_paths, jax, mo):
    circuit.verify()

    params, static = eqx.partition(
        circuit,
        eqx.is_inexact_array,
    )

    sim = circuit.compile(params, static, dim=dim, optimize="greedy")
    sim_jit = sim.jit()

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
    return (
        leaves,
        names,
        params,
        paths,
        sim,
        sim_jit,
        sliders,
        static,
        treedef,
    )


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
def _(params, sim):
    # def classical_fisher_information(params):
    #     grad = sim_jit.grad(params)
    #     pr = sim_jit.probability(params)
    #     return (grad.ops["phase"].phi ** 2 / (pr + 1e-12)).sum()

    sim.forward(params)
    return


@app.cell(hide_code=True)
def _(button, jax, jnp, mo, params, plt, sim, sim_jit, sliders, treedef):
    # all the plotting goes in this cell
    # first bit is boiler plate
    mo.stop(not button.value)

    _leaves = [jnp.array(jnp.pi * slider.value) for slider in sliders]
    _params = jax.tree.unflatten(treedef, _leaves)
    print(_params)
    _pr = sim_jit.probability(_params)

    # can add more calculations, plotting here
    # cfi = classical_fisher_information(_params)

    fig, ax = plt.subplots()
    ket = sim.forward(params)
    print(ket)
    # if len(ket.shape) == 2:
    #     sns.heatmap(jnp.abs(ket), ax=ax)
    # elif len(ket.shape) == 1:
    #     sns.heatmap(jnp.abs(ket[None, :]), ax=ax)

    # # add it all to the markdown/HTML
    # mo.vstack(
    #     [
    #         mo.as_html(fig),
    #         # mo.md(f"CFI: {cfi}"),
    #         # mo.md(f"Total probability: {_pr.sum()}")
    #     ]
    # )
    return ax, fig, ket


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

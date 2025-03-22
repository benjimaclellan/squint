import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Benchmarking `squint`""")
    return


@app.cell
def _():
    import itertools
    import timeit
    from functools import partial
    from typing import Literal

    import equinox as eqx
    import hvplot.polars
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    from beartype import beartype
    from rich.pretty import pprint

    from squint.circuit import Circuit
    from squint.ops.base import SharedGate
    from squint.ops.dv import Conditional, DiscreteState, HGate, Phase, XGate

    return (
        Circuit,
        Conditional,
        DiscreteState,
        HGate,
        Literal,
        Phase,
        SharedGate,
        XGate,
        beartype,
        eqx,
        hvplot,
        itertools,
        jax,
        jnp,
        partial,
        pl,
        plt,
        pprint,
        sns,
        timeit,
    )


@app.cell
def _(
    Circuit,
    Conditional,
    DiscreteState,
    HGate,
    Phase,
    SharedGate,
    XGate,
    eqx,
    jnp,
    pprint,
):
    def circuit_factory(dim: int, n: int, n_phi: int, depth: int):
        circuit = Circuit()
        for i in range(n):
            circuit.add(DiscreteState(wires=(i,), n=(0,)))

        circuit.add(HGate(wires=(0,)))
        for i in range(n - 1):
            circuit.add(Conditional(gate=XGate, wires=(i, i + 1)))

        block_width = n // n_phi
        for k, phi in enumerate(range(n_phi)):
            wires = [i + k * block_width for i in (range(1, block_width))]

            circuit.add(
                SharedGate(
                    op=Phase(wires=(k * block_width,), phi=0.1 * jnp.pi),
                    wires=tuple(wires),
                ),
                f"phase{k}",
            )

        for ell in range(depth):
            for i in range(n):
                circuit.add(HGate(wires=(i,)))

        for i in range(n):
            circuit.add(HGate(wires=(i,)))

        get = lambda pytree: jnp.array(
            [pytree.ops[f"phase{k}"].op.phi for k in range(n_phi)]
        )

        return circuit, get

    circuit, get = circuit_factory(dim=2, n=2, n_phi=1, depth=2)
    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    sim = circuit.compile(params, static, dim=2, optimize="greedy")

    pprint(circuit)
    return circuit, circuit_factory, get, params, sim, static


@app.cell
def _(
    Literal,
    beartype,
    circuit_factory,
    eqx,
    itertools,
    jax,
    partial,
    pl,
    timeit,
):
    @beartype
    def benchmark(
        dim: int,
        n: int,
        n_phi: int,
        depth: int,
        jit: bool,
        device: Literal["cpu", "gpu"],
    ):
        circuit, get = circuit_factory(dim, n, n_phi, depth)
        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(params, static, dim=dim, optimize="greedy")
        if jit:
            sim = sim.jit(device=jax.devices(device)[0])

        times = {
            "prob.forward": timeit.Timer(partial(sim.prob.forward, params)).repeat(
                3, 1
            ),
            "prob.grad": timeit.Timer(partial(sim.prob.grad, params)).repeat(3, 1),
            "prob.cfim": timeit.Timer(partial(sim.prob.cfim, get, params)).repeat(3, 1),
        }
        return {
            "dim": dim,
            "n": n,
            "n_phi": n_phi,
            "depth": depth,
            "jit": jit,
            "device": device,
            "max(prob.forward)": max(times["prob.forward"]),
            "max(prob.grad)": max(times["prob.grad"]),
            "max(prob.cfim)": max(times["prob.cfim"]),
            "min(prob.forward)": min(times["prob.forward"]),
            "min(prob.grad)": min(times["prob.grad"]),
            "min(prob.cfim)": min(times["prob.cfim"]),
        }

    def batch(dims, ns, n_phis, depths, jits, devices):
        df = []
        config = list(itertools.product(dims, ns, n_phis, depths, jits, devices))
        for i, (dim, n, n_phi, depth, jit, device) in enumerate(config):
            print(dim, n, n_phi, depth, jit, device)
            df.append(benchmark(dim, n, n_phi, depth, jit, device))
        return pl.DataFrame(df)

    return batch, benchmark


@app.cell
def _(batch):
    df1 = batch(
        dims=(1, 2, 3, 4),
        ns=(4,),
        n_phis=(1,),
        depths=(0,),
        jits=(True,),
        devices=("cpu",),
    )
    return (df1,)


@app.cell
def _():
    # df1.hvplot.scatter(x="dim", y="min(prob.forward)")
    return


@app.cell
def _(batch):
    df2 = batch(
        dims=(2,),
        ns=list(range(2, 16)),
        n_phis=(2,),
        depths=(0,),
        jits=(True, False),
        devices=("cpu", "gpu"),
    )
    return (df2,)


@app.cell
def _(df2, sns):
    _g = sns.FacetGrid(df2, row="jit", col="jit", hue="device")
    _g.map(sns.lineplot, "n", "min(prob.cfim)", markers="o")
    _g.add_legend()
    return


@app.cell
def _(batch):
    df3 = batch(
        dims=(2, 3, 4),
        ns=list(range(2, 6)),
        n_phis=(1,),
        depths=(0,),
        jits=(True, False),
        devices=("cpu", "gpu"),
        # devices = ('cpu', )
    )
    return (df3,)


@app.cell
def _(df3, sns):
    _g = sns.FacetGrid(df3, row="jit", col="dim", hue="device")
    _g.map(sns.lineplot, "n", "min(prob.cfim)", markers="o")
    _g.add_legend()
    return


@app.cell
def _(df3, pl, sns):
    _g = sns.FacetGrid(df3.filter(pl.col("jit") == True), col="dim", hue="device")
    _g.map(sns.lineplot, "n", "min(prob.cfim)", markers="o")
    _g.add_legend()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

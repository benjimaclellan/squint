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
    import polars as pl
    import seaborn as sns
    from rich.pretty import pprint
    from functools import partial
    import timeit
    import hvplot.polars

    from squint.circuit import Circuit
    from squint.ops.dv import DiscreteState, HGate, Phase, Conditional, XGate
    from squint.ops.base import SharedGate
    return (
        Circuit,
        Conditional,
        DiscreteState,
        HGate,
        Phase,
        SharedGate,
        XGate,
        eqx,
        hvplot,
        itertools,
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
            wires = [i + k * block_width for i in(range(1, block_width))]

            circuit.add(
                SharedGate(op=Phase(wires=(k * block_width,), phi=0.1 * jnp.pi), wires=tuple(wires)),
                f"phase{k}",
            )

        for ell in range(depth):
            for i in range(n):
                circuit.add(HGate(wires=(i,)))

        for i in range(n):
            circuit.add(HGate(wires=(i,)))

        get = lambda pytree: jnp.array([pytree.ops[f"phase{k}"].op.phi for k in range(n_phi)])

        return circuit, get

    circuit, get = circuit_factory(dim=2, n=2, n_phi=1, depth=2)
    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    sim = circuit.compile(params, static, dim=2, optimize='greedy')

    pprint(circuit)
    return circuit, circuit_factory, get, params, sim, static


@app.cell
def _(circuit_factory, eqx, itertools, partial, pl, timeit):
    def benchmark(dim: int, n: int, n_phi: int, depth: int, jit: bool):
        circuit, get = circuit_factory(dim, n, n_phi, depth)
        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(params, static, dim=dim, optimize='greedy')
        if jit:
            sim = sim.jit()

        times = {
            "prob.forward":  timeit.Timer(partial(sim.prob.forward, params)).repeat(3, 1),
            "prob.grad":  timeit.Timer(partial(sim.prob.grad, params)).repeat(3, 1),
            "prob.cfim":  timeit.Timer(partial(sim.prob.cfim, get, params)).repeat(3, 1),
        }
        return {
            'dim': dim,
            'n': n,
            'n_phi': n_phi,
            'depth': depth,
            'jit': jit,
             "max(prob.forward)":  max(times['prob.forward']),
             "max(prob.grad)":  max(times['prob.grad']),
             "max(prob.cfim)":  max(times['prob.cfim']),
             "min(prob.forward)":  min(times['prob.forward']),
             "min(prob.grad)":  min(times['prob.grad']),
             "min(prob.cfim)":  min(times['prob.cfim']),
        }


    ns = (1, 2, 3, 4, 5, 6)
    dims = (2, 3)
    depths = (0, )
    jits = (True,)
    n_phis = (1, 2)

    df = []

    def factorize(num):
        return [n for n in range(1, num + 1) if num % n == 0]


    config = list(itertools.product(dims, ns, n_phis, depths, jits))
    for i, (dim, n, n_phi, depth, jit) in enumerate(config):
        # for n_phi in (1,):
        # for n_phi in factorize(n):
        print(dim, n, n_phi, depth, jit)
        df.append(benchmark(dim, n, n_phi, depth, jit))

    df = pl.DataFrame(df)
    return (
        benchmark,
        config,
        depth,
        depths,
        df,
        dim,
        dims,
        factorize,
        i,
        jit,
        jits,
        n,
        n_phi,
        n_phis,
        ns,
    )


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.hvplot.scatter(x="n", y="min(prob.forward)")
    return


@app.cell
def _(df):
    df.hvplot.scatter(x="n", y="min(prob.cfim)")
    return


@app.cell
def _(df):
    df.hvplot.scatter(x="n", y="min(prob.cfim)", color="n_phi")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

# %%
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from squint.circuit import Circuit, compile

# from squint.diagram import draw
from squint.ops.base import SharedGate
from squint.ops.dv import (
    DiscreteVariableState,
    HGate,
    RXGate,
    RXXGate,
    RYGate,
    RZGate,
)
from squint.utils import partition_op


@pytest.mark.parametrize(
    "n",
    [
        2,
    ],
)
def test_optimization_heisenberg_limited(n):
    dim = 2

    keys = jr.split(jr.PRNGKey(1234), 1000)
    idx = itertools.count(0)

    circuit = Circuit(backend="pure")
    for i in range(n):
        circuit.add(DiscreteVariableState(wires=(i,), n=(0,)))

    for k in range(3):
        for i in range(n):
            circuit.add(RXGate(wires=(i,), phi=jr.normal(keys[next(idx)]).item()))
            circuit.add(RYGate(wires=(i,), phi=jr.normal(keys[next(idx)]).item()))
        for i in range(0, n - 1, 2):
            circuit.add(
                RXXGate(wires=(i, i + 1), angle=jr.normal(keys[next(idx)]).item())
            )
        for i in range(1, n - 1, 2):
            circuit.add(
                RXXGate(wires=(i, i + 1), angle=jr.normal(keys[next(idx)]).item())
            )

    circuit.add(
        SharedGate(op=RZGate(wires=(0,), phi=0.1 * jnp.pi), wires=tuple(range(1, n))),
        "phase",
    )
    for i in range(n):
        circuit.add(HGate(wires=(i,)))

    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    params_est, params_opt = partition_op(params, "phase")
    params = (params_est, params_opt)

    sim = compile(static, dim, *params, **{"optimize": "greedy", "argnum": 0})  # .jit()

    print(sim.amplitudes.forward(*params))
    print(sim.probabilities.forward(*params).sum())

    print(sim.probabilities.cfim(*params))
    print(sim.amplitudes.qfim(*params))

    lr = 1e-3
    optimizer = optax.chain(optax.adam(lr), optax.scale(-1.0))
    opt_state = optimizer.init(params_opt)

    def loss(params_est, params_opt):
        return sim.probabilities.cfim(params_est, params_opt).squeeze()

    value_and_grad = jax.value_and_grad(loss, argnums=1)

    @jax.jit
    def step(opt_state, params_est, params_opt):
        val, grad = value_and_grad(params_est, params_opt)
        updates, opt_state = optimizer.update(grad, opt_state)
        params_opt = optax.apply_updates(params_opt, updates)
        return params_opt, opt_state, val

    _ = step(opt_state, params_est, params_opt)

    cfims = []
    for _ in range(3000):
        params_opt, opt_state, val = step(opt_state, params_est, params_opt)
        cfims.append(val)

    assert jnp.abs(val - n**2) < 0.5, (
        f"Optimization did not converge to Heiseberg limit for n={n}, final value {val}"
    )


if __name__ == "__main__":
    test_optimization_heisenberg_limited(n=4)

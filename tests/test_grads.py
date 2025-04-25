# %%
import equinox as eqx
import jax.numpy as jnp
import pytest
import functools
import jax
from typing import Literal, Sequence
from jaxtyping import PyTree
import paramax
import jax.tree_util as jtu
import jax.random as jr
import itertools
import optax 
import time 

import squint
from squint.circuit import Circuit, get_symbol, compile_experimental
from squint.ops.base import SharedGate
from squint.ops.dv import Conditional, DiscreteVariableState, HGate, RZGate, XGate, RXGate, RYGate, RXXGate
from squint.ops.noise import BitFlipChannel, DepolarizingChannel, ErasureChannel
from squint.utils import partition_op
from squint.ops.base import AbstractErasureChannel
from squint.diagram import draw

#%%

@pytest.mark.parametrize('n', [2,])
def test_optimization_heisenberg_limited(n):
    dim = 2

    keys = jr.split(jr.PRNGKey(1234), 1000)
    idx = itertools.count(0)

    circuit = Circuit(backend="pure")
    for i in range(n):
        circuit.add(DiscreteVariableState(wires=(i,), n=(0,)))

    # for i in range(n - 1):
            # circuit.add(Conditional(gate=XGate, wires=(i, i + 1)))
            
    for k in range(3):
        for i in range(n):
            circuit.add(RXGate(wires=(i,), phi=jr.normal(keys[next(idx)]).item()))
            circuit.add(RYGate(wires=(i,), phi=jr.normal(keys[next(idx)]).item()))
        for i in range(0, n-1, 2):
            circuit.add(RXXGate(wires=(i, i + 1), angle=jr.normal(keys[next(idx)]).item()))
        for i in range(1, n-1, 2):
            circuit.add(RXXGate(wires=(i, i + 1), angle=jr.normal(keys[next(idx)]).item()))

    circuit.add(
        SharedGate(op=RZGate(wires=(0,), phi=0.1 * jnp.pi), wires=tuple(range(1, n))),
        "phase",
    )
    for i in range(n):
        circuit.add(HGate(wires=(i,)))

    #%%
    fig = draw(circuit, "mpl")
    fig.show()

    #%%
    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    # params = (params,)

    params_est, params_opt = partition_op(params, "phase")
    # params1, params2 = partition_op(params1, "dummy1")
    params = (params_est, params_opt)

    #%%
    sim = compile_experimental(static, dim, *params, **{'optimize': 'greedy', 'argnum': 0})#.jit()

    #%%
    print(sim.amplitudes.forward(*params))
    print(sim.probabilities.forward(*params).sum())

    # print(jax.tree.flatten(sim.amplitudes.grad(*params)))
    # print(jax.tree.flatten(sim.probabilities.grad(*params)))

    print(sim.probabilities.cfim(*params))
    print(sim.amplitudes.qfim(*params))


    #%% 

    lr = 1e-3
    optimizer = optax.chain(optax.adam(lr), optax.scale(-1.0))
    opt_state = optimizer.init(params_opt)


    def loss(params_est, params_opt):
        return sim.probabilities.cfim(params_est, params_opt).squeeze()


    value_and_grad = jax.value_and_grad(loss, argnums=1)


    #%%
    @jax.jit
    def step(opt_state, params_est, params_opt):
        val, grad = value_and_grad(params_est, params_opt)
        updates, opt_state = optimizer.update(grad, opt_state)
        params_opt = optax.apply_updates(params_opt, updates)
        return params_opt, opt_state, val


    _ = step(opt_state, params_est, params_opt)

    # %%
    cfims = []
    for _ in range(3000):
        params_opt, opt_state, val = step(opt_state, params_est, params_opt)
        cfims.append(val)
        print(val)

    assert jnp.abs(val - n**2) < 0.5, f"Optimization did not converge to Heiseberg limit for n={n}, final value {val}"
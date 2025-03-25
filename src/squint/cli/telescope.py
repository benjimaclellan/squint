# %%
import functools

import time
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm
from pydantic import BaseModel
from rich.pretty import pprint
import jax.random as jr

from squint.circuit import Circuit
from squint.ops.fock import LOPC, FockState, Phase
from squint.utils import print_nonzero_entries

jax.config.update("jax_enable_x64", True)


# %%
class TelescopeArgs(BaseModel):
    n: int  # number of ancilla photons
    cut: int  # fock cutoff
    n_steps: int = 300
    lr0: float = 1e-1
    lr1: float = 1e-3
    key: int = 1234

    def make(self):
        cut = self.cut
        n = self.n  # number of resource photons
        m = self.n + 1  # number of modes

        circuit = Circuit()
        key = jr.PRNGKey(self.key)
        subkeys = jr.split(key, 3)
        
        # we add in the stellar photon, which is in an even superposition of spatial modes 0 and 2 (left and right telescopes)
        for i in range(m):
            circuit.add(
                FockState(
                    wires=(
                        i,
                        m + i,
                    ),
                    n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
                )
            )

        circuit.add(Phase(wires=(0,), phi=0.01), "phase")

        circuit.add(LOPC(wires=tuple(range(0, m)), rs=jnp.ones(m * (m - 1) // 2) * 0.1))
        circuit.add(LOPC(wires=tuple(range(m, 2 * m)), rs=jnp.ones(m * (m - 1) // 2) * 0.1))

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        circuit = eqx.combine(params, static)  # todo: check this is correct syntax
        return circuit


def telescope(args: TelescopeArgs):
    t0 = time.time()
    
    cut = args.cut

    circuit = args.make()
    # params, static = partition_op(circuit, "phase")
    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    sim = circuit.compile(params, static, dim=cut, optimize="greedy")  # .jit()
    get = lambda pytree: jnp.array([pytree.ops["phase"].phi])
    pprint(circuit)  # ket = sim.amplitudes.grad(params)

    prob = sim.prob.forward(params)
    print_nonzero_entries(prob)

    # %% Differentiate with respect to parameters of interest
    def _loss_fn(params, sim, get):
        return sim.prob.cfim(get, params).squeeze()

    loss_fn = functools.partial(_loss_fn, sim=sim, get=get)
    print(
        f"Classical Fisher information of starting parameterization is {loss_fn(params)}"
    )

    # %%
    
    # lr_schedule = optax.linear_schedule(
    #     init_value=args.lr0,
    #     end_value=args.lr1,
    #     transition_steps=args.n_steps
    # )
    
    # optimizer = optax.chain(optax.sgd(lr_schedule), optax.scale(-1.0))
    optimizer = optax.chain(optax.adam(args.lr0), optax.scale(-1.0))
    opt_state = optimizer.init(params)

    # %%
    @jax.jit
    def update(_params, _opt_state):
        _val, _grad = jax.value_and_grad(loss_fn)(_params)
        _updates, _opt_state = optimizer.update(_grad, _opt_state)
        _params = optax.apply_updates(_params, _updates)
        return _params, _opt_state, _val

    # %%
    update(params, opt_state)

    pbar = tqdm.tqdm(range(args.n_steps), desc="Training", unit="it")

    cfims, steps = [], []
    for step in pbar:
        params, opt_state, val = update(params, opt_state)

        pbar.set_postfix({"loss": f"{val:0.4f}"})
        pbar.update(1)

        cfims.append(val)
        steps.append(step)

    datasets = dict(cfims=jnp.array(cfims), steps=jnp.array(steps), runtime=jnp.array(time.time() - t0))
    circuit = eqx.combine(params, static)  # todo: check this is correct syntax

    return circuit, args, datasets

# %%

import io
import pathlib

import equinox as eqx
import h5py
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from typing import Literal

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import polars as pl
import tqdm
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.fock import LOPC, FockState, Phase
from squint.utils import print_nonzero_entries

jax.config.update("jax_enable_x64", True)

from pydantic import BaseModel

#%%
class TelescopeArgs(BaseModel):
    path: pathlib.Path
    filename: str
    m: int
    cut: int
    n_steps: int = 300
    lr: float = 1e-1
    
    def make(self):
        cut = self.cut 
        m = self.m

        circuit = Circuit()

        # we add in the stellar photon, which is in an even superposition of spatial modes 0 and 2 (left and right telescopes)
        circuit.add(
            FockState(
                wires=(
                    0,
                    m,
                ),
                n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
            )
        )
        circuit.add(FockState(wires=(1,), n=(2,)))
        for i in range(2 * m):
            if i in (0, 1, m):
                continue
            circuit.add(FockState(wires=(i,), n=[(1.0, (0,))]))


        circuit.add(Phase(wires=(0,), phi=0.01), "phase")


        circuit.add(
            LOPC(
                wires=tuple(list(range(1, m)) + list(range(m + 1, 2 * m))),
                rs=jnp.ones(m * (m - 1) // 2) * 0.1,
            )
        )

        circuit.add(LOPC(wires=tuple(range(0, m)), rs=jnp.ones(m * (m - 1) // 2) * 0.1))
        circuit.add(LOPC(wires=tuple(range(m, 2 * m)), rs=jnp.ones(m * (m - 1) // 2) * 0.1))

        return circuit
    


def telescope(args: TelescopeArgs):

    path, filename = args.path, args.filename
    cut = args.cut
    filepath = pathlib.Path(path).joinpath(f"{filename}-{args.m}.h5")
    filepath.parent.mkdir(exist_ok=True, parents=True)
    print(filepath)

    circuit = args.make()
    # params, static = partition_op(circuit, "phase")
    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    sim = circuit.compile(params, static, dim=cut, optimize="greedy")  # .jit()
    get = lambda pytree: jnp.array([pytree.ops["phase"].phi])
    pprint(circuit)# ket = sim.amplitudes.grad(params)

    prob = sim.prob.forward(params)
    # grad = sim.prob.grad(params)
    print_nonzero_entries(prob)
    
    # %% Differentiate with respect to parameters of interest
    def _loss_fn(params, sim, get):
        return sim.prob.cfim(get, params).squeeze()


    loss_fn = functools.partial(_loss_fn, sim=sim, get=get)
    print(f"Classical Fisher information of starting parameterization is {loss_fn(params)}")

    # %%
    start_learning_rate = args.lr
    optimizer = optax.chain(optax.adam(start_learning_rate), optax.scale(-1.0))
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

        pbar.set_postfix({"loss": val})
        pbar.update(1)

        # df.append({"cfim": val, "step": step})
        cfims.append(val)
        steps.append(step)


    # save(filepath=filepath, args=args, model=circuit)

    datasets = dict(cfims=jnp.array(cfims), ps=jnp.array(steps))
    
    # with h5py.File(filepath, "a") as f:
    #     for key, value in datasets.items():
    #         f.create_dataset(key, data=value)
    return circuit, args, datasets

    # df = pl.DataFrame(df)

    # # %%
    # fig, ax = plt.subplots()
    # ax.plot(df["step"], df["cfim"])
    # fig.show()

    # # %%
    # prob = sim.prob.forward(params)
    # print_nonzero_entries(prob)
    eqx.tree_pprint(params, short_arrays=False)
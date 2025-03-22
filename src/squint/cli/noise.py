# %%

import io
import pathlib

import equinox as eqx
import h5py
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from typing import Literal

from squint.circuit import Circuit
from squint.ops.base import SharedGate
from squint.ops.dv import Conditional, DiscreteState, HGate, Phase, XGate
from squint.ops.noise import BitFlipChannel, DepolarizingChannel
from squint.utils import print_nonzero_entries

from pydantic import BaseModel

#%%
class Args(BaseModel):
    path: pathlib.Path
    filename: str
    n: int
    state: Literal['ghz']
    channel: Literal["depolarizing", "bitflip"]
    loc: Literal["state", "measurement"]

    def make(self):
        n, state, channel, loc = self.n, self.state, self.channel, self.loc
        
        if channel == "depolarizing":
            Channel = DepolarizingChannel
        elif channel == "bitflip":
            Channel = BitFlipChannel

        circuit = Circuit()
        for i in range(n):
            circuit.add(DiscreteState(wires=(i,), n=(0,)))

        circuit.add(HGate(wires=(0,)))
        for i in range(n - 1):
            circuit.add(Conditional(gate=XGate, wires=(i, i + 1)))

        if loc == "state":
            circuit.add(
                SharedGate(op=Channel(wires=(0,), p=0.2), wires=tuple(range(1, n))),
                "noise",
            )

        circuit.add(
            SharedGate(op=Phase(wires=(0,), phi=0.1 * jnp.pi), wires=tuple(range(1, n))),
            "phase",
        )

        for i in range(n):
            circuit.add(HGate(wires=(i,)))

        if loc == "measurement":
            circuit.add(
                SharedGate(op=Channel(wires=(0,), p=0.2), wires=tuple(range(1, n))),
                "noise",
            )
        return circuit
    
    
# %%
@beartype
def noise(
    # path: pathlib.Path, filename: str, n: int, state: str, channel: str, loc: str
    args: Args
):
    dim = 2
    path, filename = args.path, args.filename
    n, state, channel, loc = args.n, args.state, args.channel, args.loc
    
    # hyperparameters = dict(state=state, channel=channel, n=n, loc=loc)

    filepath = pathlib.Path(path).joinpath(f"{filename}-{n}-{state}-{channel}-{loc}.h5")
    filepath.parent.mkdir(exist_ok=True, parents=True)
    print(filepath)
    # circuit = make(**hyperparameters)
    circuit = args.make()

    params, static = eqx.partition(circuit, eqx.is_inexact_array)

    get = lambda pytree: jnp.array([pytree.ops["phase"].op.phi])

    path = circuit.path(dim=dim)
    sim = circuit.compile(params, static, dim=2).jit()
    print_nonzero_entries(sim.prob.forward(params))

    params = eqx.tree_at(lambda pytree: pytree.ops["noise"].op.p, params, 0.1)
    print(sim.prob.cfim(get, params))

    ps = jnp.logspace(-6, 0, 250)
    params = eqx.tree_at(lambda pytree: pytree.ops["noise"].op.p, params, ps)
    params = eqx.tree_at(
        lambda pytree: pytree.ops["phase"].op.phi, params, jnp.ones_like(ps) * 0.01
    )
    cfims = eqx.filter_vmap(sim.prob.cfim, in_axes=(None, 0))(get, params)

    # save(filepath=filepath, hyperparameters=hyperparameters, model=circuit)

    # datasets = dict(cfims=cfims, ps=np.array(ps))
    # with h5py.File(filepath, "a") as f:
    #     for key, value in datasets.items():
    #         f.create_dataset(key, data=value)

    return

def save(filepath: pathlib.Path, hyperparameters: dict, model: eqx.Module):
    with h5py.File(filepath, "a") as f:
        hyperparameters_group = f.create_group("hyperparameters")
        for key, value in hyperparameters.items():
            hyperparameters_group.attrs[key] = value
        buf = io.BytesIO()
        eqx.tree_serialise_leaves(buf, model)
        buf.seek(0)
        f.create_dataset("circuit", data=np.void(buf.getvalue()))


def load(filepath: pathlib.Path):
    with h5py.File(filepath, "r") as f:
        hyperparameters = dict(f["hyperparameters"].attrs)
        model = make(**hyperparameters)
        buf = io.BytesIO(f["circuit"][()].tobytes())
        deserialized = eqx.tree_deserialise_leaves(buf, model)
        return model, hyperparameters


if __name__ == "__main__":
    args = Args(path=pathlib.Path("data/"), filename="test", n=4, channel="bitflip", state="ghz", loc="state")
    print(args)
    noise(args)
    args.model_dump_json()
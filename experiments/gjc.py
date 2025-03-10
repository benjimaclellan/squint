# %%
import itertools
import functools 

import tqdm
import jax
import optax
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from rich.pretty import pprint
import polars as pl
from squint.circuit import Circuit
from squint.ops.fock import BeamSplitter, FockState, Phase
from squint.utils import partition_op, print_nonzero_entries

# %%  Express the optical circuit.
# ------------------------------------------------------------------
cut = 4
circuit = Circuit()


# we add in the stellar photon, which is in an even superposition of spatial modes 0 and 2 (left and right telescopes)
circuit.add(
    FockState(
        wires=(
            0,
            2,
        ),
        n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
    )
)
# the stellar photon accumulates a phase shift in left telescope. 
circuit.add(Phase(wires=(0,), phi=0.01), "phase")

# we add the resources photon, which is in an even superposition of spatial modes 1 and 3
circuit.add(
    FockState(
        wires=(
            1,
            3,
        ),
        n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
    )
)

# we add the linear optical circuit at each telescope
circuit.add(
    BeamSplitter(
        wires=(
            0,
            1,
        ),
        r=0.1,
    )
)
circuit.add(
    BeamSplitter(
        wires=(
            2,
            3,
        ),
        r=0.1,
    )
)
pprint(circuit)
   
# we split out the params which can be varied (in this example, it is just the "phase" phi value), and all the static parameters (wires, etc.)
# params, static = partition_op(circuit, "phase")
params, static = eqx.partition(circuit, eqx.is_inexact_array)

#%%
# next we compile the circuit description into function calls, which compute, e.g., the quantum state, probabilities, partial derivates of the quantum state, and partial derivatives of the probabilities
sim = circuit.compile(params, static, dim=cut, optimize="greedy").jit()

# we define a function which indexes in the circuit object, and all other pytrees computed from it, a specific value. this will be necessary to access, e.g., the gradients
get = lambda pytree: jnp.array([pytree.ops["phase"].phi])

pprint(circuit)
circuit.verify()

# %%
prob = sim.prob.forward(params)
print_nonzero_entries(prob)

# %% Differentiate with respect to parameters of interest
def _loss_fn(params, sim, get):
    return sim.prob.cfim(get, params).squeeze()


loss_fn = functools.partial(_loss_fn, sim=sim, get=get)
print(f"Classical Fisher information of starting parameterization is {loss_fn(params)}")

# %%
start_learning_rate = 1e-1
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
df = []
update(params, opt_state)
n_steps = 300
pbar = tqdm.tqdm(range(n_steps), desc="Training", unit="it")
for step in pbar:
    params, opt_state, val = update(params, opt_state)

    pbar.set_postfix({"loss": val})
    pbar.update(1)

    df.append({'cfim': val, 'step': step})

df = pl.DataFrame(df)

#%%
fig, ax = plt.subplots()
ax.plot(df['step'], df['cfim'])
fig.show()

#%%
prob = sim.prob.forward(params)
print_nonzero_entries(prob)
eqx.tree_pprint(params, short_arrays=False)

# %%

# %%

import equinox as eqx
import jax.numpy as jnp
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.fock import BeamSplitter, FockState, Phase
from squint.utils import print_nonzero_entries

# %%  Express the optical circuit.
# ------------------------------------------------------------------
cutoff = 4
circuit = Circuit()


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


# circuit.add(BeamSplitter(wires=(2, 3,), r=jnp.pi/4.5))

pprint(circuit)
circuit.verify()

# %%
##%% split into training parameters and static
# ------------------------------------------------------------------
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    # is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)
print(params)

# %%
sim = circuit.compile(params, static, dim=cutoff, optimize="greedy")
sim_jit = sim.jit()

##%%
tensor = sim.forward(params)
pr = sim.probability(params)
print_nonzero_entries(pr)

# %%
key = jr.PRNGKey(0)
samples = sim.sample(key, params, shape=(4, 5))
print(samples)

# %% Differentiate with respect to parameters of interest
name = "phase"
# params, static = partition_op(circuit, name)
# sim = circuit.compile(params, static, dim=cutoff, optimize="greedy")
# sim_jit = sim.jit()


def classical_fisher_information(params):
    pr = sim.probability(params)
    grad = sim.grad(params)
    cfi = (grad.ops["phase"].phi ** 2 / (pr + 1e-12)).sum()
    return cfi


# %%
cfim = classical_fisher_information(params)
value_and_grad = jax.value_and_grad(classical_fisher_information)

val, grad = value_and_grad(params)
print(grad.ops["phase"].phi)
print(val)

# %%
start_learning_rate = 1e-1
optimizer = optax.chain(optax.adam(start_learning_rate), optax.scale(-1.0))
opt_state = optimizer.init(params)


# %%
@jax.jit
def step(_params, _opt_state):
    _val, _grad = value_and_grad(_params)
    _updates, _opt_state = optimizer.update(_grad, _opt_state)
    _params = optax.apply_updates(_params, _updates)
    return _params, _opt_state, _val


# %%
cfims = []
step(params, opt_state)
pbar = tqdm.tqdm(range(300), desc="Training", unit="step")
for _ in pbar:
    params, opt_state, val = step(params, opt_state)
    cfims.append(val)
    pbar.set_postfix({"loss": val})
    pbar.update(1)

pr = sim.probability(params)
print_nonzero_entries(pr)

print(classical_fisher_information(params))
# # %%
# %%

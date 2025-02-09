# %%

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import paramax
from rich.pretty import pprint

from qtelescope.ops import BeamSplitter, Circuit, FockState, Phase
from qtelescope.utils import print_nonzero_entries

jax.config.update("jax_enable_x64", True)

# %%  Express the optical circuit.
# ------------------------------------------------------------------
cutoff = 6
circuit = Circuit()

# circuit.add(
#     FockState(
#         wires=(0, 1),
#         n=[
#             # (1.0, (3, 0)),
#             (1 / jnp.sqrt(2).item(), (1, 2)),
#             (1 / jnp.sqrt(2).item(), (1, 1)),
#         ],
#     )
# )

circuit.add(FockState(wires=(0,), n=(1,)))
circuit.add(FockState(wires=(1,), n=(1,)))
circuit.add(Phase(wires=(0,), phi=98 * jnp.pi / 100))
circuit.add(BeamSplitter(wires=(0, 1), r=jnp.pi / 0.4))
circuit.add(Phase(wires=(0,), phi=98 * jnp.pi / 100), "phase")
circuit.add(BeamSplitter(wires=(0, 1), r=20 * jnp.pi / 40))
pprint(circuit)

# %% split into training parameters and static
# ------------------------------------------------------------------
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)
# print(params)

sim = circuit.compile(params, static, cut=cutoff, optimize="greedy")
sim = sim.jit()
print(sim.forward(params))
pr = sim.probability(params)
print_nonzero_entries(pr)

# %% split into probe parameters and static, sweep some parameter over
# ------------------------------------------------------------------
# name = "phase"
# params, static = partition_op(circuit, name)
# sim = circuit.compile(params, static, cut=cutoff, optimize="greedy")
# sim_jit = sim.jit()


def classical_fisher_information(_params):
    _grad = sim.grad(_params)
    # cfim = jnp.sum(grad.ops['phase'].phi ** 2 / (pr + 1e-14))
    # cfim = jnp.nansum(grad.ops['phase'].phi ** 2 / (pr + 1e-14))
    A = _grad.ops["phase"].phi ** 2
    B = sim.probability(params)
    cfim = jnp.sum(A * (B / (B**2 + 1e-18)))  # soft-approximation of CFI
    return cfim


def loss(params):
    grad = sim.grad(params)
    A = grad.ops["phase"].phi ** 2
    val = jnp.sum(A)  # soft-approximation of CFI
    return val


cfim = classical_fisher_information(params)  # , phi=0.0)
print(cfim)
value_and_grad = jax.value_and_grad(loss)

# %%
val, grad = value_and_grad(params)
print(grad.ops["phase"].phi)
print(val)

# %%
start_learning_rate = 1e-2

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
for i in range(300):
    params, opt_state, val = step(params, opt_state)
    cfims.append(val)
    print(val, classical_fisher_information(params))
    # print(params.ops['phase'].phi, params.ops[2].r, params.ops[4].r, cfim)  # params.ops[1].r,

pr = sim.probability(params)
print_nonzero_entries(pr)

print(classical_fisher_information(params))
# # %%
# fig, ax = plt.subplots()
# ax.plot(phis, cfims)
# fig.show()

# %%

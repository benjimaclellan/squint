# %%
import equinox as eqx
import jax.numpy as jnp

from squint.circuit import Circuit
from squint.ops.dv import Conditional, DiscreteState, H, Phase, X
from squint.utils import print_nonzero_entries

# %%
n, dim = 3, 3
circuit = Circuit()
for i in range(n):
    circuit.add(DiscreteState(wires=(i,), n=(0,)))

circuit.add(H(wires=(0,)))
for i in range(n - 1):
    circuit.add(Conditional(conditional=X, wires=(i, i + 1)))

for i in range(n):
    circuit.add(Phase(wires=(0,), phi=0.1 * jnp.pi), "phase")

for i in range(n - 1):
    circuit.add(X(wires=(i,)))


params, static = eqx.partition(circuit, eqx.is_inexact_array)
sim = circuit.compile(params, static, dim=dim, optimize="greedy")
pr = sim.probability(params)
print_nonzero_entries(pr)

# cfi = (grad.ops["phase"].phi ** 2 / (pr + 1e-12)).sum()
# print(cfi)
# %%
conditional = X(wires=(0,))
c = Conditional(
    conditional=X,
    wires=(
        0,
        1,
    ),
)
c(dim=3)

# %%
dim = 4
state = DiscreteState(
    wires=(0, 1), n=[(1 / jnp.sqrt(dim).item(), (k, k)) for k in range(dim)]
)
circuit = Circuit()
circuit.add(state)
params, static = eqx.partition(circuit, eqx.is_inexact_array)
sim = circuit.compile(params, static, dim=dim, optimize="greedy")
pr = sim.probability(params)
print_nonzero_entries(pr)

# %%

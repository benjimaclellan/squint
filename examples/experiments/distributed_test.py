# %%

import equinox as eqx
import jax
import jax.numpy as jnp
import paramax
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.base import SharedGate
from squint.ops.distributed import GlobalParameter
from squint.ops.dv import Conditional, DiscreteVariableState, HGate, RZGate, XGate

jax.config.update("jax_enable_x64", True)
# %%
circuit = Circuit()

m = 6

for i in range(m):
    circuit.add(DiscreteVariableState(wires=(i,)))

circuit.add(HGate(wires=(0,)))
circuit.add(HGate(wires=(m // 2,)))

for i in range(0, m // 2 - 1):
    circuit.add(Conditional(gate=XGate, wires=(i, i + 1)))

circuit.add(Conditional(gate=XGate, wires=(m // 2 - 1, m // 2)))

for i in range(m // 2, m - 1):
    circuit.add(Conditional(gate=XGate, wires=(i, i + 1)))


dop = GlobalParameter(
    ops=[RZGate(wires=(i,), phi=0.0) for i in range(m)],
    weights=jnp.ones(shape=(m,)) / m,
)

# for i in range(0, m//2):
# circuit.add(Phase(wires=(0,), phi=0.1), f"phase{i}")

shared1 = SharedGate(op=RZGate(wires=(0,), phi=0.1), wires=tuple(range(1, m // 2)))
# shared2 = SharedGate(op=Phase(wires=(m//2,), phi=0.2), wires=())
shared2 = SharedGate(
    op=RZGate(wires=(m // 2,), phi=0.2), wires=tuple(range(m // 2 + 1, m))
)
# shared = SharedGate(op=Phase(wires=(0,), phi=0.1), wires=tuple(range(1, m)))

circuit.add(shared1, "phases1")
circuit.add(shared2, "phases2")
# circuit.add(shared, "phases")
# circuit.add(dop, "phases")

for i in range(m):
    circuit.add(HGate(wires=(i,)))

pprint(circuit)

# %%
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)

sim = circuit.compile(params, static, dim=2, optimize="greedy")
pprint(sim)

sim_jit = sim.jit()

# sim.amplitudes.grad(params)
# %%
# give a lambda function which extracts the relevant params
# get = lambda params: jnp.array([op.phi for op in params.ops['phases'].ops])
# get = lambda params: jnp.array([params.ops['phases'].op.phi])
get = lambda pytree: jnp.array(
    [pytree.ops["phases1"].op.phi, pytree.ops["phases2"].op.phi]
)
get(params)
grads = sim.probabilities.grad(params)
get(grads)
# %%
qfim = sim.amplitudes.qfim(get, params)
cfim = sim.probabilities.cfim(get, params)

pprint(qfim)
pprint(cfim)

# %%
det = jnp.linalg.det(qfim)
# inv = jnp.linalg.inv(cfim)
pprint(det)

# %%

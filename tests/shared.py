# %%
import time
import functools
import equinox as eqx
import jax.numpy as jnp
import jax
from rich.pretty import pprint
import seaborn as sns
import polars as pl
import itertools
import matplotlib.pyplot as plt
from squint.circuit import Circuit
from squint.ops.base import SharedGate
from squint.ops.dv import Conditional, DiscreteState, HGate, XGate, Phase, ZGate
from squint.utils import print_nonzero_entries


from loguru import logger

logger.remove()
logger.add(lambda msg: None, level="WARNING")


#%%

gate = SharedGate(op=Phase(wires=(0,), phi=0.3), wires=(1,2))
print(gate._where(gate))
print(gate._get(gate))
gate_ = eqx.tree_at(gate._where, gate, gate._get(gate), is_leaf=lambda leaf: leaf is None)


#%%
op = Phase(wires=(0,), phi=0.0)
[eqx.filter(op, filter_spec=jax.tree.map(eqx.is_inexact_array, op)) for copy in gate.copies]
[eqx.filter(copy, filter_spec=jax.tree.map(eqx.is_inexact_array, copy)) for copy in gate.copies]
# %%  Express the optical circuit.
# ------------------------------------------------------------------
dim = 2

circuit = Circuit()
# circuit.add(FockState(wires=(0,), n=(1,)))
# circuit.add(FockState(wires=(1,), n=(1,)))
# circuit.add(FockState(wires=(2,), n=(1,)))
# phase = Phase(wires=(0,), phi=0.3)
# circuit.add(SharedGate(main=phase, wires=(1, 2)), "phase")

m = 2
for i in range(m):
    circuit.add(DiscreteState(wires=(i,)))
circuit.add(HGate(wires=(0,)))
for i in range(0, m - 1):
    circuit.add(Conditional(gate=XGate, wires=(i, i + 1)))

circuit.add(
    SharedGate(op=Phase(wires=(0,), phi=0.0), wires=tuple(range(1, m))), "phase"
)

for i in range(m):
    circuit.add(HGate(wires=(i,)))

pprint(circuit)
circuit.verify()

params, static = eqx.partition(circuit, eqx.is_inexact_array)
# pprint(params)
# pprint(static)

sim = circuit.compile(params, static, dim=dim)
pr = sim.probability(params)
print_nonzero_entries(pr)

params = eqx.tree_at(
    lambda params: params.ops["phase"].op.phi, params, jnp.array(jnp.pi / 8)
)
pr = sim.probability(params)
grads = sim.grad(params)
dpr = grads.ops["phase"].op.phi
print((dpr**2 / (pr + 1e-14)).sum())

# %%
ns = [
    2,
    3,
    4,
    5
]
dims = [
    2,
    3,
    4,
]

df = []
cfis = jnp.zeros(shape=(len(ns), len(dims)))
# for n, dim in itertools.product(ns, dims):


def generalized_circuit(n):
    circuit = Circuit()

    for i in range(n):
        circuit.add(DiscreteState(wires=(i,)))
    circuit.add(HGate(wires=(0,)))
    for i in range(0, n - 1):
        circuit.add(Conditional(gate=XGate, wires=(i, i + 1)))

    circuit.add(
        SharedGate(op=Phase(wires=(0,), phi=0.0), wires=tuple(range(1, n))), "phase"
    )

    for i in range(n):
        circuit.add(HGate(wires=(i,)))

    circuit.verify()
    return circuit


for a, n in enumerate(ns):
    for b, dim in enumerate(dims):
        print(a, b, n, dim)
        circuit = generalized_circuit(n)

        params, static = eqx.partition(circuit, eqx.is_inexact_array)
        sim = circuit.compile(params, static, dim=dim).jit()
        params = eqx.tree_at(
            lambda params: params.ops["phase"].op.phi, params, jnp.array(jnp.pi / 8)
        )
        _ = sim.probability(params)
        _ = sim.grad(params)

        t0 = time.time()
        pr = sim.probability(params)
        grads = sim.grad(params)
        dpr = grads.ops["phase"].op.phi
        cfi = (dpr**2 / (pr + 1e-14)).sum()
        runtime = time.time() - t0

        cfis = cfis.at[a, b].set(cfi)
        df.append({"cfi": cfi, "n": n, "dim": dim, "runtime": runtime})

df = pl.DataFrame(df)
# %%
pl.Config.set_tbl_rows(100)
df
# %%
fig, ax = plt.subplots()
sns.heatmap(cfis, ax=ax, xticklabels=ns, yticklabels=dims)
ax.set(xlabel="Particle number", ylabel="Dimension")

# %%
fix, axs = plt.subplots(nrows=2)
for dim in dims:
    _df = df.filter(pl.col("dim") == dim)
    axs[0].plot(_df["n"], _df["cfi"])
    axs[1].plot(_df["n"], _df["runtime"], label=f"{dim}")
axs[1].legend()

# %%
print(df)

# %%
dim = 5
circuit = generalized_circuit(3)
sim = circuit.compile(params, static, dim=dim)  # .jit()
params, static = eqx.partition(circuit, eqx.is_inexact_array)
params = eqx.tree_at(
    lambda params: params.ops["phase"].op.phi, params, jnp.array(jnp.pi / 3)
)
pr = sim.probability(params)
grads = sim.grad(params)
dpr = grads.ops["phase"].op.phi
cfi = (dpr**2 / (pr + 1e-14)).sum()
print(cfi)


@jax.jit
def sweep(phi: jnp.array, params):
    params = eqx.tree_at(
        lambda params: params.ops["phase"].op.phi, params, jnp.array(phi)
    )
    pr = sim.probability(params)
    grads = sim.grad(params)
    dpr = grads.ops["phase"].op.phi
    cfi = (dpr**2 / (pr + 1e-14)).sum()
    return cfi


phis = jnp.linspace(0, jnp.pi, 100)
cfi = sweep(phis[0], params)
print(cfi)

cfis = jnp.array([sweep(phi, params) for phi in phis])


fig, ax = plt.subplots()
ax.plot(phis / jnp.pi, cfis)

# %%
dim = 2
h = jnp.diag(jnp.arange(dim) * 2)
eigenvalues, eigenvectors = jnp.linalg.eig(u)
qfi_est = jnp.real((jnp.max(eigenvalues) - jnp.min(eigenvalues))**2)
print(qfi_est)

# %%



%%

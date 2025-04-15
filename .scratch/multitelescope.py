#%%
#%%import functools
import itertools
import functools 
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import polars as pl
import tqdm
from rich.pretty import pprint
from beartype import beartype
import einops 
import seaborn as sns

from squint.circuit import Circuit
# from squint.ops.dv import DiscreteState, Phase, ZGate, HGate, Conditional, AbstractGate, RX, RY
from squint.ops.fock import FockState, BeamSplitter, Phase
from squint.utils import print_nonzero_entries

jax.config.update("jax_enable_x64", True)

#%%

def weak_thermal_state(phi, g, epsilon):
    dim = 2
    v = g * jnp.exp(-1j * phi)
    state = jnp.zeros(shape=(dim, dim, dim, dim), dtype=jnp.complex128)
    state = state.at[0, 0, 0, 0].set(1-epsilon)
    state = state.at[1, 0, 1, 0].set((epsilon / 2))
    state = state.at[0, 1, 0, 1].set((epsilon / 2))
    state = state.at[1, 0, 0, 1].set(jnp.conjugate(v) * (epsilon / 2))
    state = state.at[0, 1, 1, 0].set(v * (epsilon / 2))
    return state

phi = jnp.array(0.1)
g = jnp.array(1.0)
epsilon = jnp.array(0.1)
rho = weak_thermal_state(phi, g, epsilon)
print_nonzero_entries(rho)

state_grad = jax.jacrev(weak_thermal_state, argnums=(0, 1), holomorphic=True)
drho_phi, drho_g = state_grad(phi.astype(jnp.complex128), g.astype(jnp.complex128), epsilon.astype(jnp.complex128))

print_nonzero_entries(drho_phi)
print_nonzero_entries(drho_g)

def qfim(phi, g, epsilon):
    rho = weak_thermal_state(phi, g, epsilon)
    drho_phi, drho_g = state_grad(phi.astype(jnp.complex128), g.astype(jnp.complex128), epsilon.astype(jnp.complex128))
    return 
    
#%%
n_phases = 1
n_qubit = 2
n_ancilla = 2

wires_ancilla = tuple(range(n_qubit, n_qubit + n_ancilla))
wires_qubit = tuple(range(0, n_qubit))

circuit = Circuit()
# for i in range(n_qubit):
circuit.add(
    DiscreteState(wires=wires_qubit, n=[(1.0, (0, 1)), (1.0, (1, 0))])
)
circuit.add(Phase(wires=(0,), phi=0.1), "phase")  # photon

for wire in wires_qubit:
    circuit.add(HGate(wires=(wire,)))
    
pprint(circuit)

params, static = eqx.partition(circuit, eqx.is_inexact_array)
pprint(params)

get = lambda pytree: jnp.array([pytree.ops["phase"].phi])
sim = circuit.compile(params, static, dim=2, optimize="greedy")
print_nonzero_entries(sim.amplitudes.forward(params))

#%%
n_phases = 2

circuit = Circuit()
circuit.add(
    DiscreteState(
        wires=tuple(range(0, n_phases + 1)),
        n=[(1.0, tuple(1 if i == j else 0 for i in range(n_phases + 1))) for j in range(n_phases + 1)]
    )
)

for i in range(n_phases):
    circuit.add(Phase(wires=(i,), phi=0.0), f"phase{i}") 

# for wire in wires_qubit:
    # circuit.add(HGate(wires=(wire,)))
    
pprint(circuit)

params, static = eqx.partition(circuit, eqx.is_inexact_array)
pprint(params)

get = lambda pytree: jnp.array([pytree.ops[f"phase{i}"].phi for i in range(n_phases)])
sim = circuit.compile(params, static, dim=2, optimize="greedy")
print_nonzero_entries(sim.amplitudes.forward(params))

qfim = sim.amplitudes.qfim(get, params)
pprint(qfim)

#%%
@partial(jnp.vectorize, signature='(), (), (k),(k)->(k)')
def qfim(get, params, *phis):
    params = eqx.tree_at(lambda pytree: pytree.ops["phase"].phi, params, phi)
    return sim.prob.cfim(get, params)
   

#%%
phis = jnp.stack(
    jnp.meshgrid(*[jnp.linspace(-jnp.pi, jnp.pi, 100) for _ in range(n_phases)], indexing="ij"),
    axis=0
)

def cfim_sweep_phi(phi, params, get):
    params = eqx.tree_at(lambda pytree: pytree.ops["phase"].phi, params, phi)
    return sim.prob.cfim(get, params)
   
def prob_sweep_phi(phi, params):
    params = eqx.tree_at(lambda pytree: pytree.ops["phase"].phi, params, phi)
    return sim.prob.forward(params)   

cfims = jax.lax.map(functools.partial(cfim_sweep_phi, params=params, get=get), phis)
probs = jax.lax.map(functools.partial(prob_sweep_phi, params=params), phis)
print(probs.shape)
probs = einops.reduce(probs, "A a b c d -> A a c", "mean")

#%%
n_phases = 2
wires_star = tuple(i for i in range(n_phases+1))
wires_lab = tuple(i for i in range(n_phases+1, 2*(n_phases+1)))

circuit = Circuit()
circuit.add(
    FockState(
        wires=wires_star,
        n=[(1.0, tuple(1 if i == j else 0 for i in wires_star)) for j in wires_star]
    )
)
circuit.add(
    FockState(
        wires=wires_lab,
        n=[(1.0, tuple(1 if i == j else 0 for i in wires_lab)) for j in wires_lab]
    )
)
for i in (1, n_phases):
    print(i, i)
    circuit.add(Phase(wires=(i,), phi=0.1), f"phase{i}") 

for wire_star, wire_lab in zip(wires_star, wires_lab):
    circuit.add(BeamSplitter(wires=(wire_star, wire_lab)))

pprint(circuit)

#%%
params, static = eqx.partition(circuit, eqx.is_inexact_array)
pprint(params)

get = lambda pytree: jnp.array([pytree.ops[f"phase{i}"].phi for i in range(1, n_phases+1)])
sim = circuit.compile(params, static, dim=2, optimize="greedy")
print_nonzero_entries(sim.amplitudes.forward(params))

qfim = sim.amplitudes.qfim(get, params)
pprint(qfim)

qfim = sim.prob.cfim(get, params)
pprint(qfim)

# %%

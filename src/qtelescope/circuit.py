# %%
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import paramax
from rich.pretty import pprint

from qtelescope.ops import BeamSplitter, Circuit, FockState, Phase, S2, create, destroy
from qtelescope.utils import partition_op, print_nonzero_entries

import timeit

# %%  Express the optical circuit. 
# ------------------------------------------------------------------
cutoff = 5
circuit = Circuit()

m = 2
# for i in range(m):
circuit.add(FockState(wires=(0, 1), n=[(1.0, (1,1))]))
circuit.add(FockState(wires=(0, 1), n=[(1/jnp.sqrt(2).item(), (3,0)), (1/jnp.sqrt(2).item(), (0,3))]))
# circuit.add(FockState(wires=(0,), n=(2,)))
# circuit.add(FockState(wires=(1,), n=(2,)))
# circuit.add(BeamSplitter(wires=(0, 1), r=jnp.pi/4))
circuit.add(Phase(wires=(0,), phi=jnp.pi/4), 'phase')
circuit.add(BeamSplitter(wires=(0, 1), r=jnp.pi/4))

# for i in range(m - 1):
    # circuit.add(BeamSplitter(wires=(i, i + 1), r=jnp.pi/4))
# circuit.add(Phase(wires=(0,), phi=0.4), "phase")

pprint(circuit)


##%% Find optimal contraction path
# ------------------------------------------------------------------
subscripts = circuit.subscripts
path, info = jnp.einsum_path(
    subscripts, *[op(cut=cutoff) for op in circuit.ops.values()], optimize='greedy'
)
print(info)

##%% Define the forward simulation
# ------------------------------------------------------------------
def _tensor_func(circ, subscripts: str, optimize: tuple):
    return jnp.einsum(
        subscripts, *[op(cut=cutoff) for op in circ.ops.values()], optimize=optimize
    )
    

tensor_func = jax.jit(
    functools.partial(_tensor_func, subscripts=subscripts, optimize=path)
)

##%% Partition into the trainable parameters and static
# ------------------------------------------------------------------
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)
print(params)


##%% Define the forward simulation
# ------------------------------------------------------------------
def forward(params, static):
    circuit = paramax.unwrap(eqx.combine(params, static))
    return tensor_func(circuit)

def prob(params, static):
    state = forward(params, static)
    return jnp.abs(state)**2


forward_jit = jax.jit(functools.partial(forward, static=static))
prob_jit = jax.jit(functools.partial(prob, static=static))

# for phi in jnp.linspace(0.0, jnp.pi, 25):
#     params = eqx.tree_at(
#         lambda params: params.ops['phase'].phi, 
#         params, 
#         phi
#     )

tensor = forward_jit(params)
# print(f"\nNew phi: {phi}")

pr = prob_jit(params)
print_nonzero_entries(pr)
# print(tensor)

##%%
number = 10
times = timeit.Timer(functools.partial(forward_jit, params)).repeat(repeat=3, number=number) 
# print([t / number for t in times])
print("State time:", min(times), max(times))

##%% Differentiate with respect to parameters of interest
name = "phase"
params, static = partition_op(circuit, name)
_grad = jax.jacrev(functools.partial(prob, static=static))
_hess = jax.jacfwd(_grad)

grad_jit = jax.jit(_grad)
hess_jit = jax.jit(_hess)

##%%
grad = grad_jit(params)
hess = hess_jit(params)
# cfim = (hess.ops[name].phi.ops[name].phi / (pr + 1e-12)).sum()
cfim = (grad.ops[name].phi**2 / (pr + 1e-12)).sum()
print("CFIM:", cfim)

##%%
number = 25
times = timeit.Timer(functools.partial(grad_jit, params)).repeat(repeat=3, number=number) 
print("Grad time:", min(times), max(times))
times = timeit.Timer(functools.partial(hess_jit, params)).repeat(repeat=3, number=number) 
print("Hess time:", min(times), max(times))

#%%
# todo: write basic gates, test simple experiments
# todo: more verification of the circuit
# todo: classical Fisher Information 

# %%

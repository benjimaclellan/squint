#%%
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from rich.pretty import pprint

from squint.circuit import Circuit, compile_experimental
from squint.ops.base import SharedGate
from squint.ops.dv import Conditional, DiscreteVariableState, HGate, RZGate, XGate

#%%

n = 1  # number of qubits
circuit = Circuit(backend="mixed")
for i in range(n):
    circuit.add(DiscreteVariableState(wires=(i,), n=(0,)))

circuit.add(HGate(wires=(0,)))
for i in range(n - 1):
    circuit.add(Conditional(gate=XGate, wires=(i, i + 1)))

circuit.add(
    SharedGate(op=RZGate(wires=(0,), phi=0.0 * jnp.pi), wires=tuple(range(1, n))),
    "phase",
)

for i in range(n):
    circuit.add(HGate(wires=(i,)))

pprint(circuit)

#%%
params, static = eqx.partition(circuit, eqx.is_inexact_array)
sim = compile_experimental(static, 2, params, optimize="greedy")

#%%
rho = sim.amplitudes.forward(params).reshape([2**n, 2**n])
drho = (sim.amplitudes.grad(params).ops['phase'].op.phi).reshape([2**n, 2**n])

# %%
tvec = jnp.array()



#%%
n = 1
phi = jnp.array(0.001)

def amplitudes(phi):
    ket = (jnp.zeros(n * (2,)).at[n*(0, )].set(1.0) + jnp.exp(1j * phi) * jnp.zeros(n * (2,)).at[n*(1, )].set(1.0))/jnp.sqrt(2.0)
    return jnp.einsum('a, b -> ab', ket, ket.conj())

rho = amplitudes(phi)
drho = jax.jacfwd(amplitudes, holomorphic=True)(phi.astype(jnp.complex128))

#%%
import equinox as eqx
import jax
import einops
from jaxtyping import ArrayLike, PRNGKeyArray
from beartype import beartype
from squint.ops.base import WiresTypes, AbstractGate
from typing import Optional
import time
import jax.random as jr
import jax.scipy as jsp
from opt_einsum.parser import get_symbol
import optax
import tqdm


class CholeskyDecompositionGate(AbstractGate):
    decomp: ArrayLike  # lower triangular matrix for Cholesky decomposition of hermitian matrix
    scale: ArrayLike
    
    _dim: int
    # _subscripts: str
    _hermitian: bool = True  # whether the decomposition is for a hermitian matrix
    
    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes, ...],
        dim: int,
        key: Optional[PRNGKeyArray] = None,
    ):
        super().__init__(wires=wires)
        if key is None:
            key = jr.PRNGKey(time.time_ns())
        # self.decomp = jnp.ones(shape=(dim ** len(wires), dim ** len(wires)), dtype=jnp.complex_)  # todo
        self.decomp = jr.normal(
            key=key, shape=(dim ** len(wires), dim ** len(wires))
        )  # todo
        self.scale = jnp.array(1.0)
        self._dim = dim
        return

    def __call__(self, dim: int):
        _s_matrix = (
            f"({' '.join([get_symbol(2 * k) for k in range(len(self.wires))])}) "
            f"({' '.join([get_symbol(2 * k + 1) for k in range(len(self.wires))])})"
        )

        _s_tensor = (
            ' '.join([get_symbol(2 * k) for k in range(len(self.wires))]) 
            + ' '
            + ' '.join([get_symbol(2 * k + 1) for k in range(len(self.wires))])
        )
        
        dims = {get_symbol(k): dim for k in range(2 * len(self.wires))}
        
        # tril = jnp.tril(self.decomp)
        tmp = jnp.tril(self.decomp) + 1j * jnp.triu(self.decomp, 1).T
        herm = self.scale * tmp.conj().T @ tmp 
        # u = jsp.linalg.expm(1j * herm).reshape((2 * len(self.wires)) * (dim,))
        
        if self._hermitian:
            tensor = einops.rearrange(
                herm, f"{_s_matrix} -> {_s_tensor}", **dims
            )
        else:
            tensor = einops.rearrange(
                jax.scipy.linalg.expm(1j * herm), f"{_s_matrix} -> {_s_tensor}", **dims
            )
        return tensor
    
op = CholeskyDecompositionGate(wires=(0,), dim=2, key=jr.PRNGKey(1234))

h = op(dim=2)

ops = [
    CholeskyDecompositionGate(wires=(wire,), dim=2, key=jr.PRNGKey(wire + 6))
    for wire in range(n)
]

hermitian_matrices = [op(dim=2) for op in ops]
print([op.decomp for op in ops]) 
print(hermitian_matrices)
params, static = eqx.partition(ops, eqx.is_inexact_array)

def symmetric_logarithmic_derivative(params):
    ops = eqx.combine(static, params)
    hermitian_matrices = [op(dim=2) for op in ops]
    return hermitian_matrices


def distance(params):
    sld = symmetric_logarithmic_derivative(params)
    
    right = 0.5 * (
        jnp.einsum('ab, bc -> ac', *sld, rho) 
        + jnp.einsum('ab, bc -> ac', rho, *sld) 
    )
    return jnp.abs(right - drho).sum()
    

distance(params)

#%%

lr = 1e-3
optimizer = optax.chain(optax.adam(lr), optax.scale(1.0))
opt_state = optimizer.init(params)

def loss(params_est, params_opt):
    return sim.probabilities.cfim(params_est, params_opt).squeeze()

@jax.jit
def step(opt_state, params):
    val, grad = jax.value_and_grad(distance)(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, val

#%%
distances = []
pbar = tqdm.tqdm(range(5000), desc="Optimization Progress")
for i in pbar:
    params, opt_state, val = step(opt_state, params)
    distances.append(val)
    pbar.set_postfix_str(f"Distance: {val:.4f}")
    
#%%
print(val)
sld = symmetric_logarithmic_derivative(params)
print(sld)
print(drho)

right = 0.5 * (
    jnp.einsum('ab, bc -> ac', *sld, rho) 
    + jnp.einsum('ab, bc -> ac', rho, *sld) 
)
print(right)

#%%
jnp.linalg.eig(sld[0])

#%%
# jnp.allclose(h.reshape([2**n, 2**n]), h.reshape([2**n, 2**n]).conj().T)  # check if hermitian

# circuit_cholesky = Circuit(backend="mixed")
# for wire in range(n):
#     circuit_cholesky.add(CholeskyDecompositionGate(wires=(wire,), dim=2, key=jr.PRNGKey(wire)))

# sim = circ

# %%

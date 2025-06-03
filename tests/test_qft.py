#%%
import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jr
from beartype import beartype
import itertools
import functools
import einops
from loguru import logger
from opt_einsum import get_symbol

from squint.circuit import Circuit, compile_experimental
from squint.ops.fock import FockState, BeamSplitter
from squint.ops.base import AbstractGate, create, destroy, eye
from squint.utils import print_nonzero_entries
from rich.pretty import pprint

#%%

class QFT(AbstractGate):
    # coeff: float

    @beartype
    def __init__(
        self, 
        wires: tuple[int, ...], 
        # coeff: float = 0.3
    ):
        super().__init__(wires=wires)
        # self.coeff = jnp.array(coeff)
        return

    def __call__(self, dim: int):
        wires = self.wires
        
        # all permutations of create and destroy operators for an n-wire system, 
        # with identity operators filling the rest
        perms = list(
            itertools.permutations(
                [create(dim), destroy(dim)] 
                + [eye(dim) for _ in range(len(wires) - 2)]
            )
        )
        
        numbers = [tuple(create(dim) @ destroy(dim) if wire_j == wire_i else eye(dim) for wire_j in wires) for wire_i in wires]
        
        # coeff = self.coeff

        
        terms = jnp.pi / 4 * (
            sum([functools.reduce(jnp.kron, perm) for perm in perms])
            - sum([functools.reduce(jnp.kron, number) for number in numbers])
        )
        # u = jax.scipy.linalg.expm(1j * jnp.pi / len(wires) / 2 * terms)
        # u = jax.scipy.linalg.expm(1j * coeff * terms)

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

        u = einops.rearrange(
            jax.scipy.linalg.expm(-1j * terms), f"{_s_matrix} -> {_s_tensor}", **dims
        )

        # subscript = f"{' '.join(characters[: 2 * len(wires)])} -> {' '.join([c for i in range(len(wires)) for c in (characters[i], characters[i + len(wires)])])}"
        # logger.info(f"Subscript for QFT {subscript}")
        # logger.info(f"Number of terms {len(perms)}")
        # return einops.rearrange(u.reshape(2 * len(wires) * (dim,)), "a b c d e f -> a d b e c f")
        return u

dim = 3
wires = (0, 1)
op = QFT(wires=wires)

op(dim)

bs = BeamSplitter(wires=wires, r=jnp.pi/4)
bs(dim)
print_nonzero_entries(bs(dim))
#%%

circuit = Circuit(backend='pure')
circuit.add(FockState(wires=(0, 1), n=(1, 0)))

circuit.add(QFT(wires=(0, 1)))

pprint(circuit)

params, static = eqx.partition(circuit, eqx.is_inexact_array)
sim = compile_experimental(
    static, dim, params, **{"optimize": "greedy", "argnum": 0}
)
amps = sim.amplitudes.forward(params)
print_nonzero_entries(amps)

# %%

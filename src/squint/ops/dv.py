from typing import Sequence, Type, Union, Optional
import time 

import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import paramax
import einops
from beartype import beartype
from beartype.door import is_bearable
from jaxtyping import ArrayLike, PRNGKeyArray

from squint.ops.base import AbstractGate, AbstractState, bases, characters

__all__ = [
    "DiscreteState",
    "XGate",
    "ZGate",
    "HGate",
    "Phase",
    "Conditional",
    "dv_subtypes",
]


class DiscreteState(AbstractState):
    n: Sequence[
        tuple[complex, Sequence[int]]
    ]  # todo: add superposition as n, using second typehint

    @beartype
    def __init__(
        self,
        wires: Sequence[int],
        n: Sequence[int] | Sequence[tuple[complex | float, Sequence[int]]] = None,
    ):
        super().__init__(wires=wires)
        if n is None:
            n = [(1.0, (0,) * len(wires))]  # initialize to |0, 0, ...> state
        if is_bearable(n, Sequence[int]):
            n = [(1.0, n)]
        self.n = paramax.non_trainable(n)
        return

    def __call__(self, dim: int):
        return sum(
            [
                jnp.zeros(shape=(dim,) * len(self.wires)).at[*term[1]].set(term[0])
                for term in self.n
            ]
        )


class XGate(AbstractGate):
    @beartype
    def __init__(
        self,
        wires: tuple[int] = (0,),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        return jnp.roll(jnp.eye(dim, k=0), shift=1, axis=0)


class ZGate(AbstractGate):
    @beartype
    def __init__(
        self,
        wires: tuple[int] = (0,),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        return jnp.diag(jnp.exp(1j * 2 * jnp.pi * jnp.arange(dim) / dim))


class HGate(AbstractGate):
    @beartype
    def __init__(
        self,
        wires: tuple[int] = (0,),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        return jnp.exp(
            1j
            * 2
            * jnp.pi
            / dim
            * jnp.einsum("a,b->ab", jnp.arange(dim), jnp.arange(dim))
        ) / jnp.sqrt(dim)


class Conditional(AbstractGate):
    gate: Union[XGate, ZGate]

    @beartype
    def __init__(
        self,
        gate: Union[Type[XGate], Type[ZGate]],
        wires: tuple[int, int] = (0, 1),
    ):
        super().__init__(wires=wires)
        self.gate = gate(wires=(wires[1],))
        return

    def __call__(self, dim: int):
        # u = sum(
        #     [
        #         jnp.einsum(
        #             "ab,cd -> abcd",
        #             jnp.zeros(shape=(dim, dim)).at[i, i].set(1.0),
        #             eye(dim=dim),
        #         )
        #         for i in range(dim - 1)
        #     ]
        # ) + jnp.einsum(
        #     "ab,cd -> abcd",
        #     jnp.zeros(shape=(dim, dim)).at[-1, -1].set(1.0),
        #     self.gate(dim=dim),
        # )
        u = sum(
            [
                jnp.einsum(
                    "ab,cd -> abcd",
                    jnp.zeros(shape=(dim, dim)).at[i, i].set(1.0),
                    jnp.linalg.matrix_power(self.gate(dim=dim), i),
                )
                for i in range(dim)
            ]
        )

        return u


class Phase(AbstractGate):
    phi: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[int] = (0,),
        phi: float | int = 0.0,
    ):
        super().__init__(wires=wires)
        self.phi = jnp.array(phi)
        return

    def __call__(self, dim: int):
        return jnp.diag(jnp.exp(1j * bases(dim) * self.phi))


class CholeskyDecompositionGate(AbstractGate):
    decomp: ArrayLike  # lower triangular matrix for Cholesky decomposition of hermitian matrix

    _dim: int
    _subscripts: str
    
    @beartype
    def __init__(
        self,
        wires: tuple[int, ...],
        dim: int,
        key: Optional[PRNGKeyArray] = None 
    ):
        super().__init__(wires=wires)
        if not key:
            key = jr.PRNGKey(time.time_ns())
        # self.decomp = jnp.ones(shape=(dim ** len(wires), dim ** len(wires)), dtype=jnp.complex_)  # todo
        self.decomp = jr.normal(key=key, shape=(dim ** len(wires), dim ** len(wires)), dtype=jnp.complex_)  # todo
        self._dim = dim
        self._subscripts = f"{' '.join(characters[0:2*len(wires)])} -> {' '.join(characters[0:2*len(wires):2])} {' '.join(characters[1:2*len(wires):2])}"
        return

    def __call__(self, dim: int):
        tril = jnp.tril(self.decomp)
        herm = tril @ tril.conj().T
        u = jsp.linalg.expm(1j * herm).reshape((2 * len(self.wires)) * (dim,))
        return einops.rearrange(u, self._subscripts)  # todo: interleave reshaping



class RXXGate(AbstractGate):
    theta: ArrayLike
    
    @beartype
    def __init__(
        self,
        wires: tuple[int, int],
        theta: Union[float, int]
    ):
        super().__init__(wires=wires)
        self.theta = jnp.array(theta)
        return

    def __call__(self, dim: int):
        return jnp.array(
            [
                [jnp.cos(self.theta/2), 0.0, 0.0, -1j * jnp.sin(self.theta/2)],
                [0.0,jnp.cos(self.theta/2), -1j * jnp.sin(self.theta/2), 0.0],
                [0.0, -1j * jnp.sin(self.theta/2), jnp.cos(self.theta/2), 0.0],
                [-1j * jnp.sin(self.theta/2), 0.0, 0.0, jnp.cos(self.theta/2)],
            ]
        ).reshape(4 * (dim,))
        return einops.rearrange(u, "a b c d -> a c b d")




dv_subtypes = {DiscreteState, XGate, ZGate, HGate, Conditional, Phase}

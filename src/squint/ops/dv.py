from typing import Sequence

import jax.numpy as jnp
import paramax
from beartype import beartype
from beartype.door import is_bearable
from jaxtyping import ArrayLike

from squint.ops.base import (
    AbstractGate,
    AbstractState,
    bases,
)


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
            n = [(1.0, (0,) * len(wires))]  # initialize to |1, 1, ...> state
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


class WS(AbstractGate):
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

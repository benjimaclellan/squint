# %%
from typing import Sequence

import einops
import jax
import jax.numpy as jnp
import paramax
from beartype import beartype
from beartype.door import is_bearable
from jaxtyping import ArrayLike

from squint.ops.base import (
    AbstractGate,
    AbstractState,
    Phase,
    create,
    destroy,
)

__all__ = ["FockState", "BeamSplitter", "Phase", "S2", "fock_subtypes"]


class FockState(AbstractState):
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
            n = [(1.0, (1,) * len(wires))]  # initialize to |1, 1, ...> state
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


class S2(AbstractGate):
    g: ArrayLike
    phi: ArrayLike

    @beartype
    def __init__(self, wires: Sequence[int], g, phi):
        super().__init__(wires=wires)
        self.g = jnp.array(g)
        self.phi = jnp.array(phi)
        return


class BeamSplitter(AbstractGate):
    r: ArrayLike
    # phi: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[int, int],
        r: float = jnp.pi / 4,
        # phi: float = 0.0,
    ):
        super().__init__(wires=wires)
        self.r = jnp.array(r)
        # self.phi = jnp.array(phi)
        return

    def __call__(self, dim: int):
        bs_l = jnp.kron(create(dim), destroy(dim))
        bs_r = jnp.kron(destroy(dim), create(dim))
        u = jax.scipy.linalg.expm(1j * self.r * (bs_l + bs_r)).reshape(4 * (dim,))
        return einops.rearrange(u, "a b c d -> a c b d")


fock_subtypes = {FockState, BeamSplitter, Phase}

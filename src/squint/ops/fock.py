# %%
import functools
import itertools
from typing import Sequence

import einops
import jax
import jax.numpy as jnp
import paramax
from beartype import beartype
from beartype.door import is_bearable
from jaxtyping import ArrayLike
from loguru import logger

from squint.ops.base import (
    AbstractGate,
    AbstractState,
    bases,
    characters,
    create,
    destroy,
    eye,
)

__all__ = ["FockState", "BeamSplitter", "Phase", "S2", "QFT", "fock_subtypes"]


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
    r: ArrayLike
    phi: ArrayLike

    @beartype
    def __init__(self, wires: Sequence[int], r, phi):
        super().__init__(wires=wires)
        self.r = jnp.asarray(r)
        self.phi = jnp.asarray(phi)
        return

    def __call__(self, dim: int):
        s2_l = jnp.kron(create(dim), create(dim))
        s2_r = jnp.kron(destroy(dim), destroy(dim))
        u = jax.scipy.linalg.expm(
            1j * jnp.tanh(self.r) * (jnp.conjugate(self.phi) * s2_l - self.phi * s2_r)
        ).reshape(4 * (dim,))
        return einops.rearrange(u, "a b c d -> a c b d")


class BeamSplitter(AbstractGate):
    r: ArrayLike
    # phi: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[int, int],
        # r: float = jnp.pi / 4,
        r: float = 0.24492,
        # phi: float = 0.0,
    ):
        super().__init__(wires=wires)
        self.r = jnp.array(r)
        # self.phi = jnp.array(phi)
        return

    def __call__(self, dim: int):
        bs_l = jnp.kron(create(dim), destroy(dim))
        bs_r = jnp.kron(destroy(dim), create(dim))
        u = jax.scipy.linalg.expm(
            1j * jnp.tanh(self.r) * jnp.pi * (bs_l + bs_r)
        ).reshape(4 * (dim,))
        # u = jax.scipy.linalg.expm(1j * self.r * (bs_l + bs_r)).reshape(4 * (dim,))
        return einops.rearrange(u, "a b c d -> a c b d")


class QFT(AbstractGate):
    coeff: float

    @beartype
    def __init__(self, wires: tuple[int, ...], coeff: float = 0.3):
        super().__init__(wires=wires)
        self.coeff = jnp.array(coeff)
        return

    def __call__(self, dim: int):
        wires = self.wires
        perms = list(
            itertools.permutations(
                [create(dim), destroy(dim)] + [eye(dim) for _ in range(len(wires) - 2)]
            )
        )
        coeff = self.coeff

        terms = sum([functools.reduce(jnp.kron, perm) for perm in perms])
        # u = jax.scipy.linalg.expm(1j * jnp.pi / len(wires) / 2 * terms)
        u = jax.scipy.linalg.expm(1j * coeff * terms)

        subscript = f"{' '.join(characters[: 2 * len(wires)])} -> {' '.join([c for i in range(len(wires)) for c in (characters[i], characters[i + len(wires)])])}"
        logger.info(f"Subscript for QFT {subscript}")
        logger.info(f"Number of terms {len(perms)}")
        # return einops.rearrange(u.reshape(2 * len(wires) * (dim,)), "a b c d e f -> a d b e c f")
        return einops.rearrange(u.reshape(2 * len(wires) * (dim,)), subscript)

        # return u
        # return einops.rearrange(u, subscript)


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


fock_subtypes = {FockState, BeamSplitter, Phase}

# %%
if __name__ == "__main__":
    dim = 3
    wires = (0, 1, 2)
    coeff = jnp.pi / 2
    perms = list(
        itertools.permutations(
            [create(dim), destroy(dim)] + [eye(dim) for _ in range(len(wires) - 2)]
        )
    )
    terms = sum([functools.reduce(jnp.kron, perm) for perm in perms])
    u = jax.scipy.linalg.expm(1j * coeff * terms)
    # ket = functools.reduce(jnp.kron, (jnp.zeros(dim).at[1].set(1.0),) * 3)
    ket = functools.reduce(
        jnp.kron,
        [
            jnp.zeros(dim).at[0].set(1.0),
            jnp.zeros(dim).at[0].set(1.0),
            jnp.zeros(dim).at[1].set(1.0),
        ],
    )

    print(jnp.sum(jnp.abs(u @ ket) ** 2))
    u @ ket
    # %%

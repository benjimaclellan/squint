# %%

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import ArrayLike

from squint.ops.base import (
    AbstractChannel,
    basis_operators,
)


class BitFlipChannel(AbstractChannel):
    p: ArrayLike

    @beartype
    def __init__(self, wires: tuple[int], p: float):
        super().__init__(wires=wires)
        self.p = jnp.array(p)
        # self.p = p  #paramax.non_trainable(p)
        return

    def __call__(self, dim: int):
        assert dim == 2
        return jnp.array(
            [
                jnp.sqrt(1 - self.p) * basis_operators(dim=2)[3],  # identity
                jnp.sqrt(self.p) * basis_operators(dim=2)[2],  # X
            ]
        )


class DepolarizingChannel(AbstractChannel):
    p: ArrayLike

    @beartype
    def __init__(self, wires: tuple[int], p: float):
        super().__init__(wires=wires)
        self.p = jnp.array(p)
        return

    def __call__(self, dim: int):
        assert dim == 2
        return jnp.array(
            [
                jnp.sqrt(1 - self.p) * basis_operators(dim=2)[3],  # identity
                jnp.sqrt(self.p / 3) * basis_operators(dim=2)[0],  # Z
                jnp.sqrt(self.p / 3) * basis_operators(dim=2)[1],  # Y
                jnp.sqrt(self.p / 3) * basis_operators(dim=2)[2],  # X
            ]
        )

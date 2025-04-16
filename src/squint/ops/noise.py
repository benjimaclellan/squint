# %%

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import ArrayLike

from opt_einsum.parser import get_symbol

from squint.ops.base import (
    AbstractKrausChannel,
    AbstractErasureChannel,
    basis_operators,
)


class ErasureChannel(AbstractErasureChannel):
    @beartype
    def __init__(self, wires: tuple[int]):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        subscripts = [get_symbol(2*i) + get_symbol(2*i+1) for i in range(len(self.wires))]
        return jnp.einsum(f"{','.join(subscripts)} -> {''.join(subscripts)}", *(len(self.wires) * [jnp.identity(dim)]))


class BitFlipChannel(AbstractKrausChannel):
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



class PhaseFlipChannel(AbstractKrausChannel):
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
                jnp.sqrt(self.p) * basis_operators(dim=2)[0],  # Z
            ]
        )


class DepolarizingChannel(AbstractKrausChannel):
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

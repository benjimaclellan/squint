# Copyright 2024-2025 Benjamin MacLellan

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import ArrayLike
from opt_einsum.parser import get_symbol

from squint.ops.base import (
    AbstractErasureChannel,
    AbstractKrausChannel,
    WiresTypes,
    basis_operators,
)


class ErasureChannel(AbstractErasureChannel):
    r"""
    Erasure channel/photon loss.
    """

    @beartype
    def __init__(self, wires: tuple[WiresTypes]):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        subscripts = [
            get_symbol(2 * i) + get_symbol(2 * i + 1) for i in range(len(self.wires))
        ]
        return jnp.einsum(
            f"{','.join(subscripts)} -> {''.join(subscripts)}",
            *(len(self.wires) * [jnp.identity(dim)]),
        )


class BitFlipChannel(AbstractKrausChannel):
    r"""
    Qubit bit flip channel.
    """

    p: ArrayLike

    @beartype
    def __init__(self, wires: tuple[WiresTypes], p: float):
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
    r"""
    Qubit phase flip channel.
    """

    p: ArrayLike

    @beartype
    def __init__(self, wires: tuple[WiresTypes], p: float):
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
    r"""
    Qubit depolarizing channel.
    """

    p: ArrayLike

    @beartype
    def __init__(self, wires: tuple[WiresTypes], p: float):
        super().__init__(wires=wires)
        self.p = jnp.array(p)
        return

    def __call__(self, dim: int):
        assert dim == 2
        return jnp.array(
            [
                jnp.sqrt(1 - 3 * self.p / 4) * basis_operators(dim=2)[3],  # identity
                jnp.sqrt(self.p / 4) * basis_operators(dim=2)[0],  # Z
                jnp.sqrt(self.p / 4) * basis_operators(dim=2)[1],  # Y
                jnp.sqrt(self.p / 4) * basis_operators(dim=2)[2],  # X
            ]
        )

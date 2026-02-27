#%%
from squint.ops.base import AbstractMeasurement, Wire
import math
from typing import Union, Callable

import jax.numpy as jnp
import jax.scipy as jsp
import paramax
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Sequence, Type
from jaxtyping import ArrayLike, Float, Inexact, Scalar

from squint.ops.base import (
    AbstractGate,
    AbstractMixedState,
    AbstractPureState,
    Wire,
    bases,
    basis_operators,
)

#%%
class Projector(AbstractMeasurement):
    
    n: Sequence[
        tuple[complex, Sequence[int]]
    ] 
    
    @beartype
    def __init__(
        self,
        wires: Sequence[Wire],
        n: Sequence[int] | Sequence[tuple[complex | float, Sequence[int]]] = None,
    ):
        super().__init__(wires=wires)
        if n is None:
            n = [(1.0, (0,) * len(wires))]  # initialize to |0, 0, ...> state
        elif is_bearable(n, Sequence[int]):
            n = [(1.0, n)]
        elif is_bearable(n, Sequence[tuple[complex | float, Sequence[int]]]):
            norm = jnp.sum(jnp.abs(jnp.array([i[0] for i in n])) ** 2)
            n = [((amp / jnp.sqrt(norm)).item(), basis) for amp, basis in n]
        self.n = paramax.non_trainable(n)
        return

    def __call__(self):
        return sum(
            [
                jnp.zeros(
                    shape=[wire.dim for wire in self.wires]
                )
                .at[*term[1]]
                .set(term[0])
                for term in self.n
            ]
        )
        

class POVM(AbstractMeasurement):
    
    @beartype
    def __init__(
        self,
        wires: Sequence[Wire],
    ):
        super().__init__(wires=wires)
        return

    def __call__(self):
        return sum(
            [
                jnp.zeros(
                    shape=[wire.dim for wire in self.wires]
                )
                .at[*term[1]]
                .set(term[0])
                for term in self.n
            ]
        )
#%%
wire = Wire(dim=2)
p = Projector(wires=(wire,), n=(1,))
p()


#%%

# %%

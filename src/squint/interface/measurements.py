# %%

import jax.numpy as jnp
import paramax
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Sequence
from opt_einsum.parser import get_symbol
import itertools

from squint.interface.base import (
    AbstractMeasurement,
    Wire,
    ClassicalWire
)


# %%
# class AbstractProjectiveMeasurement(AbstractMeasurement):
#     @beartype
#     def __init__(
#         self,
#         wires: Sequence[Wire],
#     ):
#         super().__init__(wires=wires)
        
#%%

class AbstractProjectiveMeasurement(AbstractMeasurement):
    projectors: Sequence[tuple[complex, Sequence[int]]]
    out: Wire
    
    @beartype
    def __init__(
        self,
        wires: Sequence[Wire],
        out: Wire,
        projectors: Sequence[int] | Sequence[tuple[complex | float, Sequence[int]]] = None,
    ):
        super().__init__(wires=wires)
        # if projectors is None:
        projectors = tuple(itertools.product(*[list(range(wire.dim)) for wire in wires]))
        self.out = out
        self.projectors = paramax.non_trainable(projectors)
        return

    def __call__(self):
        ket = ''.join([get_symbol(2 * k) for k, wire in enumerate(self.wires)])
        bra = ''.join([get_symbol(2 * k + 1) for k, wire in enumerate(self.wires)])
        
        lhs = f"{ket},{bra}"
        rhs = f"{ket}{bra}"

        return jnp.stack(
            [
                jnp.einsum(
                    f"{lhs}->{rhs}",
                    *[jnp.zeros(shape=[wire.dim for wire in self.wires]).at[*basis].set(1.0)] * 2
                )
                for basis in self.projectors
            ],
            axis=0
        )
        
        
        
class ComputationalBasisMeasurement(AbstractProjectiveMeasurement):
    pass
    # def __call__(self):
    #     lhs = ",".join([f"{get_symbol(2 * k)}{get_symbol(2 * k + 1)}" for k, wire in enumerate(self.wires)])
    #     rhs = "".join([f"{get_symbol(2 * k)}" for k, wire in enumerate(self.wires)] + [f"{get_symbol(2 * k + 1)}" for k, wire in enumerate(self.wires)])
    #     # print(f"{lhs}->{rhs}")
    #     return jnp.einsum(
    #         f"{lhs}->{rhs}",
    #         *[jnp.eye(wire.dim) for wire in self.wires]
    #     )

#%%
wires = [Wire(dim=3, idx=i) for i in range(2)]
out = ClassicalWire(idx="c")
p = ComputationalBasisMeasurement(wires=wires, out=out)
print(p)
#%%
p().shape

#%%
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
                jnp.zeros(shape=[wire.dim for wire in self.wires])
                .at[*term[1]]
                .set(term[0])
                for term in self.n
            ]
        )


# %%
# wire = Wire(dim=2)
# p = Projector(wires=(wire,), n=(1,))
# p()


# %%

# %%

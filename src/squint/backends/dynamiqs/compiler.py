#%%
from plum import dispatch

from squint.interface.base import AbstractProcess, Wire
from jaxtyping import ArrayLike
from beartype import beartype

import dynamiqs as dq

from squint.backends.base import AbstractBackend, DynamiqsBackend, TensorNetworkBackend


class NumberOperator(AbstractProcess):
    omega: ArrayLike
    
    @beartype
    def __init__(
        self,
        wires: tuple[Wire] = (0,),
        omega: float = 1.0,
    ):
        super().__init__(wires=wires)
        self.omega = omega
        return

    def __call__(self, backend: AbstractBackend):
        return self.lower(backend)

    @dispatch
    def lower(self, backend: DynamiqsBackend):
        print("Dynamiqs")
        return self.omega * dq.create(self.wires[0].dim) @ dq.destroy(self.wires[0].dim)
    
    @dispatch
    def lower(self, backend: TensorNetworkBackend):
        print("TensorNetwork")
        
    
class Foo:
    pass

class TestBackend(Foo, DynamiqsBackend):
    pass    

#%%
wire = Wire(dim=2)
op = NumberOperator(wires=(wire,))
print(op)

#%%
op(TensorNetworkBackend())

#%%
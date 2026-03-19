#%%
from plum import dispatch

from squint.interface.base import AbstractProcess, Wire
from jaxtyping import ArrayLike
from beartype import beartype
from beartype.typing import Type
from rich.pretty import pprint
import dynamiqs as dq
import jax.numpy as jnp

from squint.backends.base import AbstractBackend, DynamiqsBackend, TensorNetworkBackend
from squint.interface.dv import *

#%%
H = dq.sigmax()
f = lambda t: jnp.cos(2.0 * jnp.pi * t)
tsave = jnp.linspace(0, 0.5, 101)

H = dq.modulated(f, dq.sigmax())

res = dq.sepropagator(H, tsave)
res.propagators[-1]

#%%

class FockState(AbstractProcess):
    alpha: ArrayLike
    
    @beartype
    def __init__(
        self,
        wires: tuple[Wire] = (0,),
        alpha: float = 1.0,
    ):
        super().__init__(wires=wires)
        self.alpha = alpha
        return

    @dispatch
    def lower(self, backend: DynamiqsBackend):
        print("Dynamiqs")
        # return dq.coherent(self.wires[0].dim, self.alpha)
        return dq.sepropagator(H, tsave)
    
    @dispatch
    def lower(self, backend: TensorNetworkBackend):
        print("TensorNetwork")
  
  
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
state = FockState(wires=(wire,), alpha=0.1)
op = NumberOperator(wires=(wire,), omega=0.5)
pprint(op)
pprint(state)

backend = TestBackend()

#%%
H = op(backend)
psi0 = state(backend)
tsave = jnp.linspace(0.0, 1.0, 101)

#%%
dq.sesolve(H, psi0, tsave)

#%%
def backend_processes(backend: Type[AbstractBackend]) -> list[type]:
    return [
        cls for cls in AbstractProcess._registry
        if hasattr(cls, 'lower') and any(
            backend in sig.signature.types 
            for sig in cls.lower.methods
        )
    ]

#%%
backend_processes(TensorNetworkBackend)

#%%
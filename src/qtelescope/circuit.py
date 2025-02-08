#%%
import jax
import jax.numpy as jnp
import jax.random as jr
from string import ascii_letters
import copy
import equinox as eqx
import matplotlib.pyplot as plt
import paramax
import functools 
from rich.pretty import pprint

import optax 
from typing import Sequence
from jaxtyping import ArrayLike, PyTree

#%%
class AbstractOp(eqx.Module):
    
    wires: tuple[int, ...]
    params: PyTree
    
    def __init__(
        self,
        wires = (0, 1),
        params = [0.0, 1.0]
    ):
        self.wires = wires
        self.params = params
        return

   
class AbstractState(AbstractOp):
    def __init__(
        self,
        wires = (0, 1),
        params = [0.0, 1.0]
    ):
        super().__init__(wires=wires, params=params)
        return 
    
    def __call__(self, cut: int):
       return jnp.zeros(shape=(cut,) * len(self.wires))
   
    
class AbstractGate(AbstractOp):
    def __init__(
        self,
        wires = (0, 1),
        params = [0.0, 1.0]
    ):
        super().__init__(wires=wires, params=params)
        return 

    def __call__(self, cut: int):
       return jnp.zeros(shape=(cut,) * 2 * len(self.wires))
   
    

class AbstractMeasurement(AbstractOp):
    def __init__(
        self,
        wires = (0, 1),
        params = [0.0, 1.0]
    ):
        super().__init__(wires=wires, params=params)
        return
    
    def __call__(self, cut: int):
       return jnp.zeros(shape=(cut,) * 2 * len(self.wires))
   
   
class FockState(AbstractState):
    _state: ArrayLike
    
    def __init__(
        self,
        wires = (0,),
        n: tuple[int] = (1,),
    ):
        super().__init__(wires=wires, params=None)
        self._state = jnp.zeros(shape=(cut,) * len(self.wires)).at[*list(n)].set(1.0)
        return  

    def __call__(self, cut: int):
       return self._state
   

class S2(AbstractGate):
    def __init__(
        self,
        wires = (0, 1),
        g: float = 0.0,
        phi: float = 0.0,
    ):
        params = {'g': g, "phi": phi}
        super().__init__(wires=wires, params=None)
        return

   
class BeamSplitter(AbstractGate):
    def __init__(
        self,
        wires = (0, 1),
        r: float = 0.0,
        phi: float = 0.0,
    ):
        params = {'r': r, "phi": phi}
        # params = paramax.Parameterize(  )  # make positive, 
        super().__init__(wires=wires, params=params)
        return 


class Phase(AbstractGate):
    def __init__(
        self,
        wires = (0, 1),
        r: float = 0.0,
        phi: float = 0.0,
    ):
        params = {'r': r, "phi": phi}
        # params = paramax.Parameterize(  )  # make positive, 
        super().__init__(wires=wires, params=params)
        return 
    

class Circuit(eqx.Module):
    ops: Sequence[AbstractOp]
    
    def __init__(
        self,
        ops = (),
        cutoff: int = 4  
    ):
        self.ops = ops
      
    @property
    def wires(self):
        return set(sum((op.wires for op in self.ops), ()))
        
#%%  
ops = []
# ops += [
#     FockState(wires=(0,), n=(1,)),
#     FockState(wires=(1,), n=(1,)),
#     FockState(wires=(2,), n=(1,)),
#     FockState(wires=(3,), n=(1,)),
#     SPDC(wires=(0, 1)),
#     SPDC(wires=(0, 2)),
#     SPDC(wires=(2, 1)),
#     BS(wires=(2, 1)),
# ]
n = 3
for i in range(n):
    ops.append(FockState(wires=(i,), n=(1,)))
for i in range(n-1):
    ops.append(BeamSplitter(wires=(i,i+1)))

circ = Circuit(ops=ops)
pprint(circ)
print(circ.wires)



print(ascii_letters)

chars = copy.copy(ascii_letters)
_left_axes = []
_right_axes = []

_wire_chars = {wire: [] for wire in circ.wires}

for op in circ.ops:
    _axis = []
    for wire in op.wires:
        if isinstance(op, AbstractState):
            _left_axis = ''
            _right_axes = chars[0]
            chars = chars[1:]
                    
        elif isinstance(op, AbstractGate):
            _left_axis = _wire_chars[wire][-1]
            _right_axes = chars[0]
            chars = chars[1:]
            
        elif isinstance(op, AbstractMeasurement):
            _left_axis = _wire_chars[wire][-1]
            _right_axis = '' 
            
        else:
            raise TypeError
            
        _axis += [_left_axis, _right_axes]
                    
        _wire_chars[wire].append(_right_axes)
        print(_axis)
        
    _left_axes.append("".join(_axis))

_right_axes = [val[-1] for key, val in _wire_chars.items()]

_left_expr = ','.join(_left_axes)
_right_expr = ''.join(_right_axes)
subscripts = f"{_left_expr}->{_right_expr}"
print(subscripts)

#%%
# todo: jit
cut = 6
path, info = jnp.einsum_path(subscripts, *[op(cut=cut) for op in circ.ops], optimize=True)
print(info)

# %%
def _tensor_func(circ, subscripts: str, optimize: tuple):
    return jnp.einsum(subscripts, *[op(cut=cut) for op in circ.ops], optimize=optimize)
    
tensor_func = jax.jit(functools.partial(_tensor_func, subscripts=subscripts, optimize=path))
print(tensor_func(circ).shape)
print(tensor_func(circ))

# %%

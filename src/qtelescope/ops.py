#%%
import jax
import jax.numpy as jnp
import jax.random as jr
from string import ascii_letters, ascii_lowercase, ascii_uppercase
import copy
import equinox as eqx
import matplotlib.pyplot as plt
import paramax
import functools 
from rich.pretty import pprint
from beartype import beartype
import optax 
from typing import Sequence
from jaxtyping import ArrayLike, PyTree

#%%
class AbstractOp(eqx.Module):
    wires: tuple[int, ...]
    # params: PyTree
    
    def __init__(
        self,
        wires = (0, 1),
        # params = [0.0, 1.0]
    ):
        self.wires = wires
        # self.params = params
        return

   
class AbstractState(AbstractOp):
    def __init__(
        self,
        wires = (0, 1),
        # params = [0.0, 1.0]
    ):
        super().__init__(wires=wires) #, params=params)
        return 
    
    def __call__(self, cut: int):
       return jnp.zeros(shape=(cut,) * len(self.wires))
   
    
class AbstractGate(AbstractOp):
    def __init__(
        self,
        wires = (0, 1),
        # params = [0.0, 1.0]
    ):
        super().__init__(wires=wires) #, params=params)
        return 

    def __call__(self, cut: int):
        left = ','.join([ascii_lowercase[i] + ascii_uppercase[i] for i in range(len(self.wires))])
        right = ''.join([ascii_lowercase[i] + ascii_uppercase[i] for i in range(len(self.wires))])
        subscript = f"{left}->{right}"
        return jnp.einsum(
            subscript, 
            *([jnp.eye(cut),] * len(self.wires)),
        )  # n-axis identity operator
   
    

class AbstractMeasurement(AbstractOp):
    def __init__(
        self,
        wires = (0, 1),
        # params = [0.0, 1.0]
    ):
        super().__init__(wires=wires) #, params=params)
        return
    
    def __call__(self, cut: int):
        left = ','.join([ascii_lowercase[i] + ascii_uppercase[i] for i in range(len(self.wires))])
        right = ''.join([ascii_lowercase[i] + ascii_uppercase[i] for i in range(len(self.wires))])
        subscript = f"{left}->{right}"
        return jnp.einsum(
            subscript, 
            *([jnp.eye(cut),] * len(self.wires)),
        )  # n-axis identity operator
   

#%%
class FockState(AbstractState):
    _op: ArrayLike
    
    @beartype
    def __init__(
        self,
        wires = (0,),
        n: tuple[int] = (1,),
    ):
        super().__init__(wires=wires)
        self._op = paramax.non_trainable(jnp.zeros(shape=(cut,) * len(self.wires)).at[*list(n)].set(1.0))
        return  

    def __call__(self, cut: int):
        return paramax.unwrap(self._op)
   

class S2(AbstractGate):
    g: ArrayLike
    phi: ArrayLike
    
    @beartype
    def __init__(self, wires, g, phi):
        params = {'g': g, "phi": phi}
        super().__init__(wires=wires)#, params=None)
        return

   
class BeamSplitter(AbstractGate):
    r: ArrayLike
    phi: ArrayLike
    
    @beartype
    def __init__(self, wires, r=jnp.array(jnp.pi/4), phi=jnp.array(0.0)):
        # params = {'r': paramax.Parameterize(jnp.abs, jnp.array(r)), "phi": paramax.Parameterize(jnp.abs, jnp.array(phi))}
        super().__init__(wires=wires)#, params=params)
        self.r = r
        self.phi = phi
        return 


class Phase(AbstractGate):
    phi: ArrayLike
    
    @beartype
    def __init__(
        self,
        wires = (0, 1),
        phi: float = 0.0,
    ):
        # params = paramax.Parameterize(  )  # make positive, 
        super().__init__(wires=wires)
        self.phi = phi
        return 


class Circuit(eqx.Module):
    ops: Sequence[AbstractOp]
    
    @beartype
    def __init__(
        self,
        ops = [],
        cutoff: int = 4  
    ):
        self.ops = ops
      
    @property
    def wires(self):
        return set(sum((op.wires for op in self.ops), ()))
    
    def add(self, op: AbstractOp):
        self.ops.append(op)
        
        

    def _contraction(self):
        chars = copy.copy(ascii_letters)
        _left_axes = []
        _right_axes = []
        _wire_chars = {wire: [] for wire in circuit.wires}
        for op in circuit.ops:
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
        return subscripts
    
    return 
# %%
import functools
import string
from string import ascii_lowercase, ascii_uppercase

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import ArrayLike
from typing import Sequence, Union, Callable

# %%
characters = (
    string.ascii_lowercase
    + string.ascii_uppercase
    + "".join(chr(code) for code in range(0x03B1, 0x03C0))  # greek lowercase
    + "".join(chr(code) for code in range(0x0391, 0x03A0))  # greek uppercase
)


# %%
@functools.cache
def create(dim):
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim)), k=-1)


@functools.cache
def destroy(dim):
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim)), k=1)


@functools.cache
def eye(dim):
    return jnp.eye(dim)


@functools.cache
def bases(dim):
    return jnp.arange(dim)


# %%
class AbstractOp(eqx.Module):
    wires: tuple[int, ...]

    def __init__(
        self,
        wires=(0, 1),
    ):
        if not all([wire >= 0 for wire in wires]):
            raise TypeError("All wires must be nonnegative ints.")
        self.wires = wires
        return
    
    def unwrap(self):
        return (self,)


class AbstractState(AbstractOp):
    def __init__(
        self,
        wires=(0, 1),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        return jnp.zeros(shape=(dim,) * len(self.wires))


class AbstractGate(AbstractOp):
    def __init__(
        self,
        wires=(0, 1),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        left = ",".join(
            [ascii_lowercase[i] + ascii_uppercase[i] for i in range(len(self.wires))]
        )
        right = "".join(
            [ascii_lowercase[i] + ascii_uppercase[i] for i in range(len(self.wires))]
        )
        subscript = f"{left}->{right}"
        return jnp.einsum(
            subscript,
            *(
                [
                    jnp.eye(dim),
                ]
                * len(self.wires)
            ),
        )  # n-axis identity operator


class AbstractMeasurement(AbstractOp):
    def __init__(
        self,
        wires=(0, 1),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        left = ",".join(
            [ascii_lowercase[i] + ascii_uppercase[i] for i in range(len(self.wires))]
        )
        right = "".join(
            [ascii_lowercase[i] + ascii_uppercase[i] for i in range(len(self.wires))]
        )
        subscript = f"{left}->{right}"
        return jnp.einsum(
            subscript,
            *(
                [
                    jnp.eye(dim),
                ]
                * len(self.wires)
            ),
        )  # n-axis identity operator


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


class SharedPhase(AbstractGate):
    shared: eqx.nn.Shared
    
    def __init__(self, wires: tuple[int, ...], phi: float | int = 0.0):
        super().__init__(wires=wires)
        
        phases = [Phase(phi=phi) for i in range(len(wires))]
        
        # These two weights will now be tied together.
        where = lambda shared: [phase.phi for phase in shared[1:]]
        get = lambda shared: [shared[0].phi for phase in shared[1:]]
        self.shared = eqx.nn.Shared(phases, where, get)
        return 
    
    def __call__(self, dim: int):
        raise RuntimeError("SharedPhase cannot be called directly, it must be unwrapped first.")
    
    def unwrap(self):
        return (phase for phase in self.shared())
    
    
#%%
class SharedGate(AbstractGate):
    main: AbstractOp
    copies: Sequence[AbstractOp]
    _where: Callable
    _get: Callable
    
    @beartype
    def __init__(self, main: AbstractOp, wires: Union[Sequence[int], Sequence[Sequence[int]]]):
        # todo: handle the wires coming from both main and the shared wires
        copies = [eqx.tree_at(lambda op: op.wires, main, (wire,)) for wire in wires]
        self.copies = copies
        self.main = main
        
        wires = main.wires + wires
        super().__init__(wires=wires)
        
        # set up where/get functions for copying parameters
        self._where = lambda pytree: [copy.phi for copy in pytree.copies]
        self._get = lambda pytree: [pytree.main.phi for phase in pytree.copies]
        return self
    
    def __check_init__(self):
        return object.__setattr__(self, "__dict__", eqx.tree_at(self._where, self, replace_fn=lambda _: None).__dict__)
    
    def unwrap(self):
        _self = eqx.tree_at(self._where, self, self._get(self), is_leaf=lambda leaf: leaf is None)
        return [_self.main] + [op for op in _self.copies]
     
     
wires = (1, 2)
# [eqx.tree_at(lambda op: op.wires, op, (wire,)) for wire in wires]
phase = Phase(wires=(0,), phi=0.1)
op = phase

shared = SharedGate(main=phase, wires=(1, 2))
print(shared)

# params, static = eqx.partition(shared, eqx.is_inexact_array)
# print(params)
# print(static)
# print(eqx.tree_at(shared._where, shared, shared._get(shared), is_leaf=lambda leaf: leaf is None))
print(shared.unwrap())

# %%

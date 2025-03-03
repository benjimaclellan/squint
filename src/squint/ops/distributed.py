import functools
import string
from string import ascii_lowercase, ascii_uppercase
from typing import Callable, Sequence, Union
from jaxtyping import ArrayLike
import paramax
import equinox as eqx
import jax.numpy as jnp
import scipy as sp
from beartype import beartype

from squint.ops.base import AbstractGate

__all__ = [
    "GlobalParameter"
]

class GlobalParameter(AbstractGate):
    ops: Sequence[AbstractGate]
    weights: ArrayLike
    
    @beartype
    def __init__(
        self,
        ops: Sequence[AbstractGate],
        weights: ArrayLike,
    ):
        # if not len(ops) and weights.shape[0] 
        wires = [wire for op in ops for wire in op.wires]
        super().__init__(wires=wires)
        self.ops = ops
        self.weights = paramax.non_trainable(weights)

    def unwrap(self):
        """Unwraps the shared ops for compilation and contractions."""
        return self.ops
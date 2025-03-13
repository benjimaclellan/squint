#%%
import itertools
import functools 

import einops
import equinox as eqx
import jax
import optax
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from rich.pretty import pprint
import polars as pl
from beartype import beartype
from opt_einsum.parser import get_symbol

from squint.ops.base import AbstractChannel, basis_operators, AbstractGate, AbstractState
from squint.ops.dv import DiscreteState, XGate, ZGate
from squint.circuit import Circuit
from squint.utils import print_nonzero_entries



class BitFlipChannel(AbstractChannel):
    p: float 
    
    @beartype
    def __init__(
        self,
        wires: tuple[int],
        p: float
    ):
        super().__init__(wires=wires)
        self.p = p  #paramax.non_trainable(p)
        return

    def __call__(self, dim: int):
        assert dim == 2
        return jnp.array(
            [
                jnp.sqrt(1-self.p) * basis_operators(dim=2)[3],   # identity
                jnp.sqrt(self.p) * basis_operators(dim=2)[2],     # X
            ]
        )
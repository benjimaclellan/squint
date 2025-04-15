# %%
import functools
import itertools
from collections import OrderedDict
from typing import Callable, Literal, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import paramax
from beartype import beartype
from jaxtyping import PyTree
from loguru import logger
from opt_einsum.parser import get_symbol

from squint.circuit import Circuit
from squint.estimator import Estimator


class Protocol(eqx.Module):
    
    circuit: Circuit
    estimator: Estimator
            
    @beartype
    def __init__(
        self,
    ):
        """
        """
        pass
#%%
import itertools

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.fock import BeamSplitter, FockState, Phase
from squint.utils import partition_op

#%%
u = jnp.array([[1, -1], [-1, 1]]) / jnp.sqrt(2)



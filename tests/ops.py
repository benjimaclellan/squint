#%%
import pytest

import itertools

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from rich.pretty import pprint

from squint.circuit import Circuit
from squint.ops.dv import DiscreteState, HGate, Phase, CholeskyDecompositionGate
from squint.utils import partition_op

#%%
circuit = Circuit()
circuit.add(op=DiscreteState(wires=(0,)))
# circuit.add(op=DiscreteState(wires=(1,)))
op = CholeskyDecompositionGate(wires=(0,), dim=2)
op(dim=2)
circuit.add(op)
circuit.add(Phase(wires=(0,), phi=0.1))
params, static = eqx.partition(circuit, eqx.is_inexact_array)

sim = circuit.compile(params, static, dim=2)
pr = sim.prob.forward(params)
print(pr)
print(pr.sum())
# %%
cfim = sim.amplitudes.forward(params)

# %%

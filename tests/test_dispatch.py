#%%
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from rich.pretty import pprint
import timeit

from squint.backends.tensornetwork.compiler import (
    circuit_to_optimized_tensor_network_contraction_path,
    circuit_to_tensors,
)
from oqd_compiler_infrastructure import Post, Pre, ConversionRule, Chain

from squint.interface.base import Block, Circuit, SharedGate, Wire, AbstractProcess
from squint.interface.dv import (
    CXGate,
    DiscreteVariableState,
    HGate,
    RZGate,
)
from squint.interface.fock import BeamSplitter, FockState, Phase
from squint.utils import partition_op
from squint.backends.base import TensorNetworkBackend

# %%
name = 'qubit'
# name = 'gjc'
# name = "ghz"


if name == "qubit":
    wire = Wire(dim=2, idx=0)

    circuit = Circuit()

    #          ____      ___________      ____
    # |0> --- | H | --- | Rz(\phi) | --- | H | ----
    #         ----      -----------      ----

    circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
    circuit.add(HGate(wires=(wire,)))
    circuit.add(RZGate(wires=(wire,), phi=0.5 * jnp.pi), "phase")
    circuit.add(HGate(wires=(wire,)))

    pprint(circuit)


#%%
backend = TensorNetworkBackend()

#%%
circuit.ops[0](backend)


#%%

def ops_for_backend(backend_type: type) -> list[type]:
    return [
        cls for cls in AbstractProcess._registry
        if hasattr(cls, 'lower') and any(backend_type in sig.types for sig in cls.lower.methods)
    ]
    
ops_for_backend(TensorNetworkBackend())
    
#%%
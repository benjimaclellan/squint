#%%
import jax.numpy as jnp
import jax
import equinox as eqx
from rich.pretty import pprint
import itertools
import jax.tree_util as jtu

from oqd_compiler_infrastructure.rule import PrettyPrint, RuleBase, RewriteRule, ConversionRule
from oqd_compiler_infrastructure import Chain, FixedPoint, In, Post, Pre, WalkBase
from squint.ops.base import SharedGate, Wire, Circuit, AbstractProcess, Block
from squint.ops.dv import Conditional, DiscreteVariableState, HGate, RZGate, XGate, CZGate, CXGate
from squint.ops.dv import DiscreteVariableState, HGate, RZGate
from squint.ops.noise import BitFlipChannel

from squint.ops.fock import BeamSplitter, FockState, Phase
from squint.utils import partition_op, print_nonzero_entries

from ordered_set import OrderedSet

from opt_einsum.parser import get_symbol

from squint.compiler.tensor_network import MapTensorIndicesMixed, MapTensorIndicesPure, PostSquintWalk, flatten_processes, flatten_subscripts

 #%%
# name = 'qubit'
# name = 'gjc'
name = 'ghz'


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
    
if name == "ghz":
    n = 3  # number of qubits
    wires = [Wire(dim=2, idx=i) for i in range(n)]

    circuit = Circuit()
    block = Block()
    
    
    for w in wires:
        block.add(DiscreteVariableState(wires=(w,), n=(0,)))

    circuit.add(block)
    
    circuit.add(HGate(wires=(wires[0],)))
    for i in range(n - 1):
        circuit.add(CXGate(wires=(wires[i], wires[i + 1])))

    circuit.add(
        SharedGate(op=RZGate(wires=(wires[0],), phi=0.0 * jnp.pi), wires=tuple(wires[1:])),
        "phase",
    )
    # circuit.add(op=(BitFlipChannel(wires=(wires[0],), p=0.1)), key="channel")

    for w in wires:
        circuit.add(HGate(wires=(w,)))
        
    # circuit.add(RZGate(wires=(wires[0],), phi=0.5 * jnp.pi), "phase")

    pprint(circuit)
    
if name == 'gjc':
    cut = 3  # the photon number truncation for the simulation
    wire0 = Wire(dim=cut, idx=0)
    wire1 = Wire(dim=cut, idx=1)
    wire2 = Wire(dim=cut, idx=2)
    wire3 = Wire(dim=cut, idx=3)

    circuit = Circuit()

    # note: `wires` is a spatial mode in this context (in other contexts this can be a information carrying unit, e.g., a qubit/qudit)
    # we add in the stellar photon, which is in an even superposition of spatial modes 0 and 2 (left and right telescopes)
    circuit.add(
        FockState(
            wires=(wire0, wire2),
            n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
        )
    )
    # the stellar photon accumulates a phase shift prior to collection by the left telescope.
    circuit.add(Phase(wires=(wire0,), phi=0.01), "phase")

    # we add the resources photon, which is in an even superposition of spatial modes 1 and 3
    circuit.add(
        FockState(
            wires=(wire1, wire3),
            n=[(1 / jnp.sqrt(2).item(), (1, 0)), (1 / jnp.sqrt(2).item(), (0, 1))],
        )
    )
    

    # we add the linear optical circuit at each telescope (by default this is a 50-50 beamsplitter)
    circuit.add(BeamSplitter(wires=(wire0, wire1)))
    circuit.add(BeamSplitter(wires=(wire2, wire3)))
    pprint(circuit)


#%%
circuit_subscripts, rhs = PostSquintWalk(MapTensorIndicesPure())(circuit)
circuit_subscripts.ops['phase'].copies[0].subscripts


#%%
processes = flatten_processes(circuit_subscripts)
lhs = flatten_subscripts(circuit_subscripts)

subscripts = f"{",".join(lhs)}->{rhs}"

processes = flatten_processes(circuit)
#%%
tensors = [process() for process in processes]

path, info = jnp.einsum_path(
    subscripts,
    *tensors,
    optimize='greedy',
)

jnp.einsum(subscripts, *tensors, optimize=path,)

#%%
"""
Circuit object, which is an immutable pytree.
(first we can have various verification passes)
We need to calculate the subscripts for each AbstractProcess, and attach it to it.
We then need a canonical flattening order.
We also need to output the righthand string.
"""

#%%
params, static = partition_op(circuit, "phase")

def simulate(params):
    circuit_ = eqx.combine(params, static)  # static in closure
    tensors = [process() for process in flatten_processes(circuit_)]
    return jnp.einsum(subscripts, *tensors, optimize=path,)

simulate(params)
simulate_ = jax.jit(simulate)
simulate_(params)

#%%
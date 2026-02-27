#%%
import jax.numpy as jnp
import jax
import equinox as eqx
from rich.pretty import pprint
import itertools
import jax.tree_util as jtu

from oqd_compiler_infrastructure.rule import PrettyPrint, RuleBase, RewriteRule, ConversionRule
from oqd_compiler_infrastructure import Chain, FixedPoint, In, Post, Pre, WalkBase
from squint.ops.base import SharedGate, Wire, Circuit, AbstractProcess
from squint.ops.dv import Conditional, DiscreteVariableState, HGate, RZGate, XGate, CZGate, CXGate
from squint.ops.dv import DiscreteVariableState, HGate, RZGate
from squint.ops.noise import BitFlipChannel

from squint.ops.fock import BeamSplitter, FockState, Phase
from squint.utils import partition_op, print_nonzero_entries

from ordered_set import OrderedSet

from opt_einsum.parser import get_symbol

from squint.compiler.tensor_network import MapTensorIndicesMixed, MapTensorIndicesPure, PostSquintWalk, flatten, AbstractProcessSubscripts


#%%
name = 'qubit'
# name = 'gjc'
# name = 'ghz'


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
    n = 2  # number of qubits
    wires = [Wire(dim=2, idx=i) for i in range(n)]

    circuit = Circuit()
    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

    circuit.add(HGate(wires=(wires[0],)))
    for i in range(n - 1):
        circuit.add(CXGate(wires=(wires[i], wires[i + 1])))

    # circuit.add(
    #     SharedGate(op=RZGate(wires=(wires[0],), phi=0.0 * jnp.pi), wires=tuple(wires[1:])),
    #     "phase",
    # )
    # circuit.add(op=(BitFlipChannel(wires=(wires[0],), p=0.1)), key="channel")

    for w in wires:
        circuit.add(HGate(wires=(w,)))

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
def flatten(root):
    return jtu.tree_leaves(root, is_leaf=lambda x: isinstance(x, AbstractProcessSubscripts))


circuit_subscripts, subscripts_right = PostSquintWalk(MapTensorIndicesPure())(circuit)
circuit_subscripts_flat = flatten(circuit_subscripts)

processes, subscripts_left = zip(*((leaf.process, leaf.subscripts) for leaf in circuit_subscripts_flat))

tensors = [process() for process in processes]
subscripts = f"{",".join(flatten(subscripts_left))}->{subscripts_right}"

path, info = jnp.einsum_path(
    subscripts,
    *tensors,
    optimize='greedy',
)

jnp.einsum(subscripts, *tensors, optimize=path,)

#%%
params, static = partition_op(circuit, "phase")

def _flatten(root):
    return jtu.tree_leaves(root, is_leaf=lambda x: isinstance(x, AbstractProcess))

@jax.jit
def simulate(params):
    circuit_ = eqx.combine(params, static)
    tensors = [process() for process in _flatten(circuit_)]
    return jnp.einsum(subscripts, *tensors)

simulate(params)

#%%
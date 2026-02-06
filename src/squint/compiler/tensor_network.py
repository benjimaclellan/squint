#%%
import jax.numpy as jnp
import jax
import equinox as eqx
from rich.pretty import pprint
import itertools

from oqd_compiler_infrastructure.rule import PrettyPrint, RuleBase, RewriteRule, ConversionRule
from oqd_compiler_infrastructure import Chain, FixedPoint, In, Post, Pre, WalkBase
from squint.ops.base import SharedGate, Wire, Circuit, AbstractOp
from squint.ops.dv import Conditional, DiscreteVariableState, HGate, RZGate, XGate
from squint.ops.dv import DiscreteVariableState, HGate, RZGate
from squint.ops.noise import BitFlipChannel

from squint.ops.fock import BeamSplitter, FockState, Phase
from squint.utils import partition_op, print_nonzero_entries

from ordered_set import OrderedSet

from opt_einsum.parser import get_symbol


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
    n = 3  # number of qubits
    wires = [Wire(dim=2, idx=i) for i in range(n)]

    circuit = Circuit()
    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

    circuit.add(HGate(wires=(wires[0],)))
    for i in range(n - 1):
        circuit.add(Conditional(gate=XGate, wires=(wires[i], wires[i + 1])))

    # circuit.add(
    #     SharedGate(op=RZGate(wires=(wires[0],), phi=0.0 * jnp.pi), wires=tuple(wires[1:])),
    #     "phase",
    # )
    circuit.add(op=(BitFlipChannel(wires=(wires[0],), p=0.1)), key="channel")

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

class MapTensorIndicesMixed(ConversionRule):
    """
    """
    def __init__(self, ):
        super().__init__()
        self.types = ('ket', 'bra', 'channel')
        self._wires_curr_leg = {
            'ket': {}, 
            'bra': {}, 
            'channel': {}
        }
        
        self._count = {
            'ket': itertools.count(0), 
            'bra': itertools.count(0),
            'channel': itertools.count(0),
        }
        
        self.get_next_character = {
            'ket': self.get_next_character_ket,
            'bra': self.get_next_character_bra,
            'channel': self.get_next_character_channel
        }

        self._subscripts_left = []
        self._subscripts_right = []
        
    def get_next_character_ket(self):
        return get_symbol(2 * next(self._count['ket']))
    
    def get_next_character_bra(self):
        return get_symbol(2 * next(self._count['bra']) + 1)
    
    def get_next_character_channel(self):
            return get_symbol(2 * next(self._count['channel']) + 50000)

    def map_Circuit(self, model, operands):
        # return operands
        subscripts_right = "".join(
            leg for leg in itertools.chain(
                self._wires_curr_leg['ket'].values(), 
                self._wires_curr_leg['bra'].values()
            ) if leg is not None
        )
        # subscripts_right = "".join([leg for leg in self._wires_curr_leg['ket'].values() + self._wires_curr_leg['bra'].values() if leg is not None])        
        return f"{",".join(self._subscripts_left)}->{subscripts_right}"
        # return Circuit(ops=operands['ops'])
    
    def map_AbstractMixedState(self, model, operands):
        legs_out = {'ket': [], 'bra': []}
        for wire in model.wires:
            for t in ('ket', 'bra'):
                leg_out = self.get_next_character[t]()

                legs_out[t].append(leg_out)
                self._wires_curr_leg[t][wire.idx] = leg_out

        subscripts = ''.join(legs_out['ket'] + legs_out['bra'])
        self._subscripts_left.append(subscripts)
        return {"subscripts": subscripts}
        
    def map_AbstractPureState(self, model, operands):
        legs_out = {'ket': [], 'bra': []}
        for wire in model.wires:
            for t in ('ket', 'bra'):
                leg_out = self.get_next_character[t]()

                legs_out[t].append(leg_out)
                self._wires_curr_leg[t][wire.idx] = leg_out

        subscripts = ''.join(legs_out['ket']) + ',' + ''.join(legs_out['bra'])
        self._subscripts_left.append(subscripts)
        return {"subscripts": subscripts}
    
    def map_AbstractGate(self, model, operands):
        legs_in, legs_out = {'ket': [], 'bra': []}, {'ket': [], 'bra': []}
        for wire in model.wires:
            for t in ('ket', 'bra'):
                leg_in = self._wires_curr_leg[t][wire.idx]
                leg_out = self.get_next_character[t]()

                legs_in[t].append(leg_in)
                legs_out[t].append(leg_out)
                
                self._wires_curr_leg[t][wire.idx] = leg_out

        subscripts = ''.join(legs_in['ket'] + legs_out['ket']) + ',' + ''.join(legs_in['bra'] + legs_out['bra'])
        self._subscripts_left.append(subscripts)
        return {"subscripts": subscripts}

    def map_AbstractKrausChannel(self, model, operands):
        legs_in, legs_out = {'ket': [], 'bra': []}, {'ket': [], 'bra': []}
        for wire in model.wires:
            for t in ('ket', 'bra'):
                leg_in = self._wires_curr_leg[t][wire.idx]
                leg_out = self.get_next_character[t]()

                legs_in[t].append(leg_in)
                legs_out[t].append(leg_out)
                
                self._wires_curr_leg[t][wire.idx] = leg_out

        leg_ch = self.get_next_character['channel']()
        
        subscripts = ''.join(legs_in['ket'] + legs_out['ket'] + legs_in['bra'] + legs_out['bra'] + [leg_ch])
        self._subscripts_left.append(subscripts)
        return {"subscripts": subscripts}

    def map_AbstractErasureChannel(self, model, operands):
        legs_in = {'ket': [], 'bra': []}
        for wire in model.wires:
            for t in ('ket', 'bra'):
                leg_in = self._wires_curr_leg[t][wire.idx]
                legs_in[t].append(leg_in)
                
                self._wires_curr_leg[t][wire.idx] = None

        leg_ch = self.get_next_character['channel']()
        
        subscripts = ''.join(legs_in['ket'] + [leg_ch]) + ',' + ''.join(legs_in['bra'] + [leg_ch])
        self._subscripts_left.append(subscripts)
        return {"subscripts": subscripts}


class MapTensorIndicesPure(ConversionRule):
    """
    """
    def __init__(self, ):
        super().__init__()
        self._wires_curr_leg = {}
        self._count = itertools.count(0)

        self._subscripts_left = []
        self._subscripts_right = []
        
    def get_next_character(self):
        return get_symbol(next(self._count))
    
    def map_Circuit(self, model, operands):
        subscripts_right = "".join(self._wires_curr_leg.values())
        return f"{",".join(self._subscripts_left)}->{subscripts_right}"
    
    def map_AbstractState(self, model, operands):
        legs_in, legs_out = [], []
        for wire in model.wires:
            # get new char and set as current index
            leg_out = self.get_next_character()
            self._wires_curr_leg[wire.idx] = leg_out
            legs_out.append(leg_out)
        subscripts = ''.join(legs_in + legs_out)
        self._subscripts_left.append(subscripts)
        return {"subscripts": subscripts}
    
    def map_AbstractGate(self, model, operands):
        legs_in, legs_out = [], []
        for wire in model.wires:
            leg_in = self._wires_curr_leg[wire.idx]
            legs_in.append(leg_in)
            leg_out = self.get_next_character()
            self._wires_curr_leg[wire.idx] = leg_out
            legs_out.append(leg_out)
        subscripts = ''.join(legs_in + legs_out)
        self._subscripts_left.append(subscripts)
        return {
            "subscripts": subscripts
        }
    
class GenerateTensors(ConversionRule):
    """
    """
    def __init__(self, ):
        super().__init__()
    
    def map_Circuit(self, model, operands):
        print(operands)
        return Circuit(ops=operands['ops'])
    
    def map_AbstractGate(self, model, operands):
        return model()
    
    def map_AbstractState(self, model, operands):
        return model()
    
    def map_AbstractChannel(self, model, operands):
        return model()
    
    
class GenerateMixedTensors(ConversionRule):
    """
    """
    def __init__(self, ):
        super().__init__()
    
    def map_Circuit(self, model, operands):
        print(operands)
        return Circuit(
            # ops=operands['ops']
            ops=[leaf for tree in operands['ops'] for leaf in tree] 
        )
    
    def map_AbstractGate(self, model, operands):
        arr = model()
        return [arr, arr]
    
    def map_AbstractPureState(self, model, operands):
        arr = model()
        return [arr, arr]
    
    def map_AbstractMixedState(self, model, operands):
        arr = model()
        return [arr]
    
    def map_AbstractChannel(self, model, operands):
        return [model()]
    
    # def map_SharedGate(self, model, operands):
    #     _self = eqx.tree_at(
    #         model.where, model, model.get(model), is_leaf=lambda leaf: leaf is None
    #     )
    #     print(_self)
    #     return [_self.op] + [op for op in _self.copies]
    # def map_Wire(self, model, operands):
        # return model
        

# class FlattenTreePure(RewriteRule):
    # def map(self, model):
        # for 


class PostSquintWalk(Post):
    def walk_Module(self, model):
        new_fields = {}
        for key in self.controlled_reverse(model.__dict__.keys(), self.reverse):
            new_fields[key] = self(getattr(model, key))

        if isinstance(self.rule, ConversionRule):
            self.rule.operands = new_fields
            new_model = self.rule(model)
        else:
            new_model = model.__class__(**new_fields)
            new_model = self.rule(new_model)

        return new_model

subscripts = PostSquintWalk(MapTensorIndicesMixed())(circuit)
pprint(subscripts)

flat_tree, p = jax.tree.flatten(circuit, is_leaf=lambda obj: isinstance(obj, AbstractOp))
tensors = PostSquintWalk(GenerateMixedTensors())(flat_tree)
tensors = [leaf for tree in tensors for leaf in tree] 

print([tensor.shape for tensor in tensors])
jnp.einsum(subscripts, *tensors)

#%%    
# subscripts = PostSquintWalk(MapTensorIndicesPure())(circuit)
# subscripts = PostSquintWalk(MapTensorIndicesMixed())(circuit)
# params, static = partition_op(circuit, "phase")

# def simulate(params):
#     circuit_ = eqx.combine(params, static)
#     tensor_tree = PostSquintWalk(GenerateTensors())(circuit_)
#     tensors, _ = jax.tree.flatten(tensor_tree)
#     return jnp.einsum(subscripts, *tensors)

# #%%
# # simulate(params)
# jax.jit(simulate)(params)

#%%
circuit_ = eqx.combine(params, static)
subscripts = PostSquintWalk(MapTensorIndices())(circuit_)
tensor_tree = PostSquintWalk(GenerateTensors())(circuit_)
print(tensor_tree)

tensors = PostSquintWalk(FlattenTensors())(tensor_tree)
print(tensors)

#%%
params, static = partition_op(circuit, "phase")



#%%
output = simulate(params)
print(output)
# %%

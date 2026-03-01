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

#%%    

"""
This seems like a good way to do the flattening, now it is in a canonical order
"""

class AbstractProcessSubscripts(eqx.Module):
    process: AbstractProcess
    subscripts: str


# class CircuitFlat(eqx.Module):
    # ops: list[AbstractProcess]

# def flatten(root):
    # return jtu.tree_leaves(root, is_leaf=lambda x: isinstance(x, AbstractProcessSubscripts))



class MapTensorIndicesMixed(ConversionRule):
    """
    Maps a symbolic circuit object to a string of input/output tensor leg indices
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
        
        subscripts = (
            ''.join(legs_in['ket'] + legs_out['ket'] + [leg_ch])
            + ','
            + ''.join(legs_in['bra'] + legs_out['bra'] + [leg_ch])
        )
        # subscripts = ''.join(legs_in['ket'] + legs_out['ket'] + legs_in['bra'] + legs_out['bra'] + [leg_ch])
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

"""
- checking every node means that the design of Conditional and Shared gates, with nested AbstractOps within them do not work
- 

"""
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
    
    # TODO: Wires may not be in a canonical order
    # TODO: Need to accomodate classical probability wires
    
    def map_Circuit(self, model, operands):
        subscripts_right = "".join(self._wires_curr_leg.values())
        
        return (Circuit(**operands), subscripts_right)
        # return f"{",".join(self._subscripts_left)}->{subscripts_right}"
    
    def map_AbstractState(self, model, operands):
        legs_in, legs_out = [], []
        for wire in model.wires:
            # get new char and set as current index
            leg_out = self.get_next_character()
            self._wires_curr_leg[wire.idx] = leg_out
            legs_out.append(leg_out)
        subscripts = ''.join(legs_in + legs_out)
        self._subscripts_left.append(subscripts)
        return AbstractProcessSubscripts(process=model, subscripts=subscripts)
        
        # return {"subscripts": subscripts}
    
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
        return AbstractProcessSubscripts(process=model, subscripts=subscripts)
        
        # return (model, subscripts)
        # return {
            # "subscripts": subscripts
        # }


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




# %%

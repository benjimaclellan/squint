# Copyright 2024-2026 Benjamin MacLellan

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
import itertools
import jax.numpy as jnp
import equinox as eqx
from opt_einsum.parser import get_symbol
from oqd_compiler_infrastructure import Post, Pre, ConversionRule, Chain

from squint.ops.base import Block, Circuit, SharedGate

# %%
class MapTensorIndicesMixed(ConversionRule):
    """
    Maps a symbolic circuit object to a string of input/output tensor leg indices
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.types = ("ket", "bra", "channel")
        self._wires_curr_leg = {"ket": {}, "bra": {}, "channel": {}}

        self._count = {
            "ket": itertools.count(0),
            "bra": itertools.count(0),
            "channel": itertools.count(0),
        }

        self.get_next_character = {
            "ket": self.get_next_character_ket,
            "bra": self.get_next_character_bra,
            "channel": self.get_next_character_channel,
        }

        self._subscripts_left = []
        self._subscripts_right = []

    def get_next_character_ket(self):
        return get_symbol(2 * next(self._count["ket"]))

    def get_next_character_bra(self):
        return get_symbol(2 * next(self._count["bra"]) + 1)

    def get_next_character_channel(self):
        return get_symbol(2 * next(self._count["channel"]) + 50000)

    def map_Circuit(self, model, operands):
        rhs = "".join(
            leg
            for leg in itertools.chain(
                self._wires_curr_leg["ket"].values(),
                self._wires_curr_leg["bra"].values(),
            )
            if leg is not None
        )
        return (Circuit(**operands), rhs)

    def map_AbstractMixedState(self, model, operands):
        legs_out = {"ket": [], "bra": []}
        for wire in model.wires:
            for t in ("ket", "bra"):
                leg_out = self.get_next_character[t]()

                legs_out[t].append(leg_out)
                self._wires_curr_leg[t][wire.idx] = leg_out

        subscripts = "".join(legs_out["ket"] + legs_out["bra"])
        self._subscripts_left.append(subscripts)

        object.__setattr__(model, "subscripts", subscripts)
        return model 

    def map_AbstractPureState(self, model, operands):
        legs_out = {"ket": [], "bra": []}
        for wire in model.wires:
            for t in ("ket", "bra"):
                leg_out = self.get_next_character[t]()

                legs_out[t].append(leg_out)
                self._wires_curr_leg[t][wire.idx] = leg_out

        subscripts = "".join(legs_out["ket"]) + "," + "".join(legs_out["bra"])
        self._subscripts_left.append(subscripts)

        object.__setattr__(model, "subscripts", subscripts)
        return model 

    def map_AbstractGate(self, model, operands):
        legs_in, legs_out = {"ket": [], "bra": []}, {"ket": [], "bra": []}
        for wire in model.wires:
            for t in ("ket", "bra"):
                leg_in = self._wires_curr_leg[t][wire.idx]
                leg_out = self.get_next_character[t]()

                legs_in[t].append(leg_in)
                legs_out[t].append(leg_out)

                self._wires_curr_leg[t][wire.idx] = leg_out

        subscripts = (
            "".join(legs_in["ket"] + legs_out["ket"])
            + ","
            + "".join(legs_in["bra"] + legs_out["bra"])
        )
        self._subscripts_left.append(subscripts)

        object.__setattr__(model, "subscripts", subscripts)
        return model     

    def map_AbstractKrausChannel(self, model, operands):
        legs_in, legs_out = {"ket": [], "bra": []}, {"ket": [], "bra": []}
        for wire in model.wires:
            for t in ("ket", "bra"):
                leg_in = self._wires_curr_leg[t][wire.idx]
                leg_out = self.get_next_character[t]()

                legs_in[t].append(leg_in)
                legs_out[t].append(leg_out)

                self._wires_curr_leg[t][wire.idx] = leg_out

        leg_ch = self.get_next_character["channel"]()

        subscripts = (
            "".join(legs_in["ket"] + legs_out["ket"] + [leg_ch])
            + ","
            + "".join(legs_in["bra"] + legs_out["bra"] + [leg_ch])
        )
        self._subscripts_left.append(subscripts)

        object.__setattr__(model, "subscripts", subscripts)
        return model 

    def map_AbstractErasureChannel(self, model, operands):
        legs_in = {"ket": [], "bra": []}
        for wire in model.wires:
            for t in ("ket", "bra"):
                leg_in = self._wires_curr_leg[t][wire.idx]
                legs_in[t].append(leg_in)

                self._wires_curr_leg[t][wire.idx] = None

        leg_ch = self.get_next_character["channel"]()

        subscripts = (
            "".join(legs_in["ket"] + [leg_ch])
            + ","
            + "".join(legs_in["bra"] + [leg_ch])
        )
        self._subscripts_left.append(subscripts)

        object.__setattr__(model, "subscripts", subscripts)
        return model 



class MapTensorIndicesPure(ConversionRule):
    """ """

    def __init__(
        self,
    ):
        super().__init__()
        self._wires_curr_leg = {}
        self._count = itertools.count(0)

        self._subscripts_left = []
        self._subscripts_right = []

    def get_next_character(self):
        return get_symbol(next(self._count))

    # TODO: Wires may not be in a canonical order - we need to output the wire order that defines the state obj
    # TODO: Need to accomodate classical probability wires

    def map_Circuit(self, model, operands):
        rhs = "".join(
            self._wires_curr_leg.values()
        )  # RHS subscripts for the tensor contraction
        return (Circuit(**operands), rhs)

    def map_SharedGate(self, model, operands):
        """
        SharedGate is a structural container.
        We sequentially apply:
            1. base op
            2. each copy
        """
        new_gate = object.__new__(SharedGate)
        for k, v in operands.items():
            object.__setattr__(new_gate, k, v)
        return new_gate

    def map_AbstractState(self, model, operands):
        legs_in, legs_out = [], []
        for wire in model.wires:
            # get new char and set as current index
            leg_out = self.get_next_character()
            self._wires_curr_leg[wire.idx] = leg_out
            legs_out.append(leg_out)
        subscripts = "".join(legs_in + legs_out)
        self._subscripts_left.append(subscripts)

        object.__setattr__(model, "subscripts", subscripts)
        return model 
    

    def map_AbstractGate(self, model, operands):
        legs_in, legs_out = [], []
        for wire in model.wires:
            leg_in = self._wires_curr_leg[wire.idx]
            legs_in.append(leg_in)
            leg_out = self.get_next_character()
            self._wires_curr_leg[wire.idx] = leg_out
            legs_out.append(leg_out)
        subscripts = "".join(legs_in + legs_out)
        self._subscripts_left.append(subscripts)

        object.__setattr__(model, "subscripts", subscripts)
        return model


class CollectSubscripts(ConversionRule):
    def __init__(self, ):
        super().__init__()
        self.lhs = []
    
    def map_Circuit(self, model, operands):
        return ','.join(self.lhs)
    
    def map_AbstractProcess(self, model, operands):
        self.lhs.append(model.subscripts)
    

class DistributeSharedGates(ConversionRule):
    def map_SharedGate(self, model, operands):
        # Distributes/copies the parameters across the shared gates 
        operand = eqx.tree_at(
            model.where, model, model.get(model), is_leaf=lambda leaf: leaf is None
        )
        return operand

class GeneratePureTensors(ConversionRule):
    """
    """
    def __init__(self, ):
        super().__init__()
        self.tensors = []
    
    def map_Circuit(self, model, operands):
        return self.tensors
        
    def map_Block(self, model, operands):
        return operands
    
    def map_AbstractGate(self, model, operands):
        tensor = model()
        self.tensors += [tensor]
        return [tensor]
    
    def map_AbstractPureState(self, model, operands):
        tensor = model()
        self.tensors += [tensor]
        return [tensor]
    
    
class GenerateMixedTensors(ConversionRule):
    def __init__(self, ):
        super().__init__()
        self.tensors = []
    
    def map_Circuit(self, model, operands):
        return self.tensors
        
    def map_Block(self, model, operands):
        return operands
    
    def map_AbstractGate(self, model, operands):
        tensor = model()
        self.tensors += [tensor, tensor]
        return [tensor, tensor]
    
    def map_AbstractPureState(self, model, operands):
        tensor = model()
        self.tensors += [tensor, tensor]
        return [tensor, tensor]
    
    def map_AbstractMixedState(self, model, operands):
        tensor = model()
        self.tensors.append(tensor)
        return [tensor]
    
    def map_AbstractChannel(self, model, operands):
        tensor = model()
        self.tensors.append(tensor)
        return [tensor]


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
  

class PreSquintWalk(Pre):
    def walk_Module(self, model):
        new_model = self.rule(model)
        
        # Walk children of the NEW node, not the original
        new_fields = {}
        for key in self.controlled_reverse(new_model.__dict__.keys(), self.reverse):
            new_fields[key] = self(getattr(new_model, key))  # <-- new_model, not model
        
        # Reconstruct using bypass to avoid ergonomic constructor issues
        result = object.__new__(new_model.__class__)
        for key, value in new_fields.items():
            object.__setattr__(result, key, value)
        return result
    
    

def circuit_to_tensors(circuit):
    return Chain(
        PreSquintWalk(DistributeSharedGates()),
        PostSquintWalk(GeneratePureTensors())
    )(circuit)
    
def circuit_to_optimized_tensor_network_contraction_path(circuit, optimize: str = "greedy"):
    _circuit_subscripts, rhs = PostSquintWalk(MapTensorIndicesPure())(circuit)
    lhs = PostSquintWalk(CollectSubscripts())(_circuit_subscripts)
    
    subscripts = f"{lhs}->{rhs}"
    
    tensors = circuit_to_tensors(circuit)
    
    path, info = jnp.einsum_path(
        subscripts,
        *tensors,
        optimize=optimize,
    )
    return subscripts, path

#%%

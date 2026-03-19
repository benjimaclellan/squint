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
from oqd_compiler_infrastructure import Post, Pre, ConversionRule, Chain, RewriteRule

from squint.backends.base import AbstractBackend, TensorNetworkBackend
from squint.interface.base import Circuit, SharedGate

# %%


class PureBackend(TensorNetworkBackend):
    pass

class MixedBackend(TensorNetworkBackend):
    pass


class AllowedBackendsAnalysis(ConversionRule, TensorNetworkBackend):
    def __init__(self, ):
        super().__init__()  
        self.backend = PureBackend
        
    def map_Circuit(self, model, operands):
        return self.backend
    
    def map_AbstractChannel(self, model, operands):
        self.backend = MixedBackend
  
    def map_AbstractMixedState(self, model, operands):
        self.backend = MixedBackend
        
    
class ExtractCanonicalWireOrder(ConversionRule, TensorNetworkBackend):
    def __init__(self, ):
        super().__init__()  
        self.wires = set()
        
    def map_Circuit(self, model, operands):
        return tuple(self.wires)
    
    def map_AbstractProcess(self, model, operands):
        for wire in model.wires:
            self.wires.add(wire)



class CollectSubscripts(ConversionRule, TensorNetworkBackend):
    def __init__(self, ):
        super().__init__()
        self.lhs = []
    
    def map_Circuit(self, model, operands):
        return ','.join(self.lhs)
    
    def map_AbstractProcess(self, model, operands):
        self.lhs.append(model.subscripts)
    

class DistributeSharedGates(ConversionRule, TensorNetworkBackend):
    def map_SharedGate(self, model, operands):
        # Distributes/copies the parameters across the shared gates 
        operand = eqx.tree_at(
            model.where, model, model.get(model), is_leaf=lambda leaf: leaf is None
        )
        return operand



class MapTensorIndicesMixed(ConversionRule, TensorNetworkBackend):
    """
    Maps a symbolic circuit object to a string of input/output tensor leg indices
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.types = ("ket", "bra", "channel", "prob")
        self._wires_curr_leg = {"ket": {}, "bra": {}, "channel": {}, "prob": {}}

        self._count = {
            "ket": itertools.count(0),
            "bra": itertools.count(0),
            "channel": itertools.count(0),
            "prob": itertools.count(0),
        }

        self.get_next_character = {
            "ket": self.get_next_character_ket,
            "bra": self.get_next_character_bra,
            "channel": self.get_next_character_channel,
            "prob": self.get_next_character_channel,
        }

        self._subscripts_left = []
        self._subscripts_right = []

    def get_next_character_ket(self):
        return get_symbol(2 * next(self._count["ket"]))

    def get_next_character_bra(self):
        return get_symbol(2 * next(self._count["bra"]) + 1)

    def get_next_character_channel(self):
        return get_symbol(next(self._count["channel"]) + 50000)

    def get_next_character_prob(self):
        return get_symbol(next(self._count["prob"]) + 25000)

    def map_Circuit(self, model, operands):
        # print(self._wires_curr_leg)
        rhs = "".join(
            leg
            for leg in itertools.chain(
                self._wires_curr_leg["ket"].values(),
                self._wires_curr_leg["bra"].values(),
                self._wires_curr_leg["prob"].values(),
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

    def map_AbstractProjectiveMeasurement(self, model, operands):
        legs_in, legs_out = {"ket": [], "bra": []}, {"ket": [], "bra": []}
        for wire in model.wires:
            for t in ("ket", "bra"):
                leg_in = self._wires_curr_leg[t][wire.idx]
                # leg_out = self.get_next_character[t]()

                legs_in[t].append(leg_in)
                # legs_out[t].append(None)

                self._wires_curr_leg[t][wire.idx] = None
        
        
        leg_out_prob = self.get_next_character["prob"]()
        self._wires_curr_leg["prob"][model.out.idx] = leg_out_prob
        
        subscripts = (
            "".join([leg_out_prob] + legs_in["ket"] + legs_in["bra"])
        )
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

        # the leg index that represents the contraction between the Kraus operator tensors
        # canonically, this is the last index - therefore all AbstractKrausOperators should stack along axis=-1 
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



class MapTensorIndicesPure(ConversionRule, TensorNetworkBackend):
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


class GeneratePureTensors(ConversionRule, TensorNetworkBackend):
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
        tensor = model(self)
        self.tensors += [tensor]
        return [tensor]
    
    def map_AbstractPureState(self, model, operands):
        tensor = model(self)
        self.tensors += [tensor]
        return [tensor]
  
  
class GenerateMixedTensors(ConversionRule, TensorNetworkBackend):
    def __init__(self, ):
        super().__init__()
        self.tensors = []
    
    def map_Circuit(self, model, operands):
        return self.tensors
        
    # def map_Block(self, model, operands):
        # return operands
    
    def map_AbstractGate(self, model, operands):
        tensor = model(self)
        out = [tensor, jnp.conj(tensor)]
        self.tensors += out
        return out
    
    def map_AbstractPureState(self, model, operands):
        tensor = model(self)
        out = [tensor, jnp.conj(tensor)]
        self.tensors += out
        return out
    
    def map_AbstractMixedState(self, model, operands):
        tensor = model(self)
        self.tensors.append(tensor)
        return [tensor]
    
    def map_AbstractKrausChannel(self, model, operands):
        tensor = model(self)
        self.tensors += [tensor, jnp.conj(tensor)]
        return [tensor, jnp.conj(tensor)]

    def map_AbstractErasureChannel(self, model, operands):
        tensor = model(self)
        out = [tensor, jnp.conj(tensor)]
        self.tensors += out
        return out
    
    def map_AbstractProjectiveMeasurement(self, model, operands):
        tensor = model(self)
        self.tensors.append(tensor)
        return [tensor]


class PostSquintWalk(Post):
    def walk_Module(self, model):

        new_fields = {}
        for key in self.controlled_reverse(model.__dict__.keys(), self.reverse):
            if key.startswith('__'):
                continue
            new_fields[key] = self(getattr(model, key))

        if isinstance(self.rule, ConversionRule):
            self.rule.operands = new_fields
            new_model = self.rule(model)
            
        else:
            new_model = object.__new__(model.__class__)
            for key, value in new_fields.items():
                object.__setattr__(new_model, key, value)
            new_model = self.rule(new_model)
        
        return new_model
  

class PreSquintWalk(Pre):
    def walk_Module(self, model):
        new_model = self.rule(model)
        
        # Walk children of the NEW node, not the original
        new_fields = {}
        for key in self.controlled_reverse(new_model.__dict__.keys(), self.reverse):
            if key.startswith('__'):
                continue
            new_fields[key] = self(getattr(new_model, key))
            
        # Reconstruct using bypass to avoid ergonomic constructor issues
        result = object.__new__(new_model.__class__)
        for key, value in new_fields.items():
            object.__setattr__(result, key, value)
        return result
    
    

def circuit_to_tensors(
    circuit, # TODO: change to AbstractContainer
    backend: type[AbstractBackend]
):
    if backend == PureBackend:
        chain = Chain(
            PreSquintWalk(DistributeSharedGates()),
            PostSquintWalk(GeneratePureTensors())
        )
    elif backend == MixedBackend:
        chain = Chain(
            PreSquintWalk(DistributeSharedGates()),
            PostSquintWalk(GenerateMixedTensors())
        )
    else:
        raise RuntimeError("No a valid backend")
    return chain(circuit)
    
def circuit_to_allowed_backends(circuit):
    return PostSquintWalk(AllowedBackendsAnalysis())(circuit)

def circuit_to_wire_order(circuit):
    return PostSquintWalk(ExtractCanonicalWireOrder())(circuit)

def circuit_to_subscripts(
    circuit, 
    backend: type[AbstractBackend],
    optimize: str = "greedy"
):
    if backend == PureBackend:
        chain = PostSquintWalk(MapTensorIndicesPure())
    elif backend == MixedBackend:
        chain = PostSquintWalk(MapTensorIndicesMixed())
    else:
        raise RuntimeError("No a valid backend")
    _circuit_subscripts, rhs = chain(circuit)
    
    lhs = PostSquintWalk(CollectSubscripts())(_circuit_subscripts)
    
    subscripts = f"{lhs}->{rhs}"
    return subscripts 

def circuit_to_optimized_tensor_network_contraction_path(
    circuit, 
    backend: type[AbstractBackend],
    optimize: str = "greedy"
):
    
    subscripts = circuit_to_subscripts(circuit, backend=backend)
    
    tensors = circuit_to_tensors(circuit, backend=backend)
    
    path, info = jnp.einsum_path(
        subscripts,
        *tensors,
        optimize=optimize,
    )
    return subscripts, path

#%%

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

from squint.ops.base import (
    AbstractChannel,
    AbstractGate,
    AbstractMixedState,
    AbstractMeasurement,
    AbstractOp,
    AbstractPureState,
)
from squint.ops.fock import fock_subtypes
from squint.simulator import (
    Simulator,
    SimulatorClassicalProbability,
    SimulatorQuantumAmplitude,
    classical_fisher_information_matrix,
    quantum_fisher_information_matrix,
)


class Circuit(eqx.Module):
    """
    Class representing a complex number

    Attributes:
        real (float): real part of the complex number.
        imag (float): imaginary part of the complex number.
    """
    
    dims: tuple[int, ...] = None  # todo: implement dimension per wire
    ops: OrderedDict[Union[str, int], AbstractOp]
    _backend: Literal["unitary", "nonunitary"]
    
    @beartype
    def __init__(
        self,
        backend: Literal["unitary", "nonunitary"] = "unitary"
    ):
        """
        Initializes a quantum circuit with the specified backend type.
        
        Args:
            backend (Literal["unitary", "nonunitary"]): The type of backend to use for the circuit.
            Defaults to "unitary". "unitary" represents a reversible quantum operation,
            while "nonunitary" allows for non-reversible operations.
        """
        self.ops = OrderedDict()
        self._backend = backend

    @property
    def wires(self):
        return set(sum((op.wires for op in self.unwrap()), ()))

    @beartype
    def add(self, op: AbstractOp, key: str = None):  # todo:
        """
        Add an operator to the circuit sequentially.
        """
        if key is None:
            key = len(self.ops)
        self.ops[key] = op
        
        # if isinstance(op, (AbstractChannel, AbstractMixedState)):
        #     self._backend = "nonunitary"

    def unwrap(self):
        return tuple(
            op for op_wrapped in self.ops.values() for op in op_wrapped.unwrap()
        )

    def verify(self):
        circuit_subtypes = set(map(type, self.ops.values()))
        if circuit_subtypes == fock_subtypes:
            logger.debug("Circuit is contains only Fock space components.")

    @property
    def backend(self):
        return self._backend

    @property
    def subscripts(self):
        if self.backend == "nonunitary":
            return subscripts_nonunitary(self)
        elif self.backend == "unitary":
            return subscripts_unitary(self)

    @beartype
    def path(
        self,
        dim: int,
        optimize: str = "greedy",
    ):
        path, info = jnp.einsum_path(
            self.subscripts,
            *self.evaluate(dim=dim),
            optimize=optimize,
        )
        return path, info

    @beartype
    def evaluate(
        self, 
        dim: int, 
    ):
        if self.backend == "unitary":
            return [op(dim=dim) for op in self.unwrap()]
        # TODO: change how ops are unrolled
        elif self.backend == "nonunitary":
            # unconjugated/right + conj/left direction of tensor network
            return [op(dim=dim) for op in self.unwrap()] + [
                op(dim=dim).conj() for op in self.unwrap()
            ]

    @beartype
    def compile(self, params, static, dim: int, optimize: str = "greedy"):
        path, info = self.path(dim=dim, optimize=optimize)
        logger.debug(info)

        def _tensor_func(
            circuit,
            dim: int,
            subscripts: str,
            path: tuple,
            backend: Literal["unitary", "nonunitary"],
        ):
            return jnp.einsum(
                subscripts,
                *jtu.tree_map(
                    lambda x: x.astype(jnp.complex64),
                    circuit.evaluate(dim=dim),
                ),
                optimize=path,
            )
        
        backend = self.backend
            
        _tensor = functools.partial(
            _tensor_func,
            dim=dim,
            subscripts=self.subscripts,
            path=path,
            backend=backend,
        )

        def _forward_state_func(params: PyTree, static: PyTree):
            circuit = paramax.unwrap(eqx.combine(params, static))
            return _tensor(circuit)

        _forward_state = functools.partial(_forward_state_func, static=static)

        if backend == "unitary":

            def _forward_prob(params: PyTree):
                return jnp.abs(_forward_state(params)) ** 2

        elif backend == "nonunitary":

            def _forward_prob(params: PyTree):
                _subscripts_tmp = [get_symbol(i) for i in range(len(self.wires))]
                _subscripts = (
                    "".join(_subscripts_tmp + _subscripts_tmp)
                    + "->"
                    + "".join(_subscripts_tmp)
                )
                return jnp.abs(jnp.einsum(_subscripts, _forward_state(params)))

        _grad_state_nonholomorphic = jax.jacrev(_forward_state, holomorphic=True)

        def _grad_state(params: PyTree):
            params = jtu.tree_map(lambda x: x.astype(jnp.complex64), params)
            return _grad_state_nonholomorphic(params)

        _grad_prob = jax.jacrev(_forward_prob)

        return Simulator(
            amplitudes=SimulatorQuantumAmplitude(
                forward=_forward_state,
                grad=_grad_state,
                qfim=functools.partial(
                    quantum_fisher_information_matrix, _forward_state, _grad_state
                ),
            ),
            prob=SimulatorClassicalProbability(
                forward=_forward_prob,
                grad=_grad_prob,
                cfim=functools.partial(
                    classical_fisher_information_matrix, _forward_prob, _grad_prob
                ),
            ),
            path=path,
            info=info,
        )


# %%
def subscripts_unitary(circuit: Circuit):
    """Subscripts for pure state evolution"""
    _iterator = itertools.count(0)

    _left_axes = []
    _right_axes = []
    _wire_chars = {wire: [] for wire in circuit.wires}
    for op in circuit.unwrap():
        _axis = []
        for wire in op.wires:
            if isinstance(op, AbstractPureState):
                _left_axis = ""
                _right_axes = get_symbol(next(_iterator))

            elif isinstance(op, AbstractGate):
                _left_axis = _wire_chars[wire][-1]
                _right_axes = get_symbol(next(_iterator))

            elif isinstance(op, AbstractMeasurement):
                _left_axis = _wire_chars[wire][-1]
                _right_axis = ""

            else:
                raise TypeError

            _axis += [_left_axis, _right_axes]

            _wire_chars[wire].append(_right_axes)

        _left_axes.append("".join(_axis))

    _right_axes = [val[-1] for key, val in _wire_chars.items()]

    _left_expr = ",".join(_left_axes)
    _right_expr = "".join(_right_axes)
    subscripts = f"{_left_expr}->{_right_expr}"
    return subscripts


# def subscripts_nonunitary(circuit: Circuit):
#     def _subscripts_left_right(
#         circuit: Circuit, get_symbol: Callable, get_symbol_channel: Callable
#     ):
#         _iterator = itertools.count(0)
#         _iterator_channel = itertools.count(0)

#         _in_axes = []
#         _out_axes = []
#         _wire_chars = {wire: [] for wire in circuit.wires}
#         for op in circuit.unwrap():
#             _axis = []
#             for wire in op.wires:
#                 if isinstance(op, AbstractState):
#                     _in_axis = ""
#                     _out_axes = get_symbol(next(_iterator))
                    
#                 elif isinstance(op, AbstractMixedState):
#                     _in_axis = ""
#                     _out_axes = get_symbol(next(_iterator))

#                 elif isinstance(op, AbstractGate):
#                     _in_axis = _wire_chars[wire][-1]
#                     _out_axes = get_symbol(next(_iterator))

#                 elif isinstance(op, AbstractChannel):
#                     _in_axis = _wire_chars[wire][-1]
#                     _out_axes = get_symbol(next(_iterator))

#                 elif isinstance(op, AbstractMeasurement):
#                     _in_axis = _wire_chars[wire][-1]
#                     _right_axis = ""

#                 else:
#                     raise TypeError

#                 _axis += [_in_axis, _out_axes]

#                 _wire_chars[wire].append(_out_axes)

#             # add extra axis for channel
#             if isinstance(op, AbstractChannel):
#                 _axis.insert(0, get_symbol_channel(next(_iterator_channel)))
            
#             # add extra axis for mixed
#             if isinstance(op, AbstractMixedState):
#                 _axis.insert(0, get_symbol_channel(next(_iterator_channel)))
            
            
#             _in_axes.append("".join(_axis))

#         _out_axes = [val[-1] for key, val in _wire_chars.items()]

#         _in_expr = ",".join(_in_axes)
#         _out_expr = "".join(_out_axes)
#         _subscripts = f"{_in_expr}->{_out_expr}"
#         return _in_expr, _out_expr

#     START_CHANNEL = 20000

#     def get_symbol_right(i):
#         # assert i + START_RIGHT < START_LEFT, "Collision of leg symbols"
#         return get_symbol(2 * i)
#         # return get_symbol(i + START_RIGHT)

#     def get_symbol_left(i):
#         # assert i + START_LEFT < START_CHANNEL, "Collision of leg symbols"
#         return get_symbol(2 * i + 1)
#         # return get_symbol(i + START_LEFT)

#     def get_symbol_channel(i):
#         # assert i + START_CHANNEL < START_LEFT, "Collision of leg symbols"
#         return get_symbol(2 * i + START_CHANNEL)

#     _in_expr_ket, _out_expr_ket = _subscripts_left_right(
#         circuit, get_symbol_right, get_symbol_channel
#     )
#     _in_expr_bra, _out_expr_bra = _subscripts_left_right(
#         circuit, get_symbol_left, get_symbol_channel
#     )
#     _subscripts = f"{_in_expr_ket},{_in_expr_bra}->{_out_expr_ket}{_out_expr_bra}"
#     return _subscripts

#%%
if __name__ == "__main__":
    #%%

    from squint.circuit import Circuit, AbstractPureState
    # from squint.ops.dv import DiscreteState, Phase, ZGate, HGate, Conditional, AbstractGate, RX, RY
    from squint.ops.fock import FockState, BeamSplitter, Phase, TwoModeWeakCoherentSource
    from squint.utils import print_nonzero_entries

    #%%
    def subscripts_nonunitary(circuit: Circuit):
        START_CHANNEL = 20000

        def get_symbol_ket(i):
            # assert i + START_RIGHT < START_LEFT, "Collision of leg symbols"
            return get_symbol(2 * i)
            # return get_symbol(i + START_RIGHT)

        def get_symbol_bra(i):
            # assert i + START_LEFT < START_CHANNEL, "Collision of leg symbols"
            return get_symbol(2 * i + 1)
            # return get_symbol(i + START_LEFT)

        def get_symbol_channel(i):
            # assert i + START_CHANNEL < START_LEFT, "Collision of leg symbols"
            return get_symbol(2 * i + START_CHANNEL)

        
        _iterator_ket = itertools.count(0)
        _iterator_bra = itertools.count(0)
        _iterator_channel = itertools.count(0)

        _wire_chars_ket = {wire: [] for wire in circuit.wires}
        _wire_chars_bra = {wire: [] for wire in circuit.wires}
        
        _in_subscripts = []
        for op in circuit.unwrap():
            print(op.__class__)
            # _axis = []
            # _axis_ket = []
            # _axis_bra = []
            # _in_axes = []
            # _out_axes = []
            
            _in_axes_ket = []
            _in_axes_bra = []
            _out_axes_ket = []
            _out_axes_bra = []
            
            for wire in op.wires:
                if isinstance(op, AbstractMixedState):
                    
                    # _in_axis_ket = ""
                    # _in_axis_bra = ""
                    # _out_axis_ket = get_symbol_ket(next(_iterator_ket))
                    # _out_axis_bra = get_symbol_bra(next(_iterator_bra))
                    
                    # _axis_ket += [_in_axis_ket, _out_axis_ket]  # todo: canonical ordering
                    # _axis_bra += [_in_axis_bra, _out_axis_bra]
                    _in_axes_ket.append("")
                    _in_axes_bra.append("")
                    _out_axes_ket.append(get_symbol_ket(next(_iterator_ket)))
                    _out_axes_bra.append(get_symbol_bra(next(_iterator_bra)))
                    
                    _wire_chars_ket[wire].append(_out_axes_ket[-1])
                    _wire_chars_bra[wire].append(_out_axes_bra[-1])
                    continue
                
                for _get_symbol, _iterator, _in_axes, _out_axes, _wire_chars in zip(
                    (get_symbol_ket, get_symbol_bra),
                    (_iterator_ket, _iterator_bra),
                    (_in_axes_ket, _in_axes_bra),
                    (_out_axes_ket, _out_axes_bra),
                    (_wire_chars_ket, _wire_chars_bra),
                ):
                    
                    if isinstance(op, AbstractPureState):
                        _in_axis = ""
                        _out_axis = _get_symbol(next(_iterator))
                        
                    elif isinstance(op, AbstractGate):
                        _in_axis = _wire_chars[wire][-1]
                        _out_axis = _get_symbol(next(_iterator))

                    elif isinstance(op, AbstractChannel):
                        _in_axis = _wire_chars[wire][-1]
                        _out_axis = _get_symbol(next(_iterator))

                    elif isinstance(op, AbstractMeasurement):
                        _in_axis = _wire_chars[wire][-1]
                        _out_axis = ""

                    else:
                        raise TypeError

                    _in_axes.append(_in_axis)
                    _out_axes.append(_out_axis)
                    _wire_chars[wire].append(_out_axes[-1])

            
            # add extra axis for channel (i.e. sum along Kraus operators)
            if isinstance(op, AbstractChannel):
                symbol = get_symbol_channel(next(_iterator_channel))
                _axes_ket.insert(0, symbol)
                _axes_bra.insert(0, symbol)

            
            if isinstance(op, AbstractMixedState):
                _in_axes = _in_axes_ket + _in_axes_bra 
                _out_axes = _out_axes_ket + _out_axes_bra
                _in_subscripts.append("".join(_in_axes) + "".join(_out_axes))
            else:
                _in_subscripts.append("".join(_in_axes_ket) + "".join(_out_axes_ket))
                _in_subscripts.append("".join(_in_axes_bra) + "".join(_out_axes_bra))
                
            
            # pprint(_in_axes_ket)
            # pprint(_out_axes_ket)
            # pprint(_wire_chars_ket)
            
            
        #     _in_axes.append("".join(_axis))

        _out_subscripts = "".join(
            [val[-1] for key, val in _wire_chars_ket.items()] 
            + [val[-1] for key, val in _wire_chars_bra.items()]
        )
        print(_in_subscripts)
        print(_out_subscripts)
        
        # _in_expr = ",".join(_in_axes)
        # _out_expr = "".join(_out_axes)
        _subscripts = f"{','.join(_in_subscripts)}->{_out_subscripts}"
        return _in_subscripts, _out_subscripts, _subscripts

 #%%
_in_subscripts, _out_subscripts, _subscripts = subscripts_nonunitary(circuit)
#%%
tensors = evaluate(circuit)
[tensor.shape for tensor in tensors]

#%%
jnp.einsum(_subscripts, *tensors)

#%%
    import itertools
    from rich.pretty import pprint
    n_phases = 1
    wires_star = tuple(i for i in range(n_phases+1))
    wires_lab = tuple(i for i in range(n_phases+1, 2*(n_phases+1)))

    circuit = Circuit(backend='nonunitary')
    circuit.add(
        TwoModeWeakCoherentSource(wires=(0, 1), epsilon=0.01, g=1.0, phi=0.2)
    )
    # circuit.add(
    #     FockState(
    #         wires=wires_star,
    #         n=[(1.0, tuple(1 if i == j else 0 for i in wires_star)) for j in wires_star]
    #     )
    # )
    circuit.add(
        FockState(
            wires=wires_lab,
            n=[(1.0, tuple(1 if i == j else 0 for i in wires_lab)) for j in wires_lab]
        )
    )
    for i in range(1, n_phases+1):
        circuit.add(Phase(wires=(i,), phi=0.1), f"phase{i}") 

    for wire_star, wire_lab in zip(wires_star, wires_lab):
        circuit.add(BeamSplitter(wires=(wire_star, wire_lab)))
    pprint(circuit)
    
    #%%
    params, static = eqx.partition(circuit, eqx.is_inexact_array)
    
    def evaluate(circuit):
        # circuit = eqx.combine(params, static)
        _tensors = []
        for op in circuit.unwrap():
            print(op)
            _tensor = op(2)
            # print(_tensor)
            if isinstance(op, AbstractMixedState):
                _tensors.append(_tensor)
            else:
                _tensors.append(_tensor)
                _tensors.append(jnp.conjugate(_tensor))
        return _tensors
    
    tensors = evaluate(circuit)
    [tensor.shape for tensor in tensors]
    
    #%%
    # jax.jit(evaluate)(circuit)
    #%%
    subscripts_nonunitary(circuit)

    # _in_expr_ket, _out_expr_ket = _subscripts_left_right(
    #     circuit, get_symbol_right, get_symbol_channel
    # )
    # _in_expr_bra, _out_expr_bra = _subscripts_left_right(
    #     circuit, get_symbol_left, get_symbol_channel
    # )
    # _subscripts = f"{_in_expr_ket},{_in_expr_bra}->{_out_expr_ket}{_out_expr_bra}"
    # return _subscripts

    # %%

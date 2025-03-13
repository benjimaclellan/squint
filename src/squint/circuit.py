# %%
import copy
import functools
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Union, Sequence, Literal
import itertools 
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import paramax
from beartype import beartype
from jaxtyping import PyTree
from loguru import logger
from opt_einsum.parser import get_symbol

from squint.ops.base import (
    AbstractGate,
    AbstractMeasurement,
    AbstractChannel,
    AbstractOp,
    AbstractState,
    characters,
)
from squint.ops.fock import fock_subtypes
from squint.simulator import classical_fisher_information_matrix, quantum_fisher_information_matrix, Simulator, SimulatorClassicalProbability, SimulatorQuantumAmplitude


class Circuit(eqx.Module):
    ops: OrderedDict[Union[str, int], AbstractOp]

    @beartype
    def __init__(
        self,
    ):
        self.ops = OrderedDict()

    @property
    def wires(self):
        return set(sum((op.wires for op in self.unwrap()), ()))

    @beartype
    def add(self, op: AbstractOp, key: str = None):  # todo:
        if key is None:
            key = len(self.ops)
        self.ops[key] = op

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
        if any([isinstance(op, AbstractChannel) for op in self.ops]):
            return "nonunitary"
        else: 
            return "unitary"
    
    @property
    def subscripts(self):
        if self.backed == "nonunitary":
            return subscripts_nonunitary(self)
        elif self.backed == "unitary":
            return subscripts_unitary(self)
        
    @beartype
    def path(self, dim: int, optimize: str = "greedy", backend: Optional[Literal['unitary', 'nonunitary']] = None):
        if backend is None:
            backend = self.backend
            
        path, info = jnp.einsum_path(
            self.subscripts,
            # *(op(dim=dim) for op in self.unwrap()),
            self.evaluate(dim=dim, backend=backend),
            optimize=optimize,
        )
        return path, info
    
    @beartype
    def evaluate(self, dim: int, backend: Literal['unitary', 'nonunitary']):
        if backend == 'unitary':
            return [op(dim=dim) for op in circuit.unwrap()]
        elif backend == 'nonunitary':
            # unconjugated/right + conj/left direction of tensor network
            return [op(dim=dim) for op in circuit.unwrap()] + [op(dim=dim).conj() for op in circuit.unwrap()]
             
        
    @beartype
    def compile(self, params, static, dim: int, optimize: str = "greedy"):
        path, info = self.path(dim=dim, optimize=optimize)
        logger.debug(info)
        
        def _tensor_func(circuit, dim: int, subscripts: str, path: tuple, backend: Literal['unitary', 'nonunitary']):
        # def _tensor_func(ops: list, subscripts: str, optimize: tuple):
            return jnp.einsum(
                subscripts,
                # *ops,
                *circuit.evaluate(dim=dim, backend=backend),
                # *(op(dim=dim) for op in circuit.unwrap(backend=backend)),
                optimize=path,
            )
            
        
        _tensor = functools.partial(
            _tensor_func, dim=dim, subscripts=self.subscripts, path=path, backend=self.backend,
        )

        def _forward_state_func(params: PyTree, static: PyTree):
            circuit = paramax.unwrap(eqx.combine(params, static))
            return _tensor(circuit)

        _forward_state = functools.partial(_forward_state_func, static=static)

        def _forward_prob(params: PyTree):
            return jnp.abs(_forward_state(params)) ** 2

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





#%%
def subscripts_unitary(circuit: Circuit, get_symbol: Callable):
    """ Subscripts for pure state evolution """
    _iterator = itertools.count(0)
    
    _left_axes = []
    _right_axes = []
    _wire_chars = {wire: [] for wire in circuit.wires}
    for op in circuit.unwrap():
        _axis = []
        for wire in op.wires:
            if isinstance(op, AbstractState):
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
    

def subscripts_nonunitary(circuit: Circuit):
    def _subscripts_left_right(circuit: Circuit, get_symbol: Callable, get_symbol_channel: Callable):
        _iterator = itertools.count(0)
        _iterator_channel = itertools.count(0)

        _in_axes = []
        _out_axes = []
        _wire_chars = {wire: [] for wire in circuit.wires}
        for op in circuit.unwrap():
            _axis = []
            for wire in op.wires:
                if isinstance(op, AbstractState):
                    _in_axis = ""
                    _out_axes = get_symbol(next(_iterator))

                elif isinstance(op, AbstractGate):
                    _in_axis = _wire_chars[wire][-1]
                    _out_axes = get_symbol(next(_iterator))

                elif isinstance(op, AbstractChannel):
                    _in_axis = _wire_chars[wire][-1]
                    _out_axes = get_symbol(next(_iterator))

                elif isinstance(op, AbstractMeasurement):
                    _in_axis = _wire_chars[wire][-1]
                    _right_axis = ""
                
                else:
                    raise TypeError

                _axis += [_in_axis, _out_axes]

                _wire_chars[wire].append(_out_axes)

            # add extra axis for channel
            if isinstance(op, AbstractChannel):
                _axis.insert(0, get_symbol_channel(next(_iterator_channel)))
                
            _in_axes.append("".join(_axis))

        _out_axes = [val[-1] for key, val in _wire_chars.items()]

        _in_expr = ",".join(_in_axes)
        _out_expr = "".join(_out_axes)
        _subscripts = f"{_in_expr}->{_out_expr}"
        return _in_expr, _out_expr

    # START_RIGHT = 0
    # START_LEFT = 10000
    START_CHANNEL = 20000

    def get_symbol_right(i):
        # assert i + START_RIGHT < START_LEFT, "Collision of leg symbols"
        return get_symbol(2*i)
        # return get_symbol(i + START_RIGHT)

    def get_symbol_left(i):
        # assert i + START_LEFT < START_CHANNEL, "Collision of leg symbols"
        return get_symbol(2*i + 1)
        # return get_symbol(i + START_LEFT)

    def get_symbol_channel(i):
        # assert i + START_CHANNEL < START_LEFT, "Collision of leg symbols"
        return get_symbol(2 * i + START_CHANNEL)
    
    _in_expr_ket, _out_expr_ket = _subscripts_left_right(circuit, get_symbol_right, get_symbol_channel)
    _in_expr_bra, _out_expr_bra = _subscripts_left_right(circuit, get_symbol_left, get_symbol_channel)
    _subscripts = f"{_in_expr_ket},{_in_expr_bra}->{_out_expr_ket}{_out_expr_bra}"

# _tensors_ket = [op(dim=dim) for op in circuit.unwrap()]
# _tensors_bra = [op(dim=dim).conj() for op in circuit.unwrap()]


# print(_in_expr_ket, _out_expr_ket)
# print(_in_expr_bra, _out_expr_bra)
# _subscripts = f"{_in_expr_ket},{_in_expr_bra}->{_out_expr_ket}{_out_expr_bra}"
# print(_subscripts)
# # %%

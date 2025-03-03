# %%
import copy
import functools
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Union

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import paramax
from beartype import beartype
from jaxtyping import PyTree
from loguru import logger

from squint.ops.base import (
    AbstractGate,
    AbstractMeasurement,
    AbstractOp,
    AbstractState,
    characters,
)
from squint.ops.fock import fock_subtypes


class Circuit(eqx.Module):
    # ops: Sequence[AbstractOp]
    ops: OrderedDict[Union[str, int], AbstractOp]

    @beartype
    def __init__(
        self,
    ):
        self.ops = OrderedDict()

    @property
    def wires(self):
        return set(sum((op.wires for op in self.unwrap()), ()))
        # return set(sum((op.wires for op in self.ops.values()), ()))

    @beartype
    def add(self, op: AbstractOp, key: str = None):  # todo:
        if key is None:
            key = len(self.ops)
        self.ops[key] = op

    @property
    def subscripts(self):
        chars = copy.copy(characters)
        _left_axes = []
        _right_axes = []
        _wire_chars = {wire: [] for wire in self.wires}
        for op in self.unwrap():
            # for op in self.ops.values():
            _axis = []
            for wire in op.wires:
                if isinstance(op, AbstractState):
                    _left_axis = ""
                    _right_axes = chars[0]
                    chars = chars[1:]

                elif isinstance(op, AbstractGate):
                    _left_axis = _wire_chars[wire][-1]
                    _right_axes = chars[0]
                    chars = chars[1:]

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

    def unwrap(self):
        # return [op for op_wrapped in self.ops.values() for op in op_wrapped.unwrap()]
        return tuple(
            op for op_wrapped in self.ops.values() for op in op_wrapped.unwrap()
        )

    def verify(self):
        circuit_subtypes = set(map(type, self.ops.values()))
        if circuit_subtypes == fock_subtypes:
            logger.info("Circuit is contains only Fock space components.")

    @beartype
    def path(self, dim: int, optimize: str = "greedy"):
        path, info = jnp.einsum_path(
            self.subscripts,
            *(op(dim=dim) for op in self.unwrap()),
            optimize=optimize,
        )

        return path, info

    @beartype
    def compile(self, params, static, dim: int, optimize: str = "greedy"):
        path, info = self.path(dim=dim, optimize=optimize)
        logger.info(info)

        def _tensor_func(circuit, subscripts: str, optimize: tuple):
            return jnp.einsum(
                subscripts,
                *(op(dim=dim) for op in circuit.unwrap()),
                optimize=optimize,
            )

        _tensor = functools.partial(
            _tensor_func, subscripts=self.subscripts, optimize=path
        )

        def _forward_func(params, static):
            circuit = paramax.unwrap(eqx.combine(params, static))
            return _tensor(circuit)

        _forward = functools.partial(_forward_func, static=static)

        def _probability(params):
            state = _forward(params)
            return jnp.abs(state) ** 2

        _grad = jax.jacrev(_probability)
        _hess = jax.jacfwd(_grad)

        return Simulator(
            forward=_forward,
            prob=_probability,
            grad=_grad,
            hess=_hess,
            path=path,
            info=info,
        )





@dataclass
class Simulator:
    forward: Callable
    prob: Callable
    grad: Callable
    hess: Callable
    path: Any
    info: str = None

    def jit(self):
        return Simulator(
            forward=jax.jit(self.forward),
            prob=jax.jit(self.prob),
            grad=jax.jit(self.grad),
            hess=jax.jit(self.hess),
            path=self.path,
            info=self.info,
        )

    def sample(self, key: jr.PRNGKey, params: PyTree, shape: tuple[int, ...]):
        pr = self.prob(params)
        idx = jnp.nonzero(pr)
        samples = einops.rearrange(
            jr.choice(key=key, a=jnp.stack(idx), p=pr[idx], shape=shape, axis=1),
            "s ... -> ... s",
        )
        return samples


# %%

# %%
import copy
import functools
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Union

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

        # _hess = jax.jacfwd(_grad)

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


@dataclass
class SimulatorQuantumAmplitude:
    forward: Callable
    grad: Callable
    # hess: Callable
    qfim: Callable

    def jit(self):
        return SimulatorQuantumAmplitude(
            forward=jax.jit(self.forward),
            grad=jax.jit(self.grad),
            # hess=jax.jit(self.hess),
            qfim=jax.jit(self.qfim, static_argnames=("get",)),
        )


def _quantum_fisher_information_matrix(
    get: Callable, amplitudes: PyTree, grads: PyTree
):
    _grads = get(grads)
    _grads_conj = jnp.conjugate(_grads)
    return 4 * jnp.real(
        jnp.real(jnp.einsum("i..., j... -> ij", _grads_conj, _grads))
        + jnp.einsum(
            "i,j->ij",
            jnp.einsum("i..., ... -> i", _grads_conj, amplitudes),
            jnp.einsum("j..., ... -> j", _grads_conj, amplitudes),
        )
    )


def quantum_fisher_information_matrix(
    _forward_amplitudes: Callable,
    _grad_amplitudes: Callable,
    get: Callable,
    params: PyTree,
):
    amplitudes = _forward_amplitudes(params)
    grads = _grad_amplitudes(params)
    return _quantum_fisher_information_matrix(get, amplitudes, grads)


@dataclass
class SimulatorClassicalProbability:
    forward: Callable
    grad: Callable
    # hess: Callable
    cfim: Callable

    def jit(self):
        return SimulatorClassicalProbability(
            forward=jax.jit(self.forward),
            grad=jax.jit(self.grad),
            cfim=jax.jit(self.cfim, static_argnames=("get",)),
            # hess=jax.jit(self.hess),
        )


def _classical_fisher_information_matrix(get: Callable, probs: PyTree, grads: PyTree):
    return jnp.einsum(
        "i..., j... -> ij",
        get(grads),
        jnp.nan_to_num(get(grads) / probs[None, ...], 0.0),
    )
    # return jnp.einsum("i..., j..., ... -> ij", get(grads), get(grads), 1 / probs)


def classical_fisher_information_matrix(
    _forward_prob: Callable,
    _grad_prob: Callable,
    get: Callable,
    params: PyTree,
):
    probs = _forward_prob(params)
    grads = _grad_prob(params)
    return _classical_fisher_information_matrix(get, probs, grads)


@dataclass
class Simulator:
    amplitudes: SimulatorQuantumAmplitude
    prob: SimulatorClassicalProbability
    path: Any
    info: str = None

    def jit(self):
        return Simulator(
            amplitudes=self.amplitudes.jit(),
            prob=self.prob.jit(),
            path=self.path,
            info=self.info,
        )

    def sample(self, key: jr.PRNGKey, params: PyTree, shape: tuple[int, ...]):
        pr = self.prob.forward(params)
        idx = jnp.nonzero(pr)
        samples = einops.rearrange(
            jr.choice(key=key, a=jnp.stack(idx), p=pr[idx], shape=shape, axis=1),
            "s ... -> ... s",
        )
        return samples


# %%

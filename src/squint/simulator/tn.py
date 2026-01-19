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
from __future__ import annotations

import functools
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Type, Union

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import paramax
from beartype import beartype
from jaxtyping import Array, PyTree
from opt_einsum.parser import get_symbol

__all__ = ["SimulatorQuantumAmplitudes", "SimulatorClassicalProbabilities", "Simulator"]

from squint.circuit import Circuit
from squint.ops import (
    AbstractErasureChannel,
    AbstractGate,
    AbstractKrausChannel,
    AbstractMeasurement,
    AbstractMixedState,
    AbstractPureState,
)
from squint.ops.base import Block


class AbstractBackend(ABC):
    @staticmethod
    @abstractmethod
    def evaluate(obj: Union[Circuit, Block]) -> Sequence[ArrayLike]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def subscripts(obj: Union[Circuit, Block]) -> str:
        raise NotImplementedError


@dataclass
class Simulator:
    """
    Simulator for quantum circuits, providing callable methods for computing
    forward, backward, and Fisher Information matrix calculations on the
    quantum amplitudes and classical probabilities, given a set of parameters PyTrees

    Attributes:
        amplitudes (SimulatorQuantumAmplitudes): Object for quantum amplitudes computations.
        probabilities (SimulatorClassicalProbabilities): Object for classical probabilities computations.
        path (Any): Path to the simulator, can be used for saving/loading.
        info (str, optional): Additional information about the simulator.
    """

    circuit: Circuit
    backend: AbstractBackend

    amplitudes: SimulatorQuantumAmplitudes
    probabilities: SimulatorClassicalProbabilities

    path: Any
    info: str = None

    @beartype
    @classmethod
    def compile(
        cls,
        static: PyTree,
        *params,
        **kwargs,
    ):
        """
        Compiles the circuit into a tensor contraction function.

        Args:
            static (PyTree): The static PyTree, following the `equinox` convention. These are parameters that are fixed.
            # dim (int): The dimension of the local Hilbert space (the same dimension across all wires).
            params (Sequence[PyTree]): The parameterized PyTree, following the `equinox` convention. These are parameters that will be used in gradient and Fisher information calculations.

        Returns:
            sim (Simulator): A class which contains methods for computing the parameterized forward, grad, and Fisher information functions.
        """

        circuit = paramax.unwrap(functools.reduce(eqx.combine, (static,) + params))
        backend = _select_backend(circuit)

        def _tensor_func(
            circuit,
            subscripts: str,
            path: tuple,
            backend: AbstractBackend,
        ):
            return jnp.einsum(
                subscripts,
                *jtu.tree_map(
                    lambda x: x.astype(dtype_complex),
                    backend.evaluate(circuit),
                ),
                optimize=path,
            )

        optimize = kwargs.get("optimize", "greedy")
        argnum = kwargs.get("argnum", 0)

        dtype_complex = jnp.complex128  # TODO: Add to config

        subscripts = backend.subscripts(circuit)
        path, info = _path(circuit, backend, optimize=optimize)

        wires = circuit.wires
        wires_ptrace = set(
            sum(
                (
                    op.wires
                    for op in circuit.unwrap()
                    if isinstance(op, AbstractErasureChannel)
                ),
                (),
            )
        )

        _tensor = functools.partial(
            _tensor_func,
            subscripts=subscripts,
            path=path,
            backend=backend,
        )

        def _forward_state_func(static: PyTree, *params):
            circuit = paramax.unwrap(functools.reduce(eqx.combine, (static,) + params))
            return _tensor(circuit)

        _forward_state = functools.partial(_forward_state_func, static)

        if backend is PureBackend:

            def _forward_prob(*params: Sequence[PyTree]):
                return jnp.abs(_forward_state(*params)) ** 2

        elif backend is MixedBackend:

            def _forward_prob(*params: Sequence[PyTree]):
                # remove wires that have been traced out
                _subscripts_tmp = [
                    get_symbol(i) for i in range(len(wires - wires_ptrace))
                ]
                _subscripts = (
                    "".join(_subscripts_tmp + _subscripts_tmp)
                    + "->"
                    + "".join(_subscripts_tmp)
                )
                return jnp.abs(jnp.einsum(_subscripts, _forward_state(*params)))
        else:
            raise RuntimeError("Backend not found or provided.")

        _grad_state_holomorphic = jax.jacfwd(
            _forward_state, argnums=argnum, holomorphic=True
        )
        _grad_prob = jax.jacfwd(_forward_prob, argnums=argnum)

        # _grad_state_holomorphic = jax.jacrev(
        #     _forward_state, argnums=argnum, holomorphic=True
        # )
        # _grad_prob = jax.jacrev(_forward_prob, argnums=argnum)

        def _grad_state(*params: Sequence[PyTree]):
            params = jtu.tree_map(lambda x: x.astype(dtype_complex), params)
            return _grad_state_holomorphic(*params)

        if backend is PureBackend:
            _qfim_state = functools.partial(
                quantum_fisher_information_matrix, _forward_state, _grad_state
            )

        elif backend is MixedBackend:

            def _qfim_state(*params):
                raise NotImplementedError("QFIM for mixed states not implemented")

        else:
            raise RuntimeError("Backend not found or provided.")

        _cfim_state = functools.partial(
            classical_fisher_information_matrix, _forward_prob, _grad_prob
        )

        return cls(
            circuit=circuit,
            backend=backend,
            amplitudes=SimulatorQuantumAmplitudes(
                forward=_forward_state,
                grad=_grad_state,
                qfim=_qfim_state,
            ),
            probabilities=SimulatorClassicalProbabilities(
                forward=_forward_prob,
                grad=_grad_prob,
                cfim=_cfim_state,
            ),
            path=path,
            info=info,
        )

    def jit(self, device: jax.Device = None):
        """
        JIT (just-in-time) compile the simulator methods.
        Args:
            device (jax.Device, optional): Device to compile the methods on. Defaults to None, which uses the first available device.
        """
        if not device:
            device = jax.devices()[0]

        return Simulator(
            circuit=self.circuit,
            backend=self.backend,
            amplitudes=self.amplitudes.jit(device=device),
            probabilities=self.probabilities.jit(device=device),
            path=self.path,
            info=self.info,
        )

    def sample(self, key: jr.PRNGKey, params: PyTree, shape: tuple[int, ...]):
        """
        Sample from the quantum circuit using the provided parameters and a random key.
        Args:
            key (jr.PRNGKey): Random key for sampling.
            params (PyTree): Parameters for the quantum circuit, partitioned via `eqx.partition`.
            shape (tuple[int, ...]): Shape of the output samples.
        Returns:
            samples (jnp.ndarray): Samples drawn from the quantum circuit.
        """
        pr = self.probabilities.forward(params)
        idx = jnp.nonzero(pr)
        samples = einops.rearrange(
            jr.choice(key=key, a=jnp.stack(idx), p=pr[idx], shape=shape, axis=1),
            "s ... -> ... s",
        )
        return samples


# %%
@beartype
def _path(
    circuit: Circuit,
    backend: type[AbstractBackend],
    optimize: str = "greedy",
):
    """
    Computes the einsum contraction path using the `opt_einsum` algorithm.

    Args:
        optimize (str): The argument to pass to `opt_einsum` for computing the optimal contraction path. Defaults to `greedy`.
    """

    path, info = jnp.einsum_path(
        backend.subscripts(circuit),
        *backend.evaluate(circuit),
        optimize=optimize,
    )
    return path, info


@beartype
def _select_backend(circuit: Circuit) -> Type[AbstractBackend]:
    if any(
        [
            isinstance(
                op,
                (
                    AbstractMixedState,
                    AbstractKrausChannel,
                    AbstractErasureChannel,
                ),
            )
            for op in circuit.unwrap()
        ]
    ):
        return MixedBackend
    else:
        return PureBackend


# TODO: better verification system for composing checks
def verify(self):
    """
    Performs a verification check on the circuit object to ensure it is valid prior to being compiled.
    """
    grid = {}
    for op in self.unwrap():
        for wire in op.wires:
            if wire not in grid.keys():
                grid[wire] = []
            grid[wire].append(op)

    # check that the first op on each wire is an AbstractState, and no others are AbstractState ops
    for wire, ops in grid.items():
        if not isinstance(ops[0], (AbstractPureState, AbstractMixedState)):
            raise RuntimeError(
                f"The first op on wire {wire} is of type {type(ops[0])}"
                "The first op on each wire must be a subtype of `AbstractPureState` or `AbstractMixedState"
            )
        if any(
            [isinstance(op, (AbstractPureState, AbstractMixedState)) for op in ops[1:]]
        ):
            raise RuntimeError(
                f"Wire {wire} contains multiple `AbstractState` ops."
                "Only the first op on each wire can be a subtype of `AbstractPureState` or `AbstractMixedState"
            )

    # check that we are using the correct backend
    if any(
        [
            isinstance(
                op,
                (AbstractKrausChannel, AbstractErasureChannel, AbstractMixedState),
            )
            for op in self.unwrap()
        ]
    ):
        _backend = "mixed"
        if self.backend != _backend:
            raise RuntimeError(
                "Backend must be `mixed` as the circuit contains one or more `AbstractChannel` and/or `AbstractMixedState`"
            )
    else:
        _backend = "pure"
        if self.backend != _backend:
            warnings.warn(
                f"Circuit backend is set to `{self.backend}`; however the circuit is `pure`."
                "Consider switching the backend to `pure`.",
                UserWarning,
                stacklevel=2,
            )


class PureBackend(AbstractBackend):
    @staticmethod
    def evaluate(obj: Union[Circuit, Block]) -> Sequence[ArrayLike]:
        return [op() for op in obj.unwrap()]

    @staticmethod
    def subscripts(obj: Union[Circuit, Block]) -> str:
        """
        Generate einsum subscript string for pure state tensor network contraction.

        Iterates through all operations in the circuit/block and assigns unique
        character indices to each tensor leg. Input and output indices are tracked
        per wire to construct the full einsum expression for contracting the
        tensor network.

        Args:
            obj: A Circuit or Block containing quantum operations.

        Returns:
            str: An einsum subscript string in the format "input1,input2,...->output"
                suitable for use with jnp.einsum.

        Raises:
            RuntimeError: If a gate is applied to a wire before a state is initialized.
            TypeError: If an unknown operation type is encountered.
        """

        _iterator = itertools.count(0)
        _wire_chars = {wire: [] for wire in obj.wires}
        _in_subscripts = []
        _get_symbol = get_symbol

        for op in obj.unwrap():
            _in_axes = []
            _out_axes = []
            for wire in op.wires:
                # construct the indices for both the right and left (ket and bra) operators

                if isinstance(op, AbstractPureState):
                    _in_axes.append("")
                    _out_axes.append(_get_symbol(next(_iterator)))
                    _wire_chars[wire].append(_out_axes[-1])

                elif isinstance(op, AbstractGate):
                    if len(_wire_chars[wire]) == 0 and isinstance(obj, Circuit):
                        raise RuntimeError(
                            f"Wire {wire} has no input state before gate {op}. The first op on each wire must be a subtype of `AbstractPureState` or `AbstractMixedState`"
                        )
                    elif len(_wire_chars[wire]) == 0 and isinstance(obj, Block):
                        _symbol = _get_symbol(next(_iterator))
                        _in_axes.append(_symbol)
                        _wire_chars[wire].append(_symbol)

                    else:
                        _in_axes.append(_wire_chars[wire][-1])

                    _out_axes.append(_get_symbol(next(_iterator)))
                    _wire_chars[wire].append(_out_axes[-1])

                elif isinstance(op, AbstractMeasurement):
                    _in_axis = _wire_chars[wire][-1]
                    _out_axis = ""

                else:
                    raise TypeError

            _in_subscripts.append("".join(_in_axes) + "".join(_out_axes))
        # print(_in_axes, _out_axes)

        if isinstance(obj, Circuit):
            _out_subscripts = "".join([val[-1] for key, val in _wire_chars.items()])
            # _subscripts = f"{','.join(_in_subscripts)}->{_out_subscripts}"

        elif isinstance(obj, Block):
            # if Block has no input states, it should be an operator
            _out_subscripts = "".join(
                [val[0] for key, val in _wire_chars.items()]
                + [val[-1] for key, val in _wire_chars.items()]
            )

        _subscripts = f"{','.join(_in_subscripts)}->{_out_subscripts}"
        return _subscripts


class MixedBackend(AbstractBackend):
    @staticmethod
    def evaluate(obj: Union[Circuit, Block]) -> Sequence[ArrayLike]:
        _tensors = []
        for op in obj.unwrap():
            _tensor = op()
            if isinstance(op, AbstractMixedState):
                _tensors.append(_tensor)
            else:
                # unconjugated/right + conj/left direction of tensor network, sequential in the list
                _tensors.append(_tensor)
                _tensors.append(jnp.conjugate(_tensor))
        return _tensors

    @staticmethod
    def subscripts(obj: Union[Circuit, Block]) -> str:
        """
        Assigns the indices for all tensor legs when the circuit is includes mixed states, channels, and non-unitary evolution.

        The canonical ordering of indices is (input_indices, output_indices)
        """
        START_CHANNEL = 50000

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

        _wire_chars_ket = {wire: [] for wire in obj.wires}
        _wire_chars_bra = {wire: [] for wire in obj.wires}

        _in_subscripts = []

        for op in obj.unwrap():
            _in_axes_ket = []
            _in_axes_bra = []
            _out_axes_ket = []
            _out_axes_bra = []

            for wire in op.wires:
                if isinstance(op, AbstractMixedState):
                    _in_axes_ket.append("")
                    _in_axes_bra.append("")
                    _out_axes_ket.append(get_symbol_ket(next(_iterator_ket)))
                    _out_axes_bra.append(get_symbol_bra(next(_iterator_bra)))

                    _wire_chars_ket[wire].append(_out_axes_ket[-1])
                    _wire_chars_bra[wire].append(_out_axes_bra[-1])
                    continue

                elif isinstance(op, AbstractErasureChannel):
                    _ptrace_axis = get_symbol_channel(next(_iterator_channel))

                # construct the indices for both the right and left (ket and bra) operators
                for _get_symbol, _iterator, _in_axes, _out_axes, _wire_chars in zip(
                    (get_symbol_ket, get_symbol_bra),
                    (_iterator_ket, _iterator_bra),
                    (_in_axes_ket, _in_axes_bra),
                    (_out_axes_ket, _out_axes_bra),
                    (_wire_chars_ket, _wire_chars_bra),
                    strict=False,
                ):
                    if isinstance(op, AbstractPureState):
                        _in_axes.append("")
                        _out_axes.append(_get_symbol(next(_iterator)))
                        _wire_chars[wire].append(_out_axes[-1])

                    elif isinstance(op, (AbstractGate, AbstractKrausChannel)):
                        _in_axes.append(_wire_chars[wire][-1])
                        _out_axes.append(_get_symbol(next(_iterator)))
                        _wire_chars[wire].append(_out_axes[-1])

                    elif isinstance(op, AbstractErasureChannel):
                        _in_axis = _wire_chars[wire][-1]
                        _out_axis = _ptrace_axis

                        _in_axes.append(_wire_chars[wire][-1])
                        _out_axes.append(_ptrace_axis)
                        _wire_chars[wire].append("")

                    elif isinstance(op, AbstractMeasurement):
                        _in_axis = _wire_chars[wire][-1]
                        _out_axis = ""

                    else:
                        raise TypeError

            # add extra axis for channel (i.e. sum along Kraus operators)
            if isinstance(op, AbstractKrausChannel):
                symbol = get_symbol_channel(next(_iterator_channel))
                # _axes_ket.insert(0, symbol)
                # _axes_bra.insert(0, symbol)
                _in_axes_ket.insert(0, symbol)
                _in_axes_bra.insert(0, symbol)

            if isinstance(op, AbstractMixedState):
                _in_axes = _in_axes_ket + _in_axes_bra
                _out_axes = _out_axes_ket + _out_axes_bra
                _in_subscripts.append("".join(_in_axes) + "".join(_out_axes))
            else:
                _in_subscripts.append("".join(_in_axes_ket) + "".join(_out_axes_ket))
                _in_subscripts.append("".join(_in_axes_bra) + "".join(_out_axes_bra))

        _out_subscripts = "".join(
            [val[-1] for key, val in _wire_chars_ket.items()]
            + [val[-1] for key, val in _wire_chars_bra.items()]
        )
        _subscripts = f"{','.join(_in_subscripts)}->{_out_subscripts}"
        return _subscripts


@dataclass
class SimulatorQuantumAmplitudes:
    """
    Simulator object which computes quantities related to the quantum probability amplitudes,
    including forward pass, gradient computation,
    and quantum Fisher information matrix calculation.

    Attributes:
        forward (Callable): Function to compute quantum amplitudes.
        grad (Callable): Function to compute gradients of quantum amplitudes.
        qfim (Callable): Function to compute the quantum Fisher information matrix.
    """

    forward: Callable
    grad: Callable
    qfim: Callable

    def jit(self, device: jax.Device = None):
        """
        JIT (just-in-time) compile the simulator methods.
        Args:
            device (jax.Device, optional): Device to compile the methods on. Defaults to None, which uses the first available device.
        """
        return SimulatorQuantumAmplitudes(
            forward=jax.jit(self.forward, device=device),
            grad=jax.jit(self.grad, device=device),
            qfim=jax.jit(self.qfim, device=device),
            # qfim=jax.jit(self.qfim, static_argnames=("get",), device=device),
        )


def _quantum_fisher_information_matrix(
    # get: Callable,
    amplitudes: Array,
    grads: Array,
):
    """
    Computes the quantum Fisher information matrix from the already computed arrays representing
    the probability amplitudes and their gradients.

    Args:
        amplitudes (Array): Quantum amplitudes.
        grads (Array): Gradients of the quantum amplitudes.

    Returns:
        qfim (jnp.ndarray): Quantum Fisher information matrix.
    """
    _grads = grads
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
    # get: Callable,
    *params: PyTree,
):
    """
    Performs the forward pass to compute quantum amplitudes and their gradients,
    and then calculates the quantum Fisher information matrix.
    Args:
        _forward_amplitudes (Callable): Function to compute quantum amplitudes.
        _grad_amplitudes (Callable): Function to compute gradients of quantum amplitudes.
        *params (list[PyTree]): Parameters for the quantum circuit, partitioned via `eqx.partition`.
            The argnum is already defined in the callables
    Returns:
        qfim (jnp.ndarray): Quantum Fisher information matrix."""
    amplitudes = _forward_amplitudes(*params)
    grads, _ = jax.tree.flatten(_grad_amplitudes(*params))
    grads = jnp.stack(grads, axis=0)
    return _quantum_fisher_information_matrix(amplitudes, grads)


@dataclass
class SimulatorClassicalProbabilities:
    """
    Simulator object which computes quantities related to the classical probabilities,
    including forward pass, gradient computation,
    and classical Fisher information matrix calculation.

    Attributes:
        forward (Callable): Function to compute classical probabilities.
        grad (Callable): Function to compute gradients of classical probabilities.
        cfim (Callable): Function to compute the classical Fisher information matrix.
    """

    forward: Callable
    grad: Callable
    cfim: Callable

    @beartype
    def jit(self, device: jax.Device = None):
        """
        JIT (just-in-time) compile the simulator methods.
        Args:
            device (jax.Device, optional): Device to compile the methods on. Defaults to None, which uses the first available device.
        """
        return SimulatorClassicalProbabilities(
            forward=jax.jit(self.forward, device=device),
            grad=jax.jit(self.grad, device=device),
            cfim=jax.jit(self.cfim, device=device),
            # cfim=jax.jit(self.cfim, static_argnames=("get",), device=device),
        )


def _classical_fisher_information_matrix(
    probs: Array,
    grads: Array,
):
    """
    Computes the classical Fisher information matrix from the already computed arrays representing
    the probabilities and their gradients.
    Args:
        probs (Array): Classical probabilities.
        grads (Array): Gradients of the classical probabilities.
    Returns:
        cfim (jnp.ndarray): Classical Fisher information matrix.
    """

    return jnp.einsum(
        "i..., j..., ... -> ij",
        grads,
        grads,
        1
        / (probs[None, ...] + 1e-14),  # add a small constant to avoid division by zero
    )


def classical_fisher_information_matrix(
    _forward_prob: Callable,
    _grad_prob: Callable,
    # get: Callable,
    *params: PyTree,
):
    """
    Performs the forward pass to compute classical probabilities and their gradients,
    and then calculates the classical Fisher information matrix.
    Args:
        _forward_prob (Callable): Function to compute classical probabilities.
        _grad_prob (Callable): Function to compute gradients of classical probabilities.
        *params (list[PyTree]): Parameters for the quantum circuit, partitioned via `eqx.partition`.
            The argnum is already defined in the callables
    Returns:
        cfim (jnp.ndarray): Classical Fisher information matrix.
    """
    probs = _forward_prob(*params)
    grads, _ = jax.tree.flatten(_grad_prob(*params))
    grads = jnp.stack(grads, axis=0)
    return _classical_fisher_information_matrix(probs, grads)

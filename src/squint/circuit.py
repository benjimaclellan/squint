# %%
import functools
import itertools
import warnings
from collections import OrderedDict
from typing import Literal, Optional, Union

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
    AbstractErasureChannel,
    AbstractGate,
    AbstractKrausChannel,
    AbstractMeasurement,
    AbstractMixedState,
    AbstractOp,
    AbstractPureState,
)
from squint.simulator import (
    Simulator,
    SimulatorClassicalProbabilities,
    SimulatorQuantumAmplitudes,
    classical_fisher_information_matrix,
    quantum_fisher_information_matrix,
)

# %%


class Circuit(eqx.Module):
    r"""
    The `Circuit` object is a symbolic representation of a quantum circuit for qubits, qudits, or for an infinite-dimensional Fock space.
    The circuit is composed of a sequence of quantum operators on `wires` which define the evolution of the quantum

    Attributes:
        ops (OrderedDict[Union[str, int], AbstractOp]): An ordered dictionary of ops (dictionary value) with an assigned label (dictionary key).

    Example:
        ```python
        circuit = Circuit(backend='pure')
        circuit.add(DiscreteVariableState(wires=(0,)))
        circuit.add(HGate(wires=(0,)))
        ```
    """

    dims: tuple[int, ...] = None  # todo: implement dimension per wire
    ops: OrderedDict[Union[str, int], AbstractOp]
    _backend: Literal["pure", "mixed"]

    @beartype
    def __init__(self, backend: Optional[Literal["pure", "mixed"]] = None):
        """
        Initializes a quantum circuit with the specified backend type.

        Args:
            backend (Literal["pure", "mixed"]): The type of backend to use for the circuit.
            Defaults to "pure". "pure" represents a reversible quantum operation,
            while "mixed" allows for non-reversible operations.
        """
        self.ops = OrderedDict()
        self._backend = backend

    @property
    def wires(self) -> set[int]:
        """
        Initializes a quantum circuit with the specified backend type.

        Args:
            backend (Literal["pure", "mixed"]): The type of backend to use for the circuit.
            Defaults to "pure". "pure" represents a reversible quantum operation,
            while "mixed" allows for non-reversible operations.
        """
        return set(sum((op.wires for op in self.unwrap()), ()))

    @beartype
    def add(self, op: AbstractOp, key: str = None) -> None:
        """
        Add an operator to the circuit.

        Operators are added sequential along the wires. The first operator on each wire must be a state
        (a subtype of AbstractPureState or AbstractMixedState).

        Args:
            op (AbstractOp): The operator instance to add to the circuit.
            key (Optional[str]): A string key for indexing into the circuit PyTree instance. Defaults to `None` and an integer counter is used.
        """

        if key is None:
            key = len(self.ops)
        self.ops[key] = op

    def unwrap(self) -> tuple[AbstractOp]:
        """
        Unwrap all operators in the circuit by recursively calling the `op.unwrap()` method.
        """
        return tuple(
            op for op_wrapped in self.ops.values() for op in op_wrapped.unwrap()
        )

    @property
    def backend(self) -> str:
        if self._backend == None:
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
                    for op in self.unwrap()
                ]
            ):
                return "mixed"
            else:
                return "pure"
        return self._backend

    @property
    def subscripts(self) -> str:
        """
        Returns the einsum subscript expression as a string.
        """
        if self.backend == "mixed":
            return subscripts_mixed(self)
        elif self.backend == "pure":
            return subscripts_pure(self)

    @beartype
    def path(
        self,
        dim: int,
        optimize: str = "greedy",
    ):
        """
        Computes the einsum contraction path using the `opt_einsum` algorithm.

        Args:
            dim (int): The dimension of the local Hilbert space (the same dimension across all wires).
            optimize (str): The argument to pass to `opt_einsum` for computing the optimal contraction path. Defaults to `greedy`.
        """

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
        """
        Evaluates the corresponding numerical tensor for each operator in the circuit, based on the provided dimension.

        Args:
            dim (int): The dimension of the local Hilbert space (the same dimension across all wires).
        """
        if self.backend == "pure":
            return [op(dim=dim) for op in self.unwrap()]

        elif self.backend == "mixed":
            _tensors = []
            for op in self.unwrap():
                _tensor = op(dim)
                if isinstance(op, AbstractMixedState):
                    _tensors.append(_tensor)
                else:
                    # unconjugated/right + conj/left direction of tensor network, sequential in the list
                    _tensors.append(_tensor)
                    _tensors.append(jnp.conjugate(_tensor))
            return _tensors

    @beartype
    def compile(
        self, params: PyTree, static: PyTree, dim: int, optimize: str = "greedy"
    ):
        """
        Compiles the circuit into a tensor contraction function.

        Args:
            params (PyTree): The parameterized PyTree, following the `equinox` convention. These are parameters that will be used in gradient and Fisher information calculations.
            static (PyTree): The static PyTree, following the `equinox` convention. These are parameters that are fixed.
            dim (int): The dimension of the local Hilbert space (the same dimension across all wires).
            optimize (str): The argument to pass to `opt_einsum` for computing the optimal contraction path. Defaults to `greedy`.

        Returns:
            sim (Simulator): A class which contains methods for computing the parameterized forward, grad, and Fisher information functions.
        """
        self.verify()

        path, info = self.path(dim=dim, optimize=optimize)
        logger.debug(info)

        def _tensor_func(
            circuit,
            dim: int,
            subscripts: str,
            path: tuple,
            backend: Literal["pure", "mixed"],
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

        if backend == "pure":

            def _forward_prob(params: PyTree):
                return jnp.abs(_forward_state(params)) ** 2

        elif backend == "mixed":

            def _forward_prob(params: PyTree):
                # remove wires that have been traced out
                wires = self.wires - set(
                    sum(
                        (
                            op.wires
                            for op in self.unwrap()
                            if isinstance(op, AbstractErasureChannel)
                        ),
                        (),
                    )
                )
                _subscripts_tmp = [get_symbol(i) for i in range(len(wires))]
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
            amplitudes=SimulatorQuantumAmplitudes(
                forward=_forward_state,
                grad=_grad_state,
                qfim=functools.partial(
                    quantum_fisher_information_matrix, _forward_state, _grad_state
                ),
            ),
            probabilities=SimulatorClassicalProbabilities(
                forward=_forward_prob,
                grad=_grad_prob,
                cfim=functools.partial(
                    classical_fisher_information_matrix, _forward_prob, _grad_prob
                ),
            ),
            path=path,
            info=info,
        )

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
                [
                    isinstance(op, (AbstractPureState, AbstractMixedState))
                    for op in ops[1:]
                ]
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


# %%
def subscripts_pure(circuit: Circuit):
    """
    Assigns the indices for all tensor legs when the circuit is composed of only pure states and unitary evolution.
    """
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


def subscripts_mixed(circuit: Circuit):
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

    _wire_chars_ket = {wire: [] for wire in circuit.wires}
    _wire_chars_bra = {wire: [] for wire in circuit.wires}

    _in_subscripts = []

    for op in circuit.unwrap():
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
